"""
Nicholas M. Boffi

Neural networks for entropy estimation.
"""

import haiku as hk
import jax.numpy as np
import jax
from jax import vmap, lax
from typing import Optional, Tuple, Callable, Sequence, Any
from . import drifts
from . import deriv_utils


class MultiHeadAttention(hk.Module):
    """Multi-headed attention (MHA) module.
    Directly modified from source code at https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/attention.py
    to also return the attention matrix for later visualization.

    This module is intended for attending over sequences of vectors.

    Rough sketch:
    - Compute keys (K), queries (Q), and values (V) as projections of inputs.
    - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
    - Output is another projection of WV^T.

    For more detail, see the original Transformer paper:
      "Attention is all you need" https://arxiv.org/abs/1706.03762.

    Glossary of shapes:
    - T: Sequence length.
    - D: Vector (embedding) size.
    - H: Number of attention heads.
    """

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        w_init_scale: Optional[float] = None,
        *,
        w_init: Optional[hk.initializers.Initializer] = None,
        with_bias: bool = True,
        b_init: Optional[hk.initializers.Initializer] = None,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        """Initialises the module.

        Args:
          num_heads: Number of independent attention heads (H).
          key_size: The size of keys (K) and queries used for attention.
          w_init_scale: DEPRECATED. Please use w_init instead.
          w_init: Initialiser for weights in the linear map. Once `w_init_scale` is
            fully deprecated `w_init` will become mandatory. Until then it has a
            default value of `None` for backwards compatability.
          with_bias: Whether to add a bias when computing various linear
            projections.
          b_init: Optional initializer for bias. By default, zero.
          value_size: Optional size of the value projection (V). If None, defaults
            to the key size (K).
          model_size: Optional size of the output embedding (D'). If None, defaults
            to the key size multiplied by the number of heads (K * H).
          name: Optional name for this module.
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads

        # Backwards-compatibility for w_init_scale.
        if w_init_scale is not None:
            warnings.warn(
                "w_init_scale is deprecated; please pass an explicit weight "
                "initialiser instead.",
                DeprecationWarning,
            )
        if w_init and w_init_scale:
            raise ValueError("Please provide only `w_init`, not `w_init_scale`.")
        if w_init is None and w_init_scale is None:
            raise ValueError(
                "Please provide a weight initializer: `w_init`. "
                "`w_init` will become mandatory once `w_init_scale` is "
                "fully deprecated."
            )
        if w_init is None:
            w_init = hk.initializers.VarianceScaling(w_init_scale)
        self.w_init = w_init
        self.with_bias = with_bias
        self.b_init = b_init

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        mask: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.

        Args:
          query: Embeddings sequence used to compute queries; shape [..., T', D_q].
          key: Embeddings sequence used to compute keys; shape [..., T, D_k].
          value: Embeddings sequence used to compute values; shape [..., T, D_v].
          mask: Optional mask applied to attention weights; shape [..., H=1, T', T].

        Returns:
          A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., T', D'].
        """

        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        *leading_dims, sequence_length, _ = query.shape
        projection = self._linear_projection

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = projection(query, self.key_size, "query")  # [T', H, Q=K]
        key_heads = projection(key, self.key_size, "key")  # [T, H, K]
        value_heads = projection(value, self.value_size, "value")  # [T, H, V]

        # Compute attention weights.
        attn_logits = np.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        attn_logits = attn_logits / np.sqrt(self.key_size).astype(key.dtype)
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim}."
                )
            attn_logits = np.where(mask, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

        # Weight the values by the attention and flatten the head vectors.
        attn = np.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = np.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        final_projection = hk.Linear(
            self.model_size,
            w_init=self.w_init,
            with_bias=self.with_bias,
            b_init=self.b_init,
        )

        #      [H, T', T]    [T', D']
        return attn_weights, final_projection(attn)

    @hk.transparent
    def _linear_projection(
        self,
        x: jax.Array,
        head_size: int,
        name: Optional[str] = None,
    ) -> jax.Array:
        y = hk.Linear(
            self.num_heads * head_size,
            w_init=self.w_init,
            with_bias=self.with_bias,
            b_init=self.b_init,
            name=name,
        )(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))


class InducingEncoderBlock(hk.Module):
    input_dim: int
    num_heads: int
    dim_feedforward: int
    n_layers_feedforward: int
    n_inducing_ponts: int
    w0: float

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dim_feedforward: int,
        n_layers_feedforward: int,
        n_inducing_points: int,
        w0: float,
        name: str = None,
    ) -> None:
        super().__init__(name=name)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.n_layers_feedforward = n_layers_feedforward
        self.n_inducing_points = n_inducing_points
        self.w0 = w0
        self.name = name
        self.key_size = self.input_dim // self.num_heads

        # Attention layer
        assert self.input_dim % self.num_heads == 0

        # need two attns
        self.attns = [
            MultiHeadAttention(
                key_size=self.key_size,
                num_heads=self.num_heads,
                w_init=hk.initializers.VarianceScaling(),
            )
            for _ in range(2)
        ]

        # need two mlps
        self.mlps = [
            hk.Sequential(
                construct_mlp_layers(
                    self.n_layers_feedforward,
                    self.dim_feedforward,
                    jax.nn.gelu,
                    w0=w0,
                    output_dim=self.input_dim,
                )
            )
            for _ in range(2)
        ]

        # need four layer norms
        self.norms = [
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            for _ in range(4)
        ]

    def __call__(self, x: np.ndarray):
        # grab inducing points
        dtype = x.dtype
        I = hk.get_parameter(
            "I",
            [self.n_inducing_points, self.num_heads * self.key_size],
            dtype,
            init=hk.initializers.VarianceScaling(),
        )

        ## MAB(I, X)
        y = I + self.attns[0](I, x, x)[1]  # [n_inducing, d]
        y = self.norms[0](y)
        y = y + self.mlps[0](y)
        y = self.norms[1](y)

        ## MAB(X, MAB(I, X))
        x = x + self.attns[1](x, y, y)[1]  # [N, d]
        x = self.norms[2](x)
        x = x + self.mlps[1](x)
        x = self.norms[3](x)

        return x


class EncoderBlock(hk.Module):
    input_dim: int
    num_heads: int
    dim_feedforward: int
    n_layers_feedforward: int
    w0: float

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dim_feedforward: int,
        n_layers_feedforward: int,
        w0: float,
        name: str = None,
    ) -> None:
        super().__init__(name=name)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.n_layers_feedforward = n_layers_feedforward
        self.w0 = w0
        self.name = name

        # Attention layer
        assert self.input_dim % self.num_heads == 0
        self.attn = MultiHeadAttention(
            key_size=self.input_dim // self.num_heads,
            num_heads=self.num_heads,
            w_init=hk.initializers.VarianceScaling(),
        )

        # MLP layer
        self.mlp = hk.Sequential(
            construct_mlp_layers(
                self.n_layers_feedforward,
                self.dim_feedforward,
                jax.nn.gelu,
                w0=w0,
                output_dim=self.input_dim,
            )
        )

        # Layers to apply in between the main layers
        self.norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, x: np.ndarray):
        x = x + self.attn(x, x, x)[1]
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x


class Transformer(hk.Module):
    num_layers: int
    input_dim: int
    num_heads: int
    dim_feedforward: int
    n_layers_feedforward: int
    n_inducing_points: int
    w0: float

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        num_heads: int,
        dim_feedforward: int,
        n_layers_feedforward: int,
        n_inducing_points: int,
        w0: float,
        name: str = None,
    ) -> None:
        super().__init__(name=name)
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.n_layers_feedforward = n_layers_feedforward
        self.n_inducing_points = n_inducing_points
        self.w0 = w0
        self.name = name

        if n_inducing_points == 0:
            self.layers = [
                EncoderBlock(
                    self.input_dim,
                    self.num_heads,
                    self.dim_feedforward,
                    self.n_layers_feedforward,
                    self.w0,
                )
                for _ in range(self.num_layers)
            ]
        else:
            self.layers = [
                InducingEncoderBlock(
                    self.input_dim,
                    self.num_heads,
                    self.dim_feedforward,
                    self.n_layers_feedforward,
                    self.n_inducing_points,
                    self.w0,
                )
                for _ in range(self.num_layers)
            ]

        self.network = hk.Sequential(self.layers)

    def __call__(self, x):
        return self.network(x)

    def get_attention_maps(self, x):
        attention_maps = []
        for layer in self.layers:
            attention_maps.append(layer.attn(x, x, x)[0])
            x = layer(x)

        return attention_maps


class SirenInit(hk.initializers.Initializer):
    """SIREN initializer."""

    def __init__(self, is_first_layer: bool, w0: float):
        self.is_first_layer = is_first_layer
        self.w0 = w0

    def __call__(self, shape: Sequence[int], dtype: Any) -> jax.Array:
        input_size = shape[0]
        if self.is_first_layer:
            max_val = 1.0 / input_size
        else:
            max_val = np.sqrt(6 / input_size) / self.w0

        return hk.initializers.RandomUniform(-max_val, max_val)(shape, dtype)


class SirenLayer(hk.Module):
    """A single layer of a Siren network -- see https://arxiv.org/pdf/2006.09661.pdf.
    Directly modifies the Linear module provided by Haiku."""

    def __init__(
        self,
        output_size: int,
        w0: float = 30.0,
        use_residual_connection: bool = False,
        is_first_layer: bool = False,
        name: str = None,
    ):
        """Constructs the Siren module.
        Args:
          output_size: Output dimensionality.
          w0: Gain factor.
        """
        super().__init__(name=name)
        self.input_size = None
        self.output_size = output_size
        self.w0 = w0
        self.use_residual_connection = use_residual_connection
        self.is_first_layer = is_first_layer
        self.b_init = np.zeros
        self.name = name
        self.w_init = SirenInit(self.is_first_layer, self.w0)

    def __call__(
        self,
        inputs: jax.Array,
        *,
        precision: Optional[lax.Precision] = None,
    ) -> jax.Array:
        """Computes a Siren of the input."""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size

        dtype = inputs.dtype
        w_init = self.w_init
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)
        out = self.w0 * np.dot(inputs, w, precision=precision)
        b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
        b = np.broadcast_to(b, out.shape)
        out = out + b

        if input_size == output_size and self.use_residual_connection:
            return inputs + np.sin(out)
        else:
            return np.sin(out)


def get_neighbors(
    xi: np.ndarray, xs: np.ndarray, width: float, n_neighbors: int
) -> Tuple[jax.Array, jax.Array]:
    xdiffs = vmap(lambda xj: drifts.wrapped_diff(xi, xj, width))(xs)
    norms = np.linalg.norm(xdiffs, axis=1)
    inds = jax.lax.top_k(-norms, n_neighbors + 1)[1]
    return inds[1:], xdiffs[inds[1:]]


def define_full_particle_split_transformer(
    w0: float,
    d: int,
    N: int,
    num_layers: int,
    embed_dim: int,
    embed_n_hidden: int,
    decode_n_hidden: int,
    embed_n_neurons: int,
    num_heads: int,
    dim_feedforward: int,
    n_layers_feedforward: int,
    n_inducing_points: int,
    shift_func: Callable,
    particle_div_shift_func: Callable,
    div_shift_func: Callable,
) -> Tuple[Callable, Callable, Callable]:
    """Define separate scores in x and g at the total-particle level."""

    def net(
        xs: np.ndarray,  # [N, d]
        gs: np.ndarray,  # [N, d]
    ) -> np.ndarray:
        if embed_n_hidden > 0:
            g_embedding = hk.Sequential(
                construct_mlp_layers(
                    embed_n_hidden,
                    embed_n_neurons,
                    jax.nn.gelu,
                    w0=w0,
                    output_dim=embed_dim // 2,
                    use_layer_norm=False,
                    use_residual_connections=True,
                    name="g_embedding",
                )
            )

            x_embedding = hk.Sequential(
                construct_mlp_layers(
                    embed_n_hidden,
                    embed_n_neurons,
                    jax.nn.gelu,
                    w0=w0,
                    output_dim=embed_dim // 2,
                    use_layer_norm=False,
                    use_residual_connections=True,
                    name="x_embedding",
                )
            )

            # compute embeddings
            embedded_gs = g_embedding(gs)  # [N, embed_dim // 2]
            embedded_xs = x_embedding(xs)  # [N, embed_dim // 2]
            inp = np.hstack((embedded_xs, embedded_gs))  # [N, embed_dim]
        else:
            xgs = np.hstack((xs, gs))
            inp = hk.Linear(embed_dim, name="embedding")(xgs)

        ## compute the velocity
        if decode_n_hidden > 0:
            decoder = hk.Sequential(
                construct_mlp_layers(
                    decode_n_hidden,
                    embed_n_neurons,
                    jax.nn.gelu,
                    w0=w0,
                    output_dim=d,
                    use_layer_norm=False,
                    use_residual_connections=True,
                    name="decoder",
                )
            )
        else:
            decoder = hk.Linear(d, name="decoder")

        trans = Transformer(
            num_layers=num_layers,
            input_dim=embed_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            n_layers_feedforward=n_layers_feedforward,
            n_inducing_points=n_inducing_points,
            w0=w0,
        )

        return decoder(trans(inp))

    s = lambda xs, gs, key: net(xs, gs) + shift_func(xs, gs, key)

    def particle_div(
        xs: np.ndarray, gs: np.ndarray, key: str  # [N, d, N, d]  # [N, d, N, d]
    ) -> np.ndarray:
        if key == "x":
            particle_div = deriv_utils.vector_div(lambda x: net(x, gs), xs)  # [N]
        elif key == "g":
            particle_div = deriv_utils.vector_div(lambda g: net(xs, g), gs)  # [N]
        else:
            raise ValueError("key needs to be x or g.")

        return particle_div + particle_div_shift_func(xs, gs, key)

    def s_div(xs: np.ndarray, gs: np.ndarray, key: str) -> float:  # [N, d]  # [N, d]
        if key == "x":
            div = deriv_utils.scalar_div(lambda x: net(x, gs), xs)
        elif key == "g":
            div = deriv_utils.scalar_div(lambda g: net(xs, g), gs)
        else:
            raise ValueError("key needs to be x or g.")

        return div + div_shift_func(xs, gs, key)

    score_net = hk.without_apply_rng(hk.transform(s))
    particle_div_net = hk.without_apply_rng(hk.transform(particle_div))
    div_net = hk.without_apply_rng(hk.transform(s_div))

    return score_net, particle_div_net, div_net


def define_full_particle_split_mlp(
    w0: float,
    d: int,
    N: int,
    n_hidden: int,
    n_neurons: int,
    shift_func: Callable,
    particle_div_shift_func: Callable,
    div_shift_func: Callable,
    symmetric: bool,
    symmetric_point: np.ndarray,
) -> Tuple[Callable, Callable, Callable]:
    """Define separate scores in x and g at the total-particle level using a basic MLP."""

    def net(
        xs: np.ndarray,  # [N, d]
        gs: np.ndarray,  # [N, d]
    ) -> np.ndarray:
        mlp = hk.Sequential(
            construct_mlp_layers(
                n_hidden, n_neurons, jax.nn.gelu, w0=w0, output_dim=N * d
            )
        )
        inp = np.concatenate((xs.ravel(), gs.ravel()))

        if symmetric:
            return (mlp(inp) - mlp(symmetric_point - inp)).reshape((N, d))
        else:
            return mlp(inp).reshape((N, d))

    s = lambda xs, gs, key: net(xs, gs) + shift_func(xs, gs, key)
    jac_x = jax.jacfwd(net, argnums=0)
    jac_g = jax.jacfwd(net, argnums=1)

    def particle_div(
        xs: np.ndarray, gs: np.ndarray, key: str  # [N, d, N, d]  # [N, d, N, d]
    ) -> np.ndarray:
        if key == "x":
            jac = jac_x(xs, gs)  # [N, d, N, d]
        elif key == "g":
            jac = jac_g(xs, gs)  # [N, d, N, d]
        else:
            raise ValueError("key needs to be x or g.")

        return np.diag(np.trace(jac, axis1=1, axis2=3)) + particle_div_shift_func(
            xs, gs, key
        )

    def s_div(xs: np.ndarray, gs: np.ndarray, key: str) -> float:  # [N, d]  # [N, d]
        if key == "x":
            jac = jac_x(xs, gs)  # [N, d, N, d]
        elif key == "g":
            jac = jac_g(xs, gs)  # [N, d, N, d]
        else:
            raise ValueError("key needs to be x or g.")

        return np.trace(jac.reshape((N * d, N * d))) + div_shift_func(xs, gs, key)

    score_net = hk.without_apply_rng(hk.transform(s))
    particle_div_net = hk.without_apply_rng(hk.transform(particle_div))
    div_net = hk.without_apply_rng(hk.transform(s_div))

    return score_net, particle_div_net, div_net


def define_transformer_networks(
    w0: float,
    n_neighbors: int,
    num_layers: int,
    embed_dim: int,
    embed_n_hidden: int,
    embed_n_neurons: int,
    num_heads: int,
    dim_feedforward: int,
    n_layers_feedforward: int,
    width: float,
    network_type: str,
    this_particle_pooling: bool,
    scale_fac: float = 1.0,
) -> Tuple[Callable, Callable]:
    if network_type == "transformer_gi":

        def v(
            gi: np.ndarray,  # d
            xdiffs: np.ndarray,  # [N, d]
        ) -> np.ndarray:
            """Individual particle velocity"""
            ## set up latent embeddings
            g_embedding = hk.Sequential(
                construct_mlp_layers(
                    embed_n_hidden,
                    embed_n_neurons,
                    jax.nn.gelu,
                    w0=w0,
                    output_dim=embed_dim,
                )
            )
            x_embedding = hk.Sequential(
                construct_mlp_layers(
                    embed_n_hidden,
                    embed_n_neurons,
                    jax.nn.gelu,
                    w0=w0,
                    output_dim=embed_dim,
                )
            )

            ## compute the latent embedding
            embedded_gi = g_embedding(gi)

            # compute nearest neighbors
            norms = np.linalg.norm(xdiffs, axis=1)
            inds = jax.lax.top_k(-norms, n_neighbors + 1)[1]
            embedded_xs = x_embedding(xdiffs[inds[1:]])
            inp = np.vstack((embedded_gi, embedded_xs))

            ## compute the velocity
            decoder = hk.Sequential(
                construct_mlp_layers(
                    embed_n_hidden, embed_n_neurons, jax.nn.gelu, w0=w0, output_dim=2
                )
            )

            trans = Transformer(
                num_layers,
                embed_dim,
                num_heads,
                dim_feedforward,
                n_layers_feedforward,
                w0,
            )
            return decoder(np.mean(trans(inp), axis=0)) * scale_fac

        vel_jac = jax.jacfwd(v, argnums=0)
        vel_div = lambda gi, xdiffs: np.trace(vel_jac(gi, xdiffs))

        def particle_div(xgs: np.ndarray, ii: int) -> float:  # 2N x d
            xs, gs = np.split(xgs, 2)
            xi, gi = xs[ii], gs[ii]
            xdiffs = vmap(lambda xj: drifts.wrapped_diff(xi, xj, width))(xs)
            return vel_div(gi, xdiffs)

        def particle_score(xgs: np.ndarray, ii: int) -> np.ndarray:  # 2N x d
            xs, gs = np.split(xgs, 2)
            xi, gi = xs[ii], gs[ii]
            xdiffs = vmap(lambda xj: drifts.wrapped_diff(xi, xj, width))(xs)
            return v(gi, xdiffs)

        particle_score_net = hk.without_apply_rng(hk.transform(particle_score))
        particle_div_net = hk.without_apply_rng(hk.transform(particle_div))

    elif network_type == "transformer_full":

        def v(
            gi: np.ndarray,  # [d]
            xdiffs: np.ndarray,  # [N, d]
            gs: np.ndarray,  # [N, d]
        ) -> np.ndarray:
            """Individual particle velocity"""
            ## set up latent embeddings
            embedding = hk.Sequential(
                construct_mlp_layers(
                    embed_n_hidden,
                    embed_n_neurons,
                    jax.nn.gelu,
                    name="embedding",
                    w0=w0,
                    output_dim=embed_dim,
                    use_layer_norm=False,
                    use_residual_connections=True,
                )
            )

            # compute nearest neighbors
            norms = np.linalg.norm(xdiffs, axis=1)
            inds = jax.lax.top_k(-norms, n_neighbors + 1)[1]

            # compute embeddings
            neighbor_xs = xdiffs[inds]  # [n_neighbors+1, d]
            neighbor_gs = gs[inds]  # [n_neighbors+1, d]
            this_particle = np.concatenate((neighbor_xs[0], gi))  # [2*d]
            neighbors = np.hstack(
                (neighbor_xs[1:], neighbor_gs[1:])
            )  # [n_neighbors, 2*d]
            inp = embedding(
                np.vstack((this_particle, neighbors))
            )  # [n_neighbors+1, 2*d]

            ## compute the velocity
            decoder = hk.Sequential(
                construct_mlp_layers(
                    embed_n_hidden,
                    embed_n_neurons,
                    jax.nn.gelu,
                    name="decoder",
                    w0=w0,
                    output_dim=2,
                    use_layer_norm=False,
                    use_residual_connections=True,
                )
            )

            trans = Transformer(
                num_layers=num_layers,
                input_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                n_layers_feedforward=n_layers_feedforward,
                n_inducing_points=0,
                w0=w0,
            )

            if this_particle_pooling:
                decoder(trans(inp)[0])
            else:
                return decoder(np.mean(trans(inp), axis=0))

        vel_jac = jax.jacfwd(v, argnums=0)
        vel_div = lambda gi, xdiffs, gs: np.trace(vel_jac(gi, xdiffs, gs))

        def particle_div(xgs: np.ndarray, ii: int) -> float:  # 2N x d
            xs, gs = np.split(xgs, 2)
            xi, gi = xs[ii], gs[ii]
            xdiffs = vmap(lambda xj: drifts.wrapped_diff(xi, xj, width))(xs)
            return vel_div(gi, xdiffs, gs)

        def particle_score(xgs: np.ndarray, ii: int) -> np.ndarray:  # 2N x d
            xs, gs = np.split(xgs, 2)
            xi, gi = xs[ii], gs[ii]
            xdiffs = vmap(lambda xj: drifts.wrapped_diff(xi, xj, width))(xs)
            return v(gi, xdiffs, gs)

        particle_score_net = hk.without_apply_rng(hk.transform(particle_score))
        particle_div_net = hk.without_apply_rng(hk.transform(particle_div))

    elif network_type == "transformer_separate_encode":

        def v(
            gi: np.ndarray,  # d
            xdiffs: np.ndarray,  # [N, d]
            gs: np.ndarray,  # [N, d]
        ) -> np.ndarray:
            """Individual particle velocity"""
            ## set up latent embeddings
            g_embedding = hk.Sequential(
                construct_mlp_layers(
                    embed_n_hidden,
                    embed_n_neurons,
                    jax.nn.gelu,
                    name="g_embedding",
                    w0=w0,
                    output_dim=embed_dim // 2,
                    use_layer_norm=False,
                    use_residual_connections=True,
                )
            )

            x_embedding = hk.Sequential(
                construct_mlp_layers(
                    embed_n_hidden,
                    embed_n_neurons,
                    jax.nn.gelu,
                    name="x_embedding",
                    w0=w0,
                    output_dim=embed_dim // 2,
                    use_layer_norm=False,
                    use_residual_connections=True,
                )
            )

            # compute nearest neighbors
            norms = np.linalg.norm(xdiffs, axis=1)
            inds = jax.lax.top_k(-norms, n_neighbors + 1)[1]

            # compute embeddings
            embedded_gi = g_embedding(gi)  # [embed_dim]
            embedded_xs = x_embedding(xdiffs[inds])  # [n_neighbors+1, embed_dim]
            embedded_gs = g_embedding(gs[inds])  # [n_neighbors+1, embed_dim]

            # concatenate appropriately
            # note that because xdiff=0 for this particle it's always the first index.
            this_particle = np.concatenate((embedded_gi, embedded_xs[0]))
            other_particles = np.hstack((embedded_gs[1:], embedded_xs[1:]))
            inp = np.vstack((this_particle, other_particles))

            ## compute the velocity
            decoder = hk.Sequential(
                construct_mlp_layers(
                    embed_n_hidden,
                    embed_n_neurons,
                    jax.nn.gelu,
                    name="decoder",
                    w0=w0,
                    output_dim=2,
                    use_layer_norm=False,
                    use_residual_connections=True,
                )
            )

            trans = Transformer(
                num_layers=num_layers,
                input_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                n_layers_feedforward=n_layers_feedforward,
                n_inducing_points=0,
                w0=w0,
            )

            if this_particle_pooling:
                return decoder(trans(inp)[0])
            else:
                return decoder(np.mean(trans(inp), axis=0))

        vel_jac = jax.jacfwd(v, argnums=0)
        vel_div = lambda gi, xdiffs, gs: np.trace(vel_jac(gi, xdiffs, gs))

        def particle_div(xgs: np.ndarray, ii: int) -> float:  # 2N x d
            xs, gs = np.split(xgs, 2)
            xi, gi = xs[ii], gs[ii]
            xdiffs = vmap(lambda xj: drifts.wrapped_diff(xi, xj, width))(xs)
            return vel_div(gi, xdiffs, gs)

        def particle_score(xgs: np.ndarray, ii: int) -> np.ndarray:  # 2N x d
            xs, gs = np.split(xgs, 2)
            xi, gi = xs[ii], gs[ii]
            xdiffs = vmap(lambda xj: drifts.wrapped_diff(xi, xj, width))(xs)
            return v(gi, xdiffs, gs)

        particle_score_net = hk.without_apply_rng(hk.transform(particle_score))
        particle_div_net = hk.without_apply_rng(hk.transform(particle_div))

    return particle_score_net, particle_div_net


def construct_mlp_layers(
    n_hidden: int,
    n_neurons: int,
    act: Callable[[np.ndarray], np.ndarray],
    name: str = None,
    output_dim: int = 1,
    w0: float = 0.0,
    use_layer_norm: bool = False,
    use_residual_connections: bool = False,
) -> list:
    """Make a list containing the layers of an MLP.

    Args:
        n_hidden: Number of hidden layers in the MLP.
        n_neurons: Number of neurons per hidden layer.
        act: Activation function.
        output_dim: Dimension of the output.
    """
    layers = []
    for layer in range(n_hidden):
        ## construct layer
        layer_name = None if name == None else f"{name}_{layer}"

        if use_layer_norm:
            layers = layers + [
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ]

        # standard network
        if w0 == 0.0:
            layers = layers + [hk.Linear(n_neurons, name=layer_name), act]
        else:
            is_first_layer = layer == 0
            layers = layers + [
                SirenLayer(
                    n_neurons,
                    w0,
                    use_residual_connections,
                    is_first_layer,
                    name=layer_name,
                )
            ]

    ## construct output layer
    output_name = f"{name}_{n_hidden}" if name != None else f"linear_{n_hidden}"
    layers = layers + [hk.Linear(output_dim, name=output_name)]
    return layers
