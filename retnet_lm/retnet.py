from typing import Callable, Any, Optional

from flax import linen as nn
from flax import struct

import jax
from jax import lax
import jax.numpy as jnp
from jax.nn import relu, gelu, silu


import numpy as np

@struct.dataclass
class RetNetConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
    vocab_size: int
    output_vocab_size: int
    share_embeddings: bool = False
    logits_via_embedding: bool = False
    dtype: Any = jnp.float32
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    qkv_dim: int = 512
    mlp_dim: int = 2048
    max_len: int = 2048
    dropout_rate: float = 0.1
    rettention_dropout_rate: float = 0.1
    deterministic: bool = False
    decode: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    posemb_init: Optional[Callable] = None
    chunk_size: int = 0
    chunkwise_recurrent: bool = False
    recurrent_chunk_size: int = 0
    use_lm_decay: bool = False
    activation_fn: str = "gelu"
    activation_dropout: float = 0.0
    use_glu: bool = False
    use_ffn_rms_norm: bool = False
    subln: bool = False
    deepnorm: bool = False
    no_scale_embedding: bool = False
    layernorm_eps: float = 1e-5
    encoder_normalize_before: bool = False
    decoder_normalize_before: bool = False
    decoder_layers: int = 6
    elsementwise_affine: bool = True


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


def shift_inputs(x, segment_ids=None, axis=1):
  """Shift inputs and replace EOS by 0 for packed inputs."""
  shifted = shift_right(x, axis=axis)
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  if segment_ids is not None:
    shifted *= (segment_ids == shift_right(segment_ids, axis=axis))
  return shifted


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    return x.reshape(*x.shape[:-2], -1)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


def get_activation_fn_jax(activation):
    if activation == "relu":
        return relu
    elif activation == "gelu":
        return gelu
    elif activation == "swish":
        return silu
    else:
        raise NotImplementedError


class RMSNorm(nn.Module):
    config: RetNetConfig

    def setup(self):

      if self.config.elementwise_affine:
        self.weight = nn.Parameter('weight', nn.initializers.ones, (self.config,))
      else:
            self.weight = None

    def __call__(self, x):
        norm = x * lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        if self.weight is not None:
            norm *= self.weight
        return norm


class RetNetRelPos(nn.Module):
    config: RetNetConfig

    def setup(self):
        num_heads = self.config.retention_heads
        embed_dim = self.config.decoder_embed_dim
        use_lm_decay = self.config.use_lm_decay
        chunk_size = self.config.recurrent_chunk_size
        chunkwise_recurrent = self.config.chunkwise_recurrent

        angle = 1.0 / (10000 ** jnp.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.reshape(-1, 2).flatten()
        
        if use_lm_decay:
            s = jnp.log(1 / 32)
            e = jnp.log(1 / 512)
            decay = jnp.log(1 - jnp.exp(jnp.linspace(s, e, num_heads)))
        else:
            decay = jnp.log(1 - 2 ** (-5 - jnp.arange(num_heads, dtype=jnp.float32)))

        self.angle = angle
        self.decay = decay

    def __call__(self, slen, activate_recurrent=False, chunkwise_recurrent=False):

        if activate_recurrent:
            sin = jnp.sin(self.angle * (slen - 1))
            cos = jnp.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), jnp.exp(self.decay))
        elif chunkwise_recurrent:
            pass
        else:
            pass
        return retention_rel_pos


class MultiScaleRetention(nn.Module):
    config: RetNetConfig

    def setup(self):
        self.head_dim = self.value_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim ** -0.5

        #initializeを追加する必要あり
        self.q_proj = nn.Dense(self.embed_dim, use_bias=False)
        self.k_proj = nn.Dense(self.embed_dim, use_bias=False)
        self.v_proj = nn.Dense(self.embed_dim, self.value_dim, use_bias=False)
        self.g_proj = nn.Dense(self.embed_dim, self.value_dim, use_bias=False)
        self.out_proj = nn.Dense(self.value_dim, self.embed_dim, use_bias=False)

        self.group_norm = RMSNorm(self.head_dim, eps=self.config.layernorm_eps, elementwise_affine=False)

    def parallel_forward(self, qr, kr, v, mask):
        pass

    def recurrent_forward(self, qr, kr, v, decay, incremental_state):
        pass

    def chunk_recurrent_forward(self, qr, kr, v, inner_mask):
        pass

    def __call__(self, x, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        pass


class FeedForwardNetwork(nn.Module):
    config: RetNetConfig

    def setup(self):
        self.fc1 = nn.Dense(self.ffn_dim)
        self.fc2 = nn.Dense(self.embed_dim)
        self.activation_fn = get_activation_fn_jax(self.activation_fn)
        self.activation_dropout_module = nn.Dropout(rate=self.dropout_rate)
        self.dropout_module = nn.Dropout(rate=self.dropout_rate)

        if self.subln:
            if self.use_rms_norm:
                self.ffn_layernorm = RMSNorm(self.embed_dim, eps=self.layernorm_eps)
            else:
                # LayerNorm の Flax 実装を使用する
                self.ffn_layernorm = nn.LayerNorm(epsilon=self.layernorm_eps)
        else:
            self.ffn_layernorm = None

    def __call__(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x, deterministic=False)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = x.reshape(x_shape)
        x = self.dropout_module(x, deterministic=False)
        return x


class GLU(nn.Module):
    config: RetNetConfig

    def setup(self):
        self.fc1 = nn.Dense(self.ffn_dim, use_bias=False)
        self.fc2 = nn.Dense(self.embed_dim, use_bias=False)
        self.gate = nn.Dense(self.ffn_dim, use_bias=False)
        self.activation_fn = get_activation_fn_jax(self.activation_fn)
        self.activation_dropout_module = nn.Dropout(rate=self.activation_dropout_rate)
        self.dropout_module = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        g = self.gate(x)
        x = self.fc1(x)
        x = self.activation_fn(x) * g
        x = self.activation_dropout_module(x, deterministic=False)
        x = self.fc2(x)
        x = x.reshape(x_shape)
        x = self.dropout_module(x, deterministic=False)
        return x


class DropPath(nn.Module):
    config: RetNetConfig

    def __call__(self, x, rng):
        if self.drop_prob == 0. or not self.is_training():
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_mask = jax.random.bernoulli(rng, keep_prob, shape)
        return lax.select(random_mask, x, jnp.zeros_like(x)) / keep_prob


class RetNetDecoderLayer(nn.Module):
    config: RetNetConfig

    def setup(self):
        self.embed_dim = self.config.embed_dim
        self.dropout_module = nn.Dropout(rate=self.config.dropout_rate)

        if self.config.dropout_path_rate > 0:
            drop_path_prob = jnp.linspace(0, self.config.drop_path_rate, self.config.decoder_layers)[self.depth]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.retention = MultiScaleRetention(
            self.config,
            self.embed_dim,
            self.config.embed_dim,
            self.config.num_heads,
        )

        self.normalize_before = self.config.decoder_normalize_before
        self.retention_layer_norm = RMSNorm(self.embed_dim, eps=self.config.layernorm_eps)
        self.ffn_dim = self.config.ffn_dim
        self.ffn = self.build_ffn()
        self.final_layer_norm = RMSNorm(self.embed_dim, eps=self.config.layernorm_eps)
        self.alpha = jnp.sqrt(2.0 * self.config.decoder_layers) if self.config.deepnorm else 1.0

    def build_ffn(self):
        if self.config.use_glu:
            return GLU(
                config=self.config
            )
        else:
            return FeedForwardNetwork(
                config=self.config
            )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def __call__(self, x, rng=None, incremental_state=None, chunkwise_recurrent=False, retention_rel_pos=None):
        residual = x
        if self.normalize_before:
            x = self.retention_layer_norm(x)

        x = self.retention(x, retention_rel_pos, chunkwise_recurrent=chunkwise_recurrent)

        x = self.dropout_module(x, deterministic=False)

        if self.drop_path is not None:
            rng, sub_rng = jax.random.split(rng)
            x = self.drop_path(x, sub_rng)

        x = self.residual_connection(x, residual)

        if not self.normalize_before:
            x = self.retention_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.ffn(x)

        if self.drop_path is not None:
            x = self.drop_path(x, rng)

        x = self.residual_connection(x, residual)

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x


class RetNetModel(nn.Module):
    config: RetNetConfig

    def setup(self):
        self.dropout_module = nn.Dropout(rate=self.config.dropout_rate)
        self.embed_dim = self.config.embed_dim
        self.embed_scale = 1.0 if self.config.no_scale_embedding else jnp.sqrt(self.embed_dim)

        if self.config.layernorm_embedding:
            self.layernorm_embedding = RMSNorm(self.embed_dim, eps=self.config.layernorm_eps)
        else:
            self.layernorm_embedding = None

        self.layers = [RetNetDecoderLayer(self.config) for i in range(self.config.decoder_layers)]

        if self.config.decoder_normalize_before:
            self.layer_norm = RMSNorm(self.embed_dim, eps=self.config.layernorm_eps)
        else:
            self.layer_norm = None

        self.retnet_rel_pos = RetNetRelPos(self.config)
        self.chunkwise_recurrent = self.config.chunk_wise_recurrent
        self.recurrent_chunk_size = self.config.recurrent_chunk_size

    def build_output_projection(self):
        pass

    def forward_embedding(self, tokens, token_embedding=None, incremental_state=None):
        pass

    def __call__(self, prev_output_tokens, incremental_state=None, features_only=False, token_embeddings=None):
        pass

    def output_layer(self, features):
        pass


class RetNetForCausalLM(nn.Module):
    config: RetNetConfig

    def setup(self):
        assert self.config.vocab_size > 0, "You must specify vocab size"
        if self.embed_tokens is None:
            self.embed_tokens = nn.Embed(
                num_embeddings=self.config.vocab_size, 
                features=self.config.embed_dim,
                padding_idx=self.config.pad_token_id
            )
        self.model = RetNetModel(
            self.config,
            embed_tokens=self.embed_tokens,
            output_projection=self.output_projection
        )

    def __call__(
        self,
        input_ids=None,
        retention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_retentions=None,
        output_hidden_states=None,
        return_dict=None,
        recurrent_chunk_size=None,
    ):
        # モデルのフォワードパス
        outputs = self.model(
            input_ids=input_ids,
            incremental_state=past_key_values,
            token_embeddings=inputs_embeds
        )

        logits, inner_hidden_states = outputs

        loss = None
        if labels is not None:
            pass

        return logits.astype(self.config.dtype)