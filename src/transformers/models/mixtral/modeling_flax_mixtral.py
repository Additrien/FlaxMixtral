# coding=utf-8
# Copyright 2024 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flax Mixtral model."""
import dataclasses
import functools
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from t5x import partitioning as t5x_partitioning

from ...modeling_flax_outputs import (
    FlaxCausalLMOutput,
    FlaxMoeCausalLMOutputWithPast,
    FlaxMoeModelOutputWithPast,
)
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_mixtral import MixtralConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MixtralConfig"
_REAL_CHECKPOINT_FOR_DOC = "mistralai/Mistral-7B-v0.1"
_CHECKPOINT_FOR_DOC = "mistralai/Mixtral-8x7B-v0.1"

MIXTRAL_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`MixtralConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16`, or
            `jax.numpy.bfloat16`.

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

MIXTRAL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


def load_balancing_loss_func(gate_logits, num_experts=None, top_k=2, attention_mask=None) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Flax.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device()
        concatenated_gate_logits = jnp.concatenate(
            [jax.device_put(layer_gate, compute_device) for layer_gate in gate_logits], axis=0
        )

    routing_weights = nn.activation.softmax(concatenated_gate_logits, axis=-1)

    _, selected_experts = jax.lax.topk(routing_weights, top_k)

    expert_mask = nn.activation.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = jnp.mean(expert_mask, axis=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = jnp.mean(routing_weights, axis=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = attention_mask[None, :, :, None, None]
        expert_attention_mask = jnp.broadcast_to(
            expert_attention_mask, (num_hidden_layers, batch_size, sequence_length, 2, num_experts)
        )
        expert_attention_mask = jnp.reshape(expert_attention_mask, (-1, 2, num_experts))
        expert_attention_mask = jax.device_put(expert_attention_mask, compute_device)

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = jnp.sum(expert_mask * expert_attention_mask, axis=0) / jnp.sum(
            expert_attention_mask, axis=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = attention_mask[None, :, :, None]
        router_per_expert_attention_mask = jnp.broadcast_to(
            router_per_expert_attention_mask, (num_hidden_layers, batch_size, sequence_length, num_experts)
        )
        router_per_expert_attention_mask = jnp.reshape(router_per_expert_attention_mask, (-1, num_experts))
        router_per_expert_attention_mask = jax.device_put(router_per_expert_attention_mask, compute_device)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = jnp.sum(routing_weights * router_per_expert_attention_mask, axis=0) / jnp.sum(
            router_per_expert_attention_mask, axis=0
        )

    overall_loss = jnp.sum(tokens_per_expert * jnp.expand_dims(router_prob_per_expert, axis=0))
    return overall_loss * num_experts


@jax.jit
def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


@jax.jit
def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


@jax.jit
def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Sequence[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]
default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0
)


# copied from https://github.com/google-research/t5x/blob/main/t5x/examples/t5/layers.py#L349
def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


# copied from https://github.com/google-research/t5x/blob/main/t5x/examples/t5/layers.py#L354
def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


# copied from https://github.com/google-research/t5x/blob/main/t5x/examples/t5/layers.py#L364
class DenseGeneral(nn.Module):
  """A linear transformation (without bias) with flexible axes.

  Attributes:
    features: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
  """

  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  dtype: DType = jnp.float32
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal'
  )
  kernel_axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
    kernel_param_shape = (
        np.prod([inputs.shape[ax] for ax in axis]),
        np.prod(features),
    )
    kernel = param_with_axes(
        'kernel',
        self.kernel_init,
        kernel_param_shape,
        jnp.float32,
        axes=self.kernel_axes,
    )
    kernel = jnp.asarray(kernel, self.dtype)
    kernel = jnp.reshape(kernel, kernel_shape)

    contract_ind = tuple(range(0, len(axis)))
    return lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))


# copied from https://github.com/google-research/t5x/blob/main/t5x/examples/t5/layers.py#L490C1-L558C1
class Embed(nn.Module):
  """A parameterized function from integers [0, n) to d-dimensional vectors.

  Attributes:
    num_embeddings: number of embeddings.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    embedding_init: embedding initializer.
    one_hot: performs the gather with a one-hot contraction rather than a true
      gather. This is currently needed for SPMD partitioning.
  """

  num_embeddings: int
  features: int
  cast_input_dtype: Optional[DType] = None
  dtype: DType = jnp.float32
  attend_dtype: Optional[DType] = None
  embedding_init: Initializer = default_embed_init
  one_hot: bool = False
  embedding: Array = dataclasses.field(init=False)

  def setup(self):
    self.embedding = param_with_axes(
        'embedding',
        self.embedding_init,
        (self.num_embeddings, self.features),
        jnp.float32,
        axes=('vocab', 'embed'),
    )

  def __call__(self, inputs: Array) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')
    if self.one_hot:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
    else:
      output = jnp.asarray(self.embedding, self.dtype)[inputs]
      output = with_sharding_constraint(output, ('batch', 'length', 'embed'))
    return output

  def attend(self, query: Array) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return jnp.dot(query, jnp.asarray(self.embedding, dtype).T)


class FlaxMixtralRMSNorm(nn.Module):
    config: MixtralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.epsilon = self.config.rms_norm_eps
        self.weight = self.param("weight", lambda _, shape: jnp.ones(shape), self.config.hidden_size)
        self.weight = param_with_axes(
            "weigth",
            nn.with_logical_partitioning(
                lambda _, shape: jnp.ones(shape), ("embed",)
            ),
            self.config.hidden_size,
            axes = ("embed",)
        )
    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)


# Copied from transformers.models.llama.modeling_flax_llama.FlaxLlamaRotaryEmbedding with Llama->Mixtral
class FlaxMixtralRotaryEmbedding(nn.Module):
    config: MixtralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.sincos = create_sinusoidal_positions(self.config.max_position_embeddings, head_dim)

    def __call__(self, key, query, position_ids):
        sincos = self.sincos[position_ids]
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)

        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

        key = jnp.asarray(key, dtype=self.dtype)
        query = jnp.asarray(query, dtype=self.dtype)

        return key, query


# Copied from transformers.models.mistral.modeling_flax_mistral.FlaxMistralAttention with Mistral->Mixtral
class FlaxMixtralAttention(nn.Module):
    config: MixtralConfig
    dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform()

    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32
        self.rope_theta = config.rope_theta
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = DenseGeneral(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("embed", "joined_kv")
            ),
            kernel_axes=("embed", "joined_kv"),
            name="q_proj",
        )
        self.k_proj = DenseGeneral(
            self.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("embed", "joined_kv")
            ),
            kernel_axes=("embed", "joined_kv"),
            name="k_proj",
        )
        self.v_proj = DenseGeneral(
            self.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("embed", "joined_kv")
            ),
            kernel_axes=("embed", "joined_kv"),
            name="v_proj",
        )
        self.o_proj = DenseGeneral(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("joined_kv", "embed")
            ),
            kernel_axes=("joined_kv", "embed"),
            name="o_proj",
        )
        casual_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")
        self.causal_mask = jnp.triu(casual_mask, k=-config.sliding_window)
        self.rotary_emb = FlaxMixtralRotaryEmbedding(config, dtype=self.dtype)

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    @nn.compact
    # Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoSelfAttention._concatenate_to_cache
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        causal_mask: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        init_cache: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        hidden_states = with_sharding_constraint(
            hidden_states, ("batch", "length", "embed")
        )
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states, self.num_heads)
        key_states = self._split_heads(key_states, self.num_key_value_heads)
        value_states = self._split_heads(value_states, self.num_key_value_heads)

        query_states = with_sharding_constraint(
            query_states, ("batch", "heads", "length", "kv")
        )

        key_states, query_states = self.rotary_emb(key_states, query_states, position_ids)
        query_length, key_length = query_states.shape[1], key_states.shape[1]
        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = causal_mask[:, :, :query_length, :key_length]
        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)

        if self.has_variable("cache", "cached_key") or init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        key_states = with_sharding_constraint(
            key_states, ("batch", "heads", "kv_length", "kv")
        )
        value_states = with_sharding_constraint(
            value_states, ("batch", "heads", "kv_length", "kv")
        )

        key_states = jnp.repeat(key_states, self.num_key_value_groups, axis=2)
        value_states = jnp.repeat(value_states, self.num_key_value_groups, axis=2)

        key_states = with_sharding_constraint(
            key_states, ("batch", "heads", "kv_length", "kv")
        )
        value_states = with_sharding_constraint(
            value_states, ("batch", "heads", "kv_length", "kv")
        )

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )
        # usual dot product attention
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            deterministic=deterministic,
            dropout_rate=self.config.attention_dropout,
            dtype=attention_dtype,
        )

        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)

        attn_weights = with_sharding_constraint(
            attn_weights, ("batch", "heads", "length", "kv_length")
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = with_sharding_constraint(
            attn_output, ("batch", "heads", "length", "kv")
        )
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)
        attn_output = with_sharding_constraint(
            attn_output, ("batch", "heads", "length", "kv")
        )
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxMixtralBLockSparseTop2MLP(nn.Module):
    config: MixtralConfig
    dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform()

    def setup(self) -> None:
        self.w1 = DenseGeneral(
            self.config.hidden_size, 
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("intermediate", "embed")
            ),
            kernel_axes=("intermediate", "embed"),
            name="w1"    
        )
        self.w2 = DenseGeneral(
            self.config.intermediate_size, 
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("embed", "intermediate")
            ),
            kernel_axes=("embed", "intermediate"),
            name="w2"    
        )
        self.w3 = DenseGeneral(
            self.config.hidden_size, 
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("intermediate", "embed")
            ),
            kernel_axes=("intermediate", "embed"),
            name="w3"    
        )

        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = with_sharding_constraint(hidden_states, ("batch", "length", "embed"))
        current_hidden_states = self.act_fn(
            with_sharding_constraint(
                self.w1(hidden_states), ("batch", "length", "intermediate")
            )
        ) * with_sharding_constraint(self.w3(hidden_states), ("batch", "length", "intermediate"))
        current_hidden_states = with_sharding_constraint(self.w2(current_hidden_states), ("batch", "length", "embed"))

        return current_hidden_states


class FlaxMixtralBlockSparesTop2MLPCollection(nn.Module):
    config: MixtralConfig
    dtype: jnp.dtype = jnp.float32
    kernel_init = nn.initializers.xavier_uniform()

    def setup(self) -> None:
        self.experts = [
            FlaxMixtralBLockSparseTop2MLP(config=self.config, dtype=self.dtype, kernel_init=self.kernel_init, name=str(i))
            for i in range(self.config.num_local_experts)
        ]

    def __call__(
        self, expert_mask, hidden_states, routing_weights, batch_size: int, sequence_length: int, hidden_dim: int
    ):
        final_hidden_states = jnp.zeros(((batch_size * sequence_length) + 1, hidden_dim), dtype=hidden_states.dtype)

        for expert_idx in range(self.config.num_local_experts):
            selected_mask = expert_mask[expert_idx]
            idx, top_x = jnp.nonzero(selected_mask, size=sequence_length, fill_value=-1)

            if top_x.shape[0] == 0:
                continue

            def expert(layer, input_hidden_states):
                current_state = hidden_states[top_x]
                current_hidden_states = layer.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]

                input_hidden_states = input_hidden_states.at[top_x].set(
                    current_hidden_states + input_hidden_states[top_x]
                )
                return input_hidden_states

            final_hidden_states = expert(self, final_hidden_states)

        final_hidden_states = final_hidden_states[:-1]
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states


class FlaxMixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    config: MixtralConfig
    dtype: jnp.dtype = jnp.float32
    kernel_init = nn.initializers.xavier_uniform()

    def setup(self) -> None:
        self.gate = DenseGeneral(
            self.config.num_local_experts,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("mlp",),
                kernel_axes=("mlp",),
                name="gate"
            )
        )
        self.experts = FlaxMixtralBlockSparesTop2MLPCollection(config=self.config, dtype=self.dtype, kernel_type=self.kernel_type)

    def __call__(
        self,
        hidden_states,
    ):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights = jax.nn.softmax(router_logits, axis=1)

        routing_weights, selected_experts = jax.lax.top_k(routing_weights, self.config.num_experts_per_tok)

        routing_weights /= jnp.sum(routing_weights, axis=-1, keepdims=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.astype(hidden_states.dtype)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = jax.nn.one_hot(selected_experts, num_classes=self.config.num_local_experts).transpose(2, 1, 0)

        final_hidden_states = self.experts(
            expert_mask=expert_mask,
            hidden_states=hidden_states,
            routing_weights=routing_weights,
            batch_size=batch_size,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
        )

        return final_hidden_states, router_logits


class FlaxMixtralDecoderLayer(nn.Module):
    config: MixtralConfig
    dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform()

    def setup(self):
        self.self_attn = FlaxMixtralAttention(self.config, dtype=self.dtype, kernel_init=self.kernel_init)
        self.block_sparse_moe = FlaxMixtralSparseMoeBlock(self.config, dtype=self.dtype, kernel_init=self.kernel_init)
        self.input_layernorm = FlaxMixtralRMSNorm(self.config, dtype=self.dtype, kernel_init=self.kernel_init)
        self.post_attention_layernorm = FlaxMixtralRMSNorm(self.config, dtype=self.dtype, kernel_init=self.kernel_init)

    def __call__(
        self,
        hidden_states,
        causal_mask,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        outputs = self.self_attn(
            hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # residual connection
        attn_output = outputs[0]
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        # residual connection
        hidden_states = residual + hidden_states

        if self.config.output_router_logits:
            return (hidden_states,) + outputs[1:] + (router_logits,)
        else:
            return (hidden_states,) + outputs[1:]


# Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoPreTrainedModel with GPTNeo->Mixtral, GPT_NEO->MIXTRAL, transformer->model
class FlaxMixtralPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MixtralConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: MixtralConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxMixtralAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxMixtralLayerCollection(nn.Module):
    config: MixtralConfig
    dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform()

    def setup(self):
        casual_mask = make_causal_mask(jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool")
        self.causal_mask = jnp.triu(casual_mask, k=-self.config.sliding_window)
        self.blocks = [
            FlaxMixtralDecoderLayer(self.config, dtype=self.dtype, kernel_init=self.kernel_init, name=str(i))
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if self.config.output_router_logits else None
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = block(
                hidden_states,
                causal_mask=self.causal_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)
            if self.config.output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        # this contains possible `None` values - `FlaxMixtralModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions, all_router_logits)

        return outputs


class FlaxMixtralModule(nn.Module):
    config: MixtralConfig
    dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform()

    def setup(self):
        self.hidden_size = self.config.hidden_size
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        # self.embed_tokens = nn.Embed(
        #     self.config.vocab_size,
        #     self.hidden_size,
        #     embedding_init=embedding_init,
        #     dtype=self.dtype,
        # )   
        self.embed_tokens = Embed(
            num_embeddings=self.vocab_size,
            features=self.config.hidden_size,
            attend_dtype=self.dtype,
            embedding_init=nn.with_logical_partitioning(
                embedding_init, ("vocab","embed",),
            ),
            one_hot=True,
            name="embed_tokens",
        )

        self.layers = FlaxMixtralLayerCollection(self.config, dtype=self.dtype)
        self.norm = FlaxMixtralRMSNorm(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if self.config.output_router_logits else None

        input_embeds = self.embed_tokens(input_ids.astype("i4"))

        outputs = self.layers(
            input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if output_attentions:
            all_self_attns += outputs[2]

        if self.config.output_router_logits:
            all_router_logits += outputs[-1]

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attns, all_router_logits] if v is not None
            )

        return FlaxMoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


@add_start_docstrings(
    "The bare Mixtral Model transformer outputting raw hidden-states without any specific head on top.",
    MIXTRAL_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_flax_llama.FlaxLlamaModel with Llama->Mixtral
class FlaxMixtralModel(FlaxMixtralPreTrainedModel):
    module_class = FlaxMixtralModule


append_call_sample_docstring(
    FlaxMixtralModel,
    _CHECKPOINT_FOR_DOC,
    FlaxMoeModelOutputWithPast,
    _CONFIG_FOR_DOC,
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)


class FlaxMixtralForCausalLMModule(nn.Module):
    config: MixtralConfig
    dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform()
    sharded: Optional[bool] = len(jax.devices()) > 1 and len(jax.devices()) % 2 == 0

    @staticmethod
    def mesh_sharding(pspec: PartitionSpec | None, mesh: Mesh | None) -> NamedSharding:
        if mesh is None:
            mesh = Mesh(jax.devices(), (None,))
        return NamedSharding(mesh, pspec)

    @staticmethod
    def _parse_mesh_layout(device_mesh_layout):
        assert isinstance(device_mesh_layout, (list, tuple)), (
            f"device_mesh_layout must be a list or tuple. "
            f"Got {type(device_mesh_layout)}"
        )
        assert len(device_mesh_layout) == 2, (
            f"The length of device_mesh_layout must be 2. "
            f"Got {len(device_mesh_layout)}"
        )
        mesh_layout = []
        for i in range(2):
            if device_mesh_layout[i] is None:
                assert (
                    device_mesh_layout[1 - i] is not None
                ), f"Invalid device_mesh_layout. Got {device_mesh_layout}."
                mesh_layout.append(len(jax.devices()) // device_mesh_layout[1 - i])
            else:
                mesh_layout.append(device_mesh_layout[i])

        return tuple(mesh_layout)

    def get_params(self, device_mesh_layout=(1, None), weights=None):
        """
        Get the properly sharded parameters.
        Args:
            device_mesh_layout: the device mesh layout. For example:
                (1, None) means data=1, model=len(jax.devices())
                (2, None) means data=2, model=len(jax.devices()) // 2
                (None, 2) means data=len(jax.devices()) // 2, model=2
            weights: whether a tree of weights are already given (but may not be sharded)
        Returns:
            a tree of properly sharded parameters
        """
        key = jax.random.PRNGKey(0)

        mesh_layout = self._parse_mesh_layout(device_mesh_layout)

        dummy_input = jnp.array(
            [[1 for _ in range(mesh_layout[1])] for _ in range(mesh_layout[0])]
        )

        abstract_variables = jax.eval_shape(self.init, key, dummy_input)
        if self.sharded:
            mesh = Mesh(
                devices=mesh_utils.create_device_mesh(mesh_layout),
                axis_names=("data", "model"),
            )

            rules = t5x_partitioning.standard_logical_axis_rules(
                activation_partitioning_dims=1,
                parameter_partitioning_dims=1,
                additional_rules=(
                    ("kv_length", None),
                    ("intermediate", "model"),
                ),
            )
            logical_state_spec = nn.get_partition_spec(abstract_variables)
            logical_state_sharding = nn.logical_to_mesh_sharding(
                logical_state_spec, mesh, rules
            )

            x_sharding = self.mesh_sharding(
                PartitionSpec("data", None), mesh
            )  # dimensions: (batch, length)

            if weights is not None:
                assert isinstance(
                    weights, dict
                ), f"weights must be a dict, got {type(weights)}"
                assert (
                    "params" in weights
                ), f"The key params not found in 'weights'. Got {weights.keys()}"

                if self.sharded:
                    params = {
                        "params": jax.tree_map(
                            lambda x, y: jax.device_put(x, y),
                            weights["params"],
                            logical_state_sharding["params"],
                        )
                    }
                else:
                    params = weights
            else:
                params = jax.jit(
                    self.init,
                    in_shardings=(
                        self.mesh_sharding(None, mesh),
                        x_sharding,
                    ),  # PRNG key and x
                    out_shardings=logical_state_sharding,
                )(key, dummy_input)
        else:
            params = self.init(key, dummy_input)

        return params

    def prepare_input(self, inputs, device_mesh_layout=(1, None), dtype=None):
        if self.sharded:
            mesh = Mesh(
                devices=mesh_utils.create_device_mesh(
                    self._parse_mesh_layout(device_mesh_layout)
                ),
                axis_names=("data", "model"),
            )
            inputs = jax.device_put(
                inputs, self.mesh_sharding(PartitionSpec("data", None), mesh)
            )
        if dtype is not None:
            inputs = jax.tree_map(lambda x: x.astype(dtype), inputs)
        return inputs

    def setup(self):
        self.model = FlaxMixtralModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)

        aux_loss = None
        if self.config.output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.config.num_local_experts,
                self.config.num_experts_per_tok,
                attention_mask,
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            if self.config.output_router_logits:
                output = (aux_loss,) + output
            return output

        return FlaxMoeCausalLMOutputWithPast(
            aux_loss=aux_loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


@add_start_docstrings(
    """
    The Mixtral Model transformer with a language modeling head (linear layer) on top.
    """,
    MIXTRAL_START_DOCSTRING,
)
# Copied from transformers.models.gptj.modeling_flax_gptj.FlaxGPTJForCausalLM with GPTJ->Mixtral
class FlaxMixtralForCausalLM(FlaxMixtralPreTrainedModel):
    module_class = FlaxMixtralForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since Mixtral uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


append_call_sample_docstring(
    FlaxMixtralForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutput,
    _CONFIG_FOR_DOC,
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)