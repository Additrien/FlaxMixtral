# coding=utf-8
# Copyright 2024 Mistral AI and the HuggingFace Inc. team. All rights reserved.
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
"""Flax Mistral3 model."""

from typing import List, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPast,
    FlaxCausalLMOutput,
    FlaxCausalLMOutputWithCrossAttentions,
)
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, logging
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_mistral3 import Mistral3Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Mistral3Config"
_REAL_CHECKPOINT_FOR_DOC = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
_CHECKPOINT_FOR_DOC = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

MISTRAL3_START_DOCSTRING = r"""

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
        config ([`Mistral3Config`]): Model configuration class with all the parameters of the model.
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

MISTRAL3_INPUTS_DOCSTRING = r"""
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


# Copied from transformers.models.mistral.modeling_flax_mistral.FlaxMistralRMSNorm
class FlaxMistral3RMSNorm(nn.Module):
    config: Mistral3Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.epsilon = self.config.rms_norm_eps
        self.weight = self.param("weight", lambda _, shape: jnp.ones(shape), self.config.hidden_size)

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)
        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)


class FlaxMistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    config: Mistral3Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        hidden_size = self.config.vision_config.hidden_size
        self.spatial_merge_size = self.config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.merging_layer = nn.Dense(
            hidden_size, 
            use_bias=False, 
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
        )

    def __call__(self, image_features: jnp.ndarray, image_sizes: jnp.ndarray) -> jnp.ndarray:
        # Print input shapes for debugging
        print(f"FlaxMistral3PatchMerger input shapes: image_features={image_features.shape}, image_sizes={image_sizes.shape}")
        
        # Calculate patch dimensions
        image_sizes = [(image_size[0] // self.patch_size, image_size[1] // self.patch_size) for image_size in image_sizes]
        
        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]
        
        # Use vectorized implementation with jax.lax for better performance
        permuted_tensors = []
        start_idx = 0
        
        for i, num_tokens in enumerate(tokens_per_image):
            # Extract the tokens for this image
            image_tokens = image_features[start_idx:start_idx+num_tokens]
            start_idx += num_tokens
            
            # Reshape tokens into 2D grid
            h, w = image_sizes[i]
            # Reshape to [channels, height, width] and add batch dimension
            image_grid = image_tokens.reshape(h, w, d)
            image_grid = jnp.transpose(image_grid, (2, 0, 1))
            image_grid = image_grid[None, ...]  # Add batch dimension [1, C, H, W]
            
            # Calculate output dimensions
            out_h = (h - self.spatial_merge_size) // self.spatial_merge_size + 1
            out_w = (w - self.spatial_merge_size) // self.spatial_merge_size + 1
            
            if out_h > 0 and out_w > 0:
                # Use jax.lax.conv_general_dilated_patches for efficient patch extraction
                patches = jax.lax.conv_general_dilated_patches(
                    image_grid,
                    filter_shape=(self.spatial_merge_size, self.spatial_merge_size),
                    window_strides=(self.spatial_merge_size, self.spatial_merge_size),
                    padding='VALID'
                )
                
                # Reshape to flatten spatial dimensions (out_h*out_w, C*spatial_merge_size*spatial_merge_size)
                patches = patches.reshape(patches.shape[0] * patches.shape[1], -1)
                permuted_tensors.append(patches)
            
        # Concatenate all processed features
        if permuted_tensors:
            image_features = jnp.concatenate(permuted_tensors, axis=0)
            # Apply the linear projection
            image_features = self.merging_layer(image_features)
            
            # Print output shape for debugging
            print(f"FlaxMistral3PatchMerger output shape: {image_features.shape}")
            
            return image_features
        else:
            # Handle empty case
            return jnp.zeros((0, self.config.vision_config.hidden_size), dtype=self.dtype)


class FlaxMistral3MultiModalProjector(nn.Module):
    """
    Multimodal projector for Mistral3 to project vision features into text space
    """
    
    config: Mistral3Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.norm = FlaxMistral3RMSNorm(self.config, dtype=self.dtype)
        self.patch_merger = FlaxMistral3PatchMerger(self.config, dtype=self.dtype)
        
        # Determine number of feature layers
        num_feature_layers = 1 if isinstance(self.config.vision_feature_layer, int) else len(self.config.vision_feature_layer)
        
        # Create projection layers
        self.linear_1 = nn.Dense(
            self.config.text_config.hidden_size,
            use_bias=self.config.multimodal_projector_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
        )
        
        self.act = ACT2FN[self.config.projector_hidden_act]
        
        self.linear_2 = nn.Dense(
            self.config.text_config.hidden_size,
            use_bias=self.config.multimodal_projector_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
        )
    
    def __call__(self, image_features: jnp.ndarray, image_sizes: jnp.ndarray):
        # Print input shapes for debugging
        print(f"FlaxMistral3MultiModalProjector input shapes: image_features={image_features.shape}, image_sizes={image_sizes.shape}")
        
        # Apply normalization
        image_features = self.norm(image_features)
        
        # Process through patch merger
        image_features = self.patch_merger(image_features, image_sizes)
        
        # Apply projection layers
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        
        # Print output shape for debugging
        print(f"FlaxMistral3MultiModalProjector output shape: {hidden_states.shape}")
        
        return hidden_states


class FlaxMistral3PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Mistral3Config
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: Mistral3Config,
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

        print(f"FlaxMistral3PreTrainedModel initializing with input_shape: {input_shape}")
        
        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        random_params = module_init_outputs["params"]

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
        return init_variables["cache"]


class FlaxMistral3ForConditionalGenerationModule(nn.Module):
    """Flax Mistral3 model for conditional generation (multimodal with text and vision)"""
    
    config: Mistral3Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Import vision and language models dynamically to avoid circular imports
        from ...models.auto.modeling_flax_auto import FlaxAutoModel, FlaxAutoModelForCausalLM
        
        # Initialize vision tower if needed
        if self.config.vision_config is not None:
            self.vision_tower = FlaxAutoModel.from_config(
                self.config.vision_config,
                dtype=self.dtype
            )
        else:
            self.vision_tower = None
            
        # Initialize language model
        self.language_model = FlaxAutoModelForCausalLM.from_config(
            self.config.text_config,
            dtype=self.dtype
        )
        
        # Initialize multimodal projector
        if self.config.vision_config is not None:
            self.multi_modal_projector = FlaxMistral3MultiModalProjector(
                self.config,
                dtype=self.dtype
            )
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)
    
    def get_image_features(
        self,
        pixel_values: jnp.ndarray,
        vision_feature_layer: Union[int, List[int]],
        image_sizes: jnp.ndarray,
        deterministic: bool = True,
        **kwargs,
    ):
        """
        Obtains image hidden states from the vision tower and applies multimodal projection.
        
        Args:
            pixel_values (`jnp.ndarray` of shape `(batch_size, channels, height, width)`):
                The tensors corresponding to the input images.
            vision_feature_layer (`int` or `list`):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated.
            image_sizes (`jnp.ndarray`):
                Tensor containing the image sizes.
            deterministic (`bool`, *optional*, defaults to `True`):
                Whether to perform deterministic inference.
                
        Returns:
            image_features (`jnp.ndarray`): Image feature tensor.
        """
        # Print input for debugging
        print(f"FlaxMistral3ForConditionalGeneration.get_image_features inputs: pixel_values={pixel_values.shape}, image_sizes={image_sizes.shape}")
        
        # Filter out None kwargs
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Get all hidden states from vision tower
        image_outputs = self.vision_tower(
            pixel_values, 
            image_sizes=image_sizes,
            output_hidden_states=True,
            deterministic=deterministic,
            **kwargs
        )
        
        # Extract features from specified layers
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            # Concatenate features from multiple layers
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_image_feature = jnp.concatenate(hs_pool, axis=-1)
        
        # Process image features through projector
        # In PyTorch: selected_image_feature.squeeze(0)
        # In JAX, we need to be careful with squeeze to ensure we know which dimension is being removed
        if selected_image_feature.shape[0] == 1:
            selected_image_feature = selected_image_feature[0]  # Remove batch dimension
        
        image_features = self.multi_modal_projector(selected_image_feature, image_sizes)
        
        # Print output for debugging
        print(f"FlaxMistral3ForConditionalGeneration.get_image_features output: {image_features.shape}")
        
        return image_features
    
    def __call__(
        self,
        input_ids: jnp.ndarray = None,
        pixel_values: jnp.ndarray = None,
        attention_mask: jnp.ndarray = None,
        position_ids: jnp.ndarray = None,
        past_key_values: dict = None,
        inputs_embeds: jnp.ndarray = None,
        vision_feature_layer: Union[int, List[int]] = None,
        labels: jnp.ndarray = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        deterministic: bool = True,
        image_sizes: jnp.ndarray = None,
        init_cache: bool = False,
        **kwargs,
    ):
        """
        Main forward pass for the Mistral3 model with both text and image inputs.
        
        Args:
            input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            pixel_values (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
                The pixel values of the images.
            attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            position_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
            past_key_values (`Dict[str, jnp.ndarray]`, *optional*):
                Pre-computed hidden-states for faster auto-regressive decoding.
            inputs_embeds (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Alternative to input_ids: embedded representation of input sequence.
            vision_feature_layer (`int` or `list`, *optional*):
                Layer(s) to extract vision features from.
            labels (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing language modeling loss.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` are returned and can be used for faster decoding.
            output_attentions (`bool`, *optional*):
                Whether to return attention weights of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether to return a ModelOutput instead of a tuple.
            deterministic (`bool`, *optional*, defaults to `True`):
                Whether to perform deterministic inference.
            image_sizes (`jnp.ndarray`, *optional*):
                Sizes of input images.
            init_cache (`bool`, *optional*, defaults to `False`):
                Whether to initialize the cache for auto-regressive decoding.
        """
        # Print input shapes for debugging
        print(f"FlaxMistral3ForConditionalGeneration inputs: input_ids={None if input_ids is None else input_ids.shape}, "
              f"pixel_values={None if pixel_values is None else pixel_values.shape}")
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        
        # Check input validity
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both pixel_values and inputs_embeds at the same time")
        
        # Handle dynamic shapes for image processing
        if pixel_values is not None:
            # Assert shapes match expected dimensions
            assert pixel_values.ndim == 4, f"Expected pixel_values to have 4 dimensions, got {pixel_values.ndim}"
            if image_sizes is None:
                # Default to full image size if not provided
                b, c, h, w = pixel_values.shape
                image_sizes = jnp.array([[h, w]] * b)
                
        # Get embeddings from tokens if not provided directly
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        image_features = None
        # Process image inputs if provided
        if pixel_values is not None:
            # Get image features from vision tower and projector
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                image_sizes=image_sizes,
                deterministic=deterministic,
            )
            
            # More efficient image feature integration
            special_image_mask = jnp.equal(input_ids, self.config.image_token_index)
            
            # Check shape compatibility
            n_image_tokens = jnp.sum(special_image_mask.astype(jnp.int32))
            n_image_features = image_features.shape[0]
            
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features: {n_image_features}"
                )
            
            # Efficient update using functional indexing
            batch_indices, seq_indices = jnp.nonzero(special_image_mask, size=n_image_features)
            
            # Function to update embeddings with image features
            def update_embeds(embeds, b_idx, s_idx, img_features):
                for i in range(img_features.shape[0]):
                    embeds = embeds.at[b_idx[i], s_idx[i]].set(img_features[i])
                return embeds
            
            # Conditionally update embeddings only if there are image tokens
            inputs_embeds = jax.lax.cond(
                n_image_tokens > 0,
                lambda: update_embeds(inputs_embeds, batch_indices, seq_indices, image_features),
                lambda: inputs_embeds
            )
        
        # Forward pass through language model
        lm_outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
            init_cache=init_cache,
            **kwargs,
        )
        
        logits = lm_outputs[0]
        
        # Calculate loss if labels provided with improved vectorized implementation
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            
            if attention_mask is not None:
                # Get the shifted mask for valid positions (exclude padding)
                shift_mask = attention_mask[:, 1:].astype(jnp.float32)
                
                # Compute per-token loss
                token_loss = optax.softmax_cross_entropy_with_integer_labels(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1)
                ).reshape(shift_labels.shape)
                
                # Apply mask and compute mean over valid positions
                masked_loss = token_loss * shift_mask
                loss = jnp.sum(masked_loss) / jnp.maximum(jnp.sum(shift_mask), 1.0)
            else:
                # Simpler case without mask
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1)
                ).mean()
        
        # Return as dict or tuple based on return_dict flag
        if not return_dict:
            output = (logits,) + lm_outputs[1:]
            return (loss,) + output if loss is not None else output
        
        # Print output shapes for debugging
        print(f"FlaxMistral3ForConditionalGeneration output: logits={logits.shape}")
        
        return FlaxCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=lm_outputs.past_key_values,
            hidden_states=lm_outputs.hidden_states,
            attentions=lm_outputs.attentions,
            cross_attentions=None,  # Mistral3 doesn't have cross-attention
        )


@add_start_docstrings(
    """The Mistral3 model with a vision encoder and a language modeling head on top.""",
    MISTRAL3_START_DOCSTRING,
)
class FlaxMistral3ForConditionalGeneration(FlaxMistral3PreTrainedModel):
    module_class = FlaxMistral3ForConditionalGenerationModule
    
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        max_length, 
        attention_mask=None,
        pixel_values=None,
        image_sizes=None,
        past_key_values=None,
        **kwargs
    ):
        """
        Prepare inputs for generation, handling both text and possible image inputs.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum sequence length for generation
            attention_mask: Attention mask for input tokens
            pixel_values: Optional image pixel values
            image_sizes: Optional image sizes
            past_key_values: Optional past key values for faster generation
            
        Returns:
            Dictionary of prepared inputs for generation
        """
        batch_size = input_ids.shape[0]
        
        # Initialize or get cache for faster generation
        if past_key_values is None:
            past_key_values = self.init_cache(batch_size, max_length)
        
        # Initialize or extend attention mask
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, input_ids.shape[1]), dtype=jnp.int32)
        elif past_key_values is not None and input_ids.shape[1] > 1:
            # Extend attention mask for generation if needed (not first step)
            attention_mask = jnp.concatenate([
                attention_mask,
                jnp.ones((batch_size, 1), dtype=jnp.int32)
            ], axis=1)
        
        # Create position IDs
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), 
            input_ids.shape
        )
        
        # Bundle inputs
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
        }
        
        # Only pass pixel_values on first step of generation
        is_first_step = input_ids.shape[1] == 1
        if is_first_step and pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes
        
        return model_inputs
    
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """
        Update inputs for next generation step.
        
        Args:
            model_outputs: Outputs from previous generation step
            model_kwargs: Current generation input arguments
            
        Returns:
            Updated model_kwargs for next step
        """
        # Update past key values for faster generation
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        
        # Update position IDs to point to next token
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        
        # Update attention mask if needed
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = jnp.concatenate(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        
        return model_kwargs


__all__ = [
    "FlaxMistral3ForConditionalGeneration",
]

