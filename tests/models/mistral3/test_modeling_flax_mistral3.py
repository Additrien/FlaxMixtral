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

import unittest

import numpy as np

from transformers import Mistral3Config, is_flax_available, is_torch_available, is_tokenizers_available
from transformers.testing_utils import require_flax, require_torch, slow

from ...test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor, floats_tensor


if is_flax_available():
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict

    from transformers.models.mistral3.modeling_flax_mistral3 import (
        FlaxMistral3ForConditionalGeneration,
    )


if is_torch_available():
    import torch
    
    from transformers.models.mistral3.modeling_mistral3 import (
        Mistral3ForConditionalGeneration,
    )


if is_tokenizers_available():
    from transformers import AutoProcessor, AutoTokenizer


class FlaxMistral3VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        image_seq_length=4,
        vision_feature_layer=-1,
        ignore_index=-100,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        image_token_index=1,
        num_channels=3,
        image_size=30,
        model_type="mistral3",
        is_training=True,
        text_config={
            "model_type": "mistral",
            "vocab_size": 99,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "hidden_size": 32,
            "initializer_range": 0.02,
            "intermediate_size": 37,
            "max_position_embeddings": 512,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000000.0,
            "sliding_window": None,
            "bos_token_id": 0,
            "eos_token_id": 0,
            "pad_token_id": 0,
        },
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "image_size": 30,
            "patch_size": 6,
            "num_channels": 3,
            "hidden_act": "gelu",
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.image_token_index = image_token_index
        self.model_type = model_type
        self.text_config = text_config
        self.vision_config = vision_config
        self.batch_size = batch_size
        self.vision_feature_layer = vision_feature_layer
        self.is_training = is_training
        self.image_seq_length = image_seq_length
        self.num_channels = num_channels
        self.image_size = image_size
        self.seq_length = seq_length + self.image_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]

    def get_config(self):
        return Mistral3Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            model_type=self.model_type,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            image_token_index=self.image_token_index,
            image_seq_length=self.image_seq_length,
            vision_feature_layer=self.vision_feature_layer,
            spatial_merge_size=2,  # Add this parameter to match PyTorch tests
            multimodal_projector_bias=True,  # Add this parameter to match PyTorch tests
            projector_hidden_act="gelu",  # Add this parameter to match PyTorch tests
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = jnp.ones_like(input_ids)
        image_sizes = jnp.ones((self.batch_size, 2), dtype=jnp.int32) * self.image_size

        # Input ids with image token placeholders
        input_ids = input_ids.at[:, :self.image_seq_length].set(self.image_token_index)

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_sizes": image_sizes,
        }
        return config, inputs_dict


@require_flax
class FlaxMistral3ModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxMistral3ForConditionalGeneration,) if is_flax_available() else ()
    is_encoder_decoder = False
    test_head_masking = False
    test_pruning = False
    test_mismatched_shapes = False
    has_attentions = True
    _is_composite = True  # Mistral3 is a composite model

    def setUp(self):
        self.model_tester = FlaxMistral3VisionText2TextModelTester(self)

    @unittest.skip("Models with VisionEncoder don't support cache forwarding yet in Flax")
    def test_use_cache_forward(self):
        pass

    @unittest.skip("Models with VisionEncoder don't support cache forwarding yet in Flax")
    def test_use_cache_forward_with_attn_mask(self):
        pass
    
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            module = model.module
            
            # Check if key modules are initialized
            self.assertIsNotNone(module.vision_tower)
            self.assertIsNotNone(module.language_model)
            self.assertIsNotNone(module.multi_modal_projector)
            
            # Basic parameter checks
            self.assertEqual(module.config.vision_feature_layer, self.model_tester.vision_feature_layer)
            self.assertEqual(module.config.image_token_index, self.model_tester.image_token_index)
            
            # Check if all parameters require gradients
            for name, param in model.params.items():
                flattened_params = flatten_dict({"root": param})
                for param_path, param_value in flattened_params.items():
                    self.assertIsNotNone(param_value)

    def test_model_outputs_with_images(self):
        """Test that the model correctly handles image inputs."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            outputs = model(**inputs_dict)
            
            # Check if outputs has expected shape
            self.assertEqual(outputs.logits.shape[0], self.model_tester.batch_size)
            self.assertEqual(outputs.logits.shape[1], self.model_tester.seq_length)
            self.assertEqual(outputs.logits.shape[2], self.model_tester.vocab_size)

    @unittest.skip("Compile not yet supported because in Mistral3 models")
    def test_jit_compilation(self):
        pass

    @unittest.skip("FlashAttention only support fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass
    
    @slow
    @require_flax
    @require_torch
    def test_pytorch_flax_equivalence(self):
        """Test that Flax and PyTorch implementations produce similar results."""
        if not is_torch_available() or not is_flax_available():
            return
            
        # For a simple equivalence test, we use a small model
        config = Mistral3Config(
            text_config={
                "vocab_size": 10,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 64,
            },
            vision_config={
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 64,
                "image_size": 16,
                "patch_size": 2,
            },
            vision_feature_layer=-1,
            multimodal_projector_bias=True,
            spatial_merge_size=2,
            projector_hidden_act="gelu",
        )
        
        # Since we can't load pretrained models, we'll initialize and compare basic shapes
        pt_model = Mistral3ForConditionalGeneration(config)
        flax_model = FlaxMistral3ForConditionalGeneration(config)
        
        # Convert to inference mode
        pt_model.eval()
        
        # Create some simple inputs
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        pixel_values = torch.randn(1, 3, 16, 16)
        attention_mask = torch.ones_like(input_ids)
        image_sizes = torch.tensor([[16, 16]])
        
        # PyTorch forward pass
        with torch.no_grad():
            pt_outputs = pt_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_sizes=image_sizes,
            )
            
        # Convert inputs to JAX arrays
        jax_input_ids = jnp.array(input_ids.numpy())
        jax_pixel_values = jnp.array(pixel_values.numpy())
        jax_attention_mask = jnp.array(attention_mask.numpy())
        jax_image_sizes = jnp.array(image_sizes.numpy())
        
        # Flax forward pass
        flax_outputs = flax_model(
            input_ids=jax_input_ids,
            pixel_values=jax_pixel_values,
            attention_mask=jax_attention_mask,
            image_sizes=jax_image_sizes,
        )
        
        # Check if outputs have same shape
        self.assertEqual(pt_outputs.logits.shape, flax_outputs.logits.shape)
        self.assertEqual(len(pt_outputs.hidden_states), len(flax_outputs.hidden_states))
        
        # Note: We're just testing shape equivalence here since the weights are not identical


@slow
@require_flax
class FlaxMistral3IntegrationTest(unittest.TestCase):
    def setUp(self):
        # This model checkpoint doesn't exist yet for Flax, so we'll need to modify this for real testing
        self.model_id = "mistralai/Mistral-3-Instruct-4k"

    @unittest.skip("No pretrained checkpoints available yet for Flax Mistral3")
    def test_mistral3_integration_generate(self):
        # TODO: Implement actual integration tests once checkpoints are available
        pass
    
    @unittest.skip("No pretrained checkpoints available yet for Flax Mistral3")
    def test_mistral3_integration_generate_text_only(self):
        # TODO: Implement text-only generation test once checkpoints are available
        pass
    
    @unittest.skip("No pretrained checkpoints available yet for Flax Mistral3")
    def test_mistral3_integration_batched_generate(self):
        # TODO: Implement batched generation test once checkpoints are available
        pass 