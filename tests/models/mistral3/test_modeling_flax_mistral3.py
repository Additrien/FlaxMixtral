# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the Flax Mistral3 model."""

import unittest

import numpy as np

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from transformers import (
    AutoProcessor,
    Mistral3Config,
    is_flax_available,
)
from transformers.testing_utils import (
    cleanup,
    require_flax,
    require_read_token,
    slow,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor, floats_tensor


if is_flax_available():
    from transformers import (
        FlaxMistral3ForConditionalGeneration,
    )


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
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = np.ones_like(input_ids)
        image_sizes = np.array(
            [[self.image_size, self.image_size]] * self.batch_size, dtype=np.int32
        )

        # Set image token indices appropriately
        input_ids_mask = input_ids == self.image_token_index
        input_ids = np.where(input_ids_mask, self.pad_token_id, input_ids)
        for i in range(self.batch_size):
            input_ids[i, :self.image_seq_length] = self.image_token_index

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
    _is_composite = True
    
    def setUp(self):
        self.model_tester = FlaxMistral3VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Mistral3Config, has_text_modality=False)

    def test_config(self):
        # Overwritten from test_configuration_common.py
        def check_config_can_be_init_without_params():
            config = self.config_tester.config_class()
            self.config_tester.parent.assertIsNotNone(config)

        self.config_tester.check_config_can_be_init_without_params = check_config_can_be_init_without_params
        self.config_tester.run_common_tests()

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            
            # Test that params are initialized properly
            flat_params = unfreeze(model.params)
            for param_name, param in flat_params.items():
                if "kernel" in param_name or "embedding" in param_name:
                    self.assertTrue(
                        -0.1 < float(param.mean()) < 0.1,
                        msg=f"Parameter {param_name} seems not properly initialized"
                    )

    # Test inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            
            inputs = {k: v for k, v in inputs_dict.items()}
            
            input_ids = inputs.pop("input_ids")
            pixel_values = inputs.pop("pixel_values", None)
            
            # Generate embedded inputs
            inputs_embeds = model.module.get_input_embeddings()(input_ids)
            
            # Call model with embedded inputs
            outputs = model(inputs_embeds=inputs_embeds, **inputs)
            self.assertIsNotNone(outputs)

    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            
            # Prepare inputs
            inputs = {k: v for k, v in inputs_dict.items()}
            input_ids = inputs.pop("input_ids")
            if "pixel_values" in inputs:
                inputs.pop("pixel_values")
            
            # Get embeddings module
            wte = model.module.get_input_embeddings()
            inputs_embeds = wte(input_ids)
            
            # Get outputs with input_ids
            outputs_ids = model(input_ids=input_ids, **inputs)
            
            # Get outputs with inputs_embeds
            outputs_embeds = model(inputs_embeds=inputs_embeds, **inputs)
            
            # Check outputs match
            diff = jnp.max(jnp.abs(outputs_ids.logits - outputs_embeds.logits))
            self.assertLessEqual(float(diff), 1e-4)

    def test_use_cache_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        max_decoder_length = 20
        
        for model_class in self.all_model_classes:
            model = model_class(config)
            
            # Get inputs
            input_ids = inputs_dict["input_ids"]
            attention_mask = inputs_dict["attention_mask"]
            
            # Initialize cache
            past_key_values = model.init_cache(input_ids.shape[0], max_decoder_length)
            
            # Create extended attention mask
            extended_attention_mask = jnp.ones((input_ids.shape[0], max_decoder_length), dtype=jnp.int32)
            
            # Create position ids for first pass (all except last token)
            position_ids = jnp.broadcast_to(
                jnp.arange(input_ids.shape[-1] - 1), (input_ids.shape[0], input_ids.shape[-1] - 1)
            )
            
            # First forward pass with cache
            outputs_cache = model(
                input_ids[:, :-1],
                attention_mask=extended_attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            
            # Create position ids for second pass (only the last token)
            position_ids = jnp.ones((input_ids.shape[0], 1), dtype=jnp.int32) * (input_ids.shape[-1] - 1)
            
            # Second forward pass with updated cache
            outputs_cache_next = model(
                input_ids[:, -1:],
                attention_mask=extended_attention_mask,
                past_key_values=outputs_cache.past_key_values,
                position_ids=position_ids,
            )
            
            # Full forward pass without cache
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Check that outputs are the same
            diff = float(jnp.max(jnp.abs(
                outputs_cache_next.logits[:, -1, :5] - outputs.logits[:, -1, :5]
            )))
            
            self.assertLessEqual(diff, 1e-3, msg=f"Max diff is {diff}")


@slow
@require_flax
class FlaxMistral3IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_checkpoint = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    def tearDown(self):
        cleanup()

    @require_read_token
    def test_mistral3_integration_generate_text_only(self):
        # Skip for now - requires implementation with actual models
        pass

    @require_read_token
    def test_mistral3_integration_generate(self):
        # Skip for now - requires implementation with actual models
        pass

    @require_read_token
    def test_mistral3_integration_batched_generate(self):
        # Skip for now - requires implementation with actual models
        pass