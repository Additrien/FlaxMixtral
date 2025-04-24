# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the Flax Llava model. """

import unittest
import tempfile
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.traverse_util import flatten_dict
import requests
from PIL import Image

from transformers import (
    LlavaConfig,
    AutoProcessor,
    is_flax_available,
    is_vision_available,
)
from transformers.testing_utils import require_flax, slow, torch_device, require_torch_gpu

from ...test_configuration_common import ConfigTester
from ...test_modeling_flax_common import FlaxModelTesterMixin, floats_tensor, ids_tensor


if is_flax_available():
    from transformers import FlaxLlavaForConditionalGeneration
else:
    FlaxLlavaForConditionalGeneration = None


class FlaxLlavaVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=0,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        text_config={
            "model_type": "llama",
            "seq_length": 7,
            "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 0,
        },
        is_training=True,
        vision_config={
            "batch_size": 12,
            "image_size": 30,
            "patch_size": 2,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 336
        self.encoder_seq_length = 231

    def get_config(self):
        return LlavaConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = jnp.ones_like(input_ids)
        
        input_ids = jnp.array(input_ids) 
        input_ids = input_ids.at[:, 1].set(config.image_token_index)
        
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_flax
class FlaxLlavaForConditionalGenerationModelTest(FlaxModelTesterMixin, unittest.TestCase):
    """
    Test class for FlaxLlavaForConditionalGeneration.
    """
    all_model_classes = (FlaxLlavaForConditionalGeneration,) if is_flax_available() else ()
    
    def setUp(self):
        self.model_tester = FlaxLlavaVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlavaConfig, has_text_modality=False)
        
    def test_config(self):
        # Test specific to LlavaConfig
        config = LlavaConfig()
        
        # Basic validation of config attributes
        self.assertTrue(hasattr(config, "text_config"))
        self.assertTrue(hasattr(config, "vision_config"))
        self.assertTrue(hasattr(config, "ignore_index"))
        self.assertTrue(hasattr(config, "image_token_index"))
        
        # Test to_dict method
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn("text_config", config_dict)
        self.assertIn("vision_config", config_dict)
        
        # Test from_dict method
        config_from_dict = LlavaConfig.from_dict(config_dict)
        self.assertDictEqual(config.to_dict(), config_from_dict.to_dict())
        
        # Test save and load functions
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            loaded_config = LlavaConfig.from_pretrained(tmp_dir)
            self.assertDictEqual(config.to_dict(), loaded_config.to_dict())
        
    # Override some tests from FlaxModelTesterMixin that don't apply to this model
    def test_forward_signature(self):
        pass
    
    def test_jit_compilation(self):
        pass


@require_flax
@require_torch_gpu
class FlaxLlavaForConditionalGenerationIntegrationTest(unittest.TestCase):
    """
    Integration tests for FlaxLlavaForConditionalGeneration.
    """
    
    @slow
    def test_integration_generate_output(self):
        """
        Test generate output matches PyTorch model.
        Based on test_small_model_integration_test_llama in PyTorch.
        """
        model_id = "llava-hf/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(model_id)

        # Load the Flax model
        dtype = jnp.float16 
        try:
            fx_model = FlaxLlavaForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
        except OSError as e:
            self.skipTest(f"Skipping test because Flax model loading failed: {e}")
            return 

        # Prepare inputs from the reference PyTorch test
        prompt = "USER: <image>\nWhat are the things I should be cautious about when I visit this place?\nASSISTANT:"
        image_file = "https://llava-vl.github.io/static/images/view.jpg"
        
        try:
            raw_image = Image.open(requests.get(image_file, stream=True).raw)
        except Exception as e:
            self.skipTest(f"Skipping test because image could not be downloaded: {e}")
            return

        # Use return_tensors="np" for Flax
        inputs = processor(text=prompt, images=raw_image, return_tensors="np")

        # Generate output
        output_ids = fx_model.generate(
            pixel_values=inputs["pixel_values"].astype(dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=900,
            do_sample=False
        )

        # Decode and compare
        generated_text = processor.batch_decode(output_ids.sequences, skip_special_tokens=True)[0] 

        EXPECTED_DECODED_TEXT = "USER:  \nWhat are the things I should be cautious about when I visit this place?\nASSISTANT: When visiting this place, which is a pier or dock extending over a body of water, there are a few things to be cautious about. First, be aware of the weather conditions, as sudden changes in weather can make the pier unsafe to walk on. Second, be mindful of the water depth and any potential hazards, such as submerged rocks or debris, that could cause accidents or injuries. Additionally, be cautious of the presence of wildlife, such as birds or fish, and avoid disturbing their natural habitats. Lastly, be aware of any local regulations or guidelines for the use of the pier, as some areas may be restricted or prohibited for certain activities."

        self.assertEqual(generated_text, EXPECTED_DECODED_TEXT)

    @slow
    def test_integration_generate_output_batched(self):
        """
        Test generate output for a batch of inputs.
        Based on test_small_model_integration_test_llama_batched in PyTorch.
        """
        model_id = "llava-hf/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(model_id)

        # Load the Flax model
        dtype = jnp.float16
        try:
            fx_model = FlaxLlavaForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
        except OSError as e:
            self.skipTest(f"Skipping test because Flax model loading failed: {e}")
            return

        prompts = [
            "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
            "USER: <image>\nWhat is this?\nASSISTANT:",
        ]
        
        try:
            image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
            image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
        except Exception as e:
            self.skipTest(f"Skipping test because images could not be downloaded: {e}")
            return

        # Prepare batch inputs
        inputs = processor(prompts, images=[image1, image2], return_tensors="np", padding=True)

        # Generate output
        output_ids = fx_model.generate(
            pixel_values=inputs["pixel_values"].astype(dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=20,
            do_sample=False
        )

        # Decode and compare
        generated_texts = processor.batch_decode(output_ids.sequences, skip_special_tokens=True)

        EXPECTED_DECODED_TEXTS = [
            'USER:  \nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT: When visiting this place, which appears to be a dock or pier extending over a body of water', 
            'USER:  \nWhat is this?\nASSISTANT: The image features two cats lying down on a pink couch. One cat is located on'
        ]

        self.assertEqual(generated_texts, EXPECTED_DECODED_TEXTS) 