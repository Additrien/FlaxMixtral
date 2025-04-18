# TODO: Add Mistral 3 Flax Implementation to Transformers

This list outlines the steps required to add the JAX/Flax implementation for the Mistral 3 model into a single `modeling_flax_mistral3.py` file, leveraging the existing PyTorch code.

## 1. Initial Code Structure Setup

-   [x] Create `src/transformers/models/mistral3/modeling_flax_mistral3.py`.
    -   [x] Copy contents from `src/transformers/models/mistral/modeling_flax_mistral.py`.
    -   [x] Rename classes/imports: `Mistral` -> `Mistral3`, `mistral` -> `mistral3`.
    -   [x] Import `Mistral3Config` from `.configuration_mistral3`.

## 2. Port Architecture Differences

-   [ ] Compare the PyTorch implementation (primarily `src/transformers/models/mistral3/modular_mistral3.py` and `src/transformers/models/mistral3/modeling_mistral3.py`) with the new `src/transformers/models/mistral3/modeling_flax_mistral3.py`.
-   [ ] Update the Flax layer modules (`FlaxMistral3Attention`, `FlaxMistral3MLP`, `FlaxMistral3RMSNorm`, `FlaxMistral3RotaryEmbedding`, `FlaxMistral3DecoderLayer`, `FlaxMistral3Module`, etc.) *within* `modeling_flax_mistral3.py` to match the specific architecture defined in the PyTorch version.
    -   [ ] Check attention mechanism details.
    -   [ ] Check MLP structure (activations, biases).
    -   [ ] Check normalization layer details.
    -   [ ] Check RoPE implementation details.
    -   [ ] Ensure correct configuration parameters are used.
    -   [ ] Look for potential use of shared Flax utilities from `modeling_flax_utils.py`.

## 3. Update Initialization and Registration

-   [x] Edit `src/transformers/models/mistral3/__init__.py`.
    -   [x] Add conditional imports for `FlaxMistral3Model`, `FlaxMistral3ForCausalLM` within a `try...except ImportError` block for Flax.
-   [x] Edit `src/transformers/models/auto/modeling_flax_auto.py`.
    -   [x] Add `"mistral3"` to the `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` OrderedDict.
    -   [x] Add `Mistral3Config` and `FlaxMistral3Model`, `FlaxMistral3ForCausalLM` to the `_MODEL_MAPPINGS` `OrderedDict`.
-   [x] Edit `src/transformers/models/auto/configuration_auto.py`.
    -   [x] Add `"mistral3"` to `MODEL_NAMES_MAPPING`.
    -   [x] Add `Mistral3Config` to `CONFIG_MAPPING`.

## 4. Weight Conversion

-   [ ] Adapt the existing PyTorch conversion script (`src/transformers/models/mistral3/convert_mistral3_weights_to_hf.py`) or create a specific Flax conversion utility.
    -   [ ] This script needs to load the PyTorch checkpoint (`.bin` or `.safetensors`).
    -   [ ] Map PyTorch layer names to Flax layer names (Flax often uses nested structures like `model/layers/0/attention/...`).
    -   [ ] Handle potential weight transpositions (e.g., Dense layer kernels).
    -   [ ] Save the converted weights in Flax format (typically msgpack, but check current best practices).
-   [ ] Convert at least one official checkpoint and host it on the Hub (e.g., under `mistralai/Mistral-3-Instruct-4k-flax`).

## 5. Testing

-   [ ] Create `tests/models/mistral3/test_modeling_flax_mistral3.py`.
    -   [ ] Copy structure from `tests/models/mistral/test_modeling_flax_mistral.py`.
    -   [ ] Rename classes/imports (`Mistral` -> `Mistral3`, `mistral` -> `mistral3`).
-   [ ] **Integration Tests:** Implement `FlaxMistral3ModelIntegrationTests` in the test file.
    -   [ ] Load the converted Flax checkpoint.
    -   [ ] Load the corresponding PyTorch checkpoint.
    -   [ ] Feed the same `input_ids` to both models.
    -   [ ] Assert that the `last_hidden_state` (and potentially other outputs like logits if testing `ForCausalLM`) are close (`torch.allclose` / `jnp.allclose` with appropriate `atol`). Use `RUN_SLOW=1 pytest ...` to run.
-   [ ] **Common Tests:** Ensure standard model tests pass.
    -   [ ] Run `pytest tests/models/mistral3/test_modeling_flax_mistral3.py`. Fix any failures in the common tests inherited from `FlaxModelTesterMixin`.
-   [ ] **Specific Tests:** Add tests to `FlaxMistral3ModelTest` for any unique architectural features of Mistral 3 not covered by common tests.

## 6. Documentation

-   [ ] Edit `docs/source/en/model_doc/mistral3.md`.
    -   [ ] Add `FlaxMistral3Model` and `FlaxMistral3ForCausalLM` to the list of available classes at the top.
    -   [ ] Include Flax-specific usage examples.
-   [ ] Review and complete docstrings in `modeling_flax_mistral3.py`. Ensure inputs, outputs, and config parameters are clearly documented.

## 7. Code Quality and Final Checks

-   [ ] Run `make style` to format the code.
-   [ ] Run `make quality` to check for linting errors, type hints, etc. Fix any reported issues.
-   [ ] Ensure all TODO comments in the code have been addressed.
-   [ ] Perform a final review of the code for clarity and correctness.

## 8. Pull Request and Hub Upload

-   [ ] Create a Pull Request on the Hugging Face `transformers` repository (if not already done).
-   [ ] Ensure CI checks pass on the PR.
-   [ ] Address any review comments.
-   [ ] Consider uploading the converted Flax weights to the Hub using `push_to_hub` (often done after merge or by HF staff).
