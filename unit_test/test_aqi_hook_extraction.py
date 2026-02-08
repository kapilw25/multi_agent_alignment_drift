"""
Test AQI hook-based hidden state extraction for Phi2 models.

Usage:
  source venv_Agnt_Algnmt/bin/activate
  python -u unit_test/test_aqi_hook_extraction.py 2>&1 | tee logs/test_aqi_hooks.log
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'AQI', '0a_AQI_EVAL_utils', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np


def test_helper_functions():
    """Test _aqi_get_decoder_layers and _aqi_check_output_hidden_states are importable."""
    from aqi.aqi_dealign_xb_chi import _aqi_get_decoder_layers, _aqi_check_output_hidden_states, get_hidden_states_batch
    print("  _aqi_get_decoder_layers:", _aqi_get_decoder_layers)
    print("  _aqi_check_output_hidden_states:", _aqi_check_output_hidden_states)
    print("  get_hidden_states_batch:", get_hidden_states_batch)
    print("PASS: Helper functions importable")


def test_phi2_hook_extraction():
    """Test hook-based extraction on actual Phi2 model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from aqi.aqi_dealign_xb_chi import get_hidden_states_batch, _aqi_check_output_hidden_states, _aqi_get_decoder_layers

    model_name = "lxuechen/phi-2-sft"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).cuda().eval()

    # Verify hidden_states is indeed None
    device = next(model.parameters()).device
    supported = _aqi_check_output_hidden_states(model, tokenizer, device)
    print(f"  output_hidden_states supported: {supported}")
    assert not supported, "Expected Phi2 to NOT support output_hidden_states"

    # Verify decoder layers detection
    decoder_layers = _aqi_get_decoder_layers(model)
    print(f"  Decoder layers found: {len(decoder_layers)} (expected 32)")
    assert len(decoder_layers) == 32, f"Expected 32 decoder layers, got {len(decoder_layers)}"

    # Test batch extraction with hooks
    texts = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "What is the meaning of life?",
    ]
    print(f"\nExtracting hidden states from {len(texts)} texts...")
    result = get_hidden_states_batch(model, tokenizer, texts, batch_size=2, layer=-1, pooling_strategy='mean', device="cuda")

    print(f"  Result shape: {result.shape}")
    print(f"  Result dtype: {result.dtype}")
    assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"
    assert result.shape[0] == len(texts), f"Expected {len(texts)} samples, got {result.shape[0]}"
    assert result.shape[1] == 2560, f"Expected hidden_dim=2560 for Phi2, got {result.shape[1]}"
    assert not np.isnan(result).any(), "Result contains NaN values"
    assert not np.isinf(result).any(), "Result contains Inf values"

    print(f"\nPASS: Phi2 hook extraction produces valid embeddings: {result.shape}")

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()


def test_standard_model_no_hooks():
    """Test that standard models (e.g. GPT2-small) still use the normal path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from aqi.aqi_dealign_xb_chi import get_hidden_states_batch, _aqi_check_output_hidden_states

    model_name = "gpt2"
    print(f"\nLoading {model_name} (control test)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda().eval()

    device = next(model.parameters()).device
    supported = _aqi_check_output_hidden_states(model, tokenizer, device)
    print(f"  output_hidden_states supported: {supported}")
    assert supported, "Expected GPT2 to support output_hidden_states"

    texts = ["Hello world", "Testing standard path"]
    result = get_hidden_states_batch(model, tokenizer, texts, batch_size=2, layer=-1, pooling_strategy='mean', device="cuda")
    print(f"  Result shape: {result.shape}")
    assert result.shape[0] == 2
    assert result.shape[1] == 768  # GPT2 hidden_dim

    print("PASS: Standard model uses normal path correctly")

    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 60)
    print("Test: AQI Hook-Based Hidden State Extraction")
    print("=" * 60)

    print("\n--- Test 1: Import check ---")
    test_helper_functions()

    print("\n--- Test 2: Standard model (no hooks) ---")
    test_standard_model_no_hooks()

    print("\n--- Test 3: Phi2 hook extraction ---")
    test_phi2_hook_extraction()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
