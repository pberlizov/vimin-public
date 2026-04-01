import numpy as np
from typing import Dict, Tuple, Optional

class KVCacheManager:
    """
    Manages Key-Value tensors for autoregressive transformer inference.
    Prevents re-computation of past history.
    """
    def __init__(self, num_layers: int, head_dim: int, num_heads: int):
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_heads = num_heads
        
        # Cache Store: LayerIndex -> (Key, Value)
        # Shapes usually: (batch, num_heads, seq_len, head_dim)
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.current_seq_len = 0
        
    def get_past_key_values(self, layer_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve cached KV for a specific layer"""
        return self.cache.get(layer_idx)
        
    def update(self, layer_idx: int, key: np.ndarray, value: np.ndarray):
        """
        Update cache for a layer.
        Expects key/value to be the full sequence or the new chunk depending on model implementation.
        For ease of use with ONNX, we usually pass the *full* updated KV back to the model next turn,
        or we concatenate if the model returns only the new KV.
        
        Here we assume the model returns the NEW KV state (concatenated).
        """
        self.cache[layer_idx] = (key, value)
        
    def advance(self, seq_len: int) -> None:
        """Advance the sequence-length counter after a forward pass."""
        self.current_seq_len += seq_len

    def reset(self):
        """Clear cache for new conversation"""
        self.cache.clear()
        self.current_seq_len = 0
        
    def get_flat_outputs(self, batch_size: int = 1) -> Dict[str, np.ndarray]:
        """
        Flatten cache for ONNX inputs.
        Naming convention: past_key_values.{layer}.key, past_key_values.{layer}.value
        """
        inputs = {}
        for i in range(self.num_layers):
            if i in self.cache:
                k, v = self.cache[i]
                inputs[f"past_key_values.{i}.key"] = k
                inputs[f"past_key_values.{i}.value"] = v
            else:
                # Initialize zeros if not present (for first run)
                # Shape: (batch, num_key_value_heads, 0, head_dim) 
                # Note: seq_len is 0 for the first token.
                shape = (batch_size, self.num_heads, 0, self.head_dim)
                inputs[f"past_key_values.{i}.key"] = np.zeros(shape, dtype=np.float32)
                inputs[f"past_key_values.{i}.value"] = np.zeros(shape, dtype=np.float32)
        return inputs
