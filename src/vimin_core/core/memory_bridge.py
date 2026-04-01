import numpy as np
from multiprocessing import shared_memory
import uuid
from typing import Dict, Any, Tuple, cast

class SharedMemoryManager:
    """
    Manages named shared memory blocks for zero-copy tensor transfer.
    """
    def __init__(self):
        self.allocated_blocks: Dict[str, Dict[str, Any]] = {}

    def create_buffer(self, name: str, shape: Tuple[int, ...], dtype: np.dtype) -> shared_memory.SharedMemory:
        """
        Allocates a shared memory block for a specific tensor shape/dtype.
        """
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        try:
            # Create new shared memory
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
            self.allocated_blocks[name] = {
                "shm": shm,
                "shape": shape,
                "dtype": dtype,
                "size": size
            }
            return shm
        except FileExistsError:
            # Attach to existing if it exists (cleanup might have failed)
            shm = shared_memory.SharedMemory(name=name)
            self.allocated_blocks[name] = {
                "shm": shm,
                "shape": shape,
                "dtype": dtype,
                "size": size
            }
            return shm

    def get_array(self, name: str) -> np.ndarray:
        """
        Returns a numpy array view backed by the shared memory buffer.
        """
        if name not in self.allocated_blocks:
            raise ValueError(f"Shared memory block '{name}' not found.")
            
        block = self.allocated_blocks[name]
        shm = cast(shared_memory.SharedMemory, block['shm'])
        # Explicit access to buf for linter safety
        return np.ndarray(block['shape'], dtype=block['dtype'], buffer=shm.buf)

    def cleanup(self):
        """
        Unlink and close all shared memory blocks.
        """
        for name, block in self.allocated_blocks.items():
            try:
                shm = cast(shared_memory.SharedMemory, block['shm'])
                shm.close()
                shm.unlink()
            except Exception as e:
                print(f"Error cleaning up shm {name}: {e}")
        self.allocated_blocks.clear()

    def __del__(self):
        self.cleanup()
