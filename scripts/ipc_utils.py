"""
IPC utilities for OVI-MAP split-pipeline communication.

Shared between the reconstruction process (Python 3.8) and the perception
worker process (Python 3.10+). Uses only standard library + numpy.
"""

import zlib
import numpy as np
from multiprocessing import shared_memory

# ---------------------------------------------------------------------------
# Shared memory ring buffer for RGB images
# ---------------------------------------------------------------------------

# Ring buffer slot naming convention
SHM_RGB_PREFIX = "ovi_rgb"
NUM_SLOTS = 4


def create_shm_slots(img_h, img_w, num_slots=NUM_SLOTS):
    """Create NUM_SLOTS SharedMemory blocks for RGB images.

    Each slot holds a uint8 array of shape (img_h, img_w, 3).

    Returns:
        list of (slot_name, SharedMemory, np.ndarray) tuples.
        The np.ndarray is a writeable view into the shared memory buffer.
    """
    nbytes = img_h * img_w * 3
    slots = []
    for i in range(num_slots):
        name = f"{SHM_RGB_PREFIX}_{i}"
        # Clean up stale blocks from a previous crashed run
        try:
            stale = shared_memory.SharedMemory(name=name)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass
        shm = shared_memory.SharedMemory(create=True, size=nbytes, name=name)
        view = np.ndarray((img_h, img_w, 3), dtype=np.uint8, buffer=shm.buf)
        slots.append((name, shm, view))
    return slots


def attach_shm_slots(names, img_h, img_w):
    """Attach to existing SharedMemory blocks and return numpy array views.

    Args:
        names: list of shared memory block names.
        img_h, img_w: image dimensions.

    Returns:
        dict mapping slot_idx (int) to (SharedMemory, np.ndarray).
    """
    views = {}
    for name in names:
        slot_idx = int(name.rsplit("_", 1)[-1])
        shm = shared_memory.SharedMemory(name=name)
        view = np.ndarray((img_h, img_w, 3), dtype=np.uint8, buffer=shm.buf)
        views[slot_idx] = (shm, view)
    return views


def cleanup_shm_slots(slots):
    """Close and unlink all shared memory blocks in *slots*.

    Args:
        slots: iterable of (name, SharedMemory, np.ndarray) tuples
               (from create_shm_slots).
    """
    for name, shm, _ in slots:
        try:
            shm.close()
            shm.unlink()
        except (FileNotFoundError, OSError):
            pass


# ---------------------------------------------------------------------------
# obj_mask compression
# ---------------------------------------------------------------------------

def compress_mask(mask):
    """zlib-compress a boolean or uint8 numpy mask.

    Args:
        mask: np.ndarray, dtype=bool or uint8, shape (H, W).

    Returns:
        bytes: compressed representation.
    """
    if mask.dtype == bool:
        mask = mask.astype(np.uint8)
    return zlib.compress(mask.tobytes())


def decompress_mask(data, shape):
    """Decompress and reshape a mask.

    Args:
        data: bytes from compress_mask.
        shape: (H, W) tuple.

    Returns:
        np.ndarray of dtype=bool, shape=shape.
    """
    flat = np.frombuffer(zlib.decompress(data), dtype=np.uint8)
    return flat.reshape(shape).astype(bool)


# ---------------------------------------------------------------------------
# Simple smoke-test (run directly to verify IPC primitives work)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test mask compression roundtrip
    print("Testing mask compression roundtrip...")
    H, W = 480, 640
    for _ in range(10):
        mask = np.random.rand(H, W) > 0.7
        compressed = compress_mask(mask)
        restored = decompress_mask(compressed, (H, W))
        assert np.array_equal(mask, restored), "Mask roundtrip failed!"
    print(f"  OK - 10 random {H}x{W} masks roundtripped successfully")

    # Test shared memory create + attach
    print("Testing shared memory create + attach...")
    slots = create_shm_slots(H, W, num_slots=2)
    for i, (name, shm, view) in enumerate(slots):
        assert view.shape == (H, W, 3), f"Wrong shape: {view.shape}"
        view[:, :, :] = i  # fill with slot index
    # Attach by name
    names = [s[0] for s in slots]
    attached = attach_shm_slots(names, H, W)
    for slot_idx, (shm2, view2) in attached.items():
        assert view2[0, 0, 0] == slot_idx, f"Wrong value in slot {slot_idx}"
    cleanup_shm_slots(slots)
    print("  OK - shared memory create + attach + cleanup works")

    print("All IPC smoke tests passed!")
