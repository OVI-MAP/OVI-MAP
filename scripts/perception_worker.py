#!/usr/bin/env python3
"""
Perception worker process for OVI-MAP split pipeline.

Runs in a Python 3.10+ conda environment (ovimap-perception-py310).
Loads VLModel (SigLIP/CLIP) and enters a request-response loop, communicating
with the reconstruction process via stdin/stdout + shared memory.

---------------------------------------------------------------------------
Protocol
---------------------------------------------------------------------------

Architecture:

    Reconstruction (Python 3.8)          Perception Worker (Python 3.10+)
    panoptic_mapping_.py                perception_worker.py
          │                                    │
          ├─ shared memory ──────────────────→ ├─ ovi_rgb_{0..3}
          │  (ring buffer, 4 slots,            │  uint8[H, W, 3]
          │   zero-copy RGB images)            │
          │                                    │
          ├─ stdin (pipe) ───────────────────→ ├─ JSON request + binary mask
          │                                    │
          │←─ stdout (pipe) ────────────────── ┤─ JSON response

Shared memory ring buffer
~~~~~~~~~~~~~~~~~~~~~~~~~
- Created by the reconstruction process via multiprocessing.shared_memory.
- 4 named blocks: ``ovi_rgb_0`` ... ``ovi_rgb_3``.
- Each holds a uint8 ndarray of shape (img_height, img_width, 3).
- Reconstruction writes the RGB image for frame ``f_i`` to slot ``f_i % 4``
  before sending any request that references that frame.
- Worker copies the RGB immediately upon receiving the request, so the slot
  is freed for the next frame that maps to the same slot index.

Request (recon -> worker, via stdin)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
JSON line followed by a binary blob::

    {"f_i": 42, "glo_inst_id": 17, "slot_idx": 2, "request_idx": 5,
     "bbox": [x1, y1, x2, y2], "mask_len": <N>}\n
    <N bytes of zlib-compressed uint8 obj_mask>

Fields:
    f_i             Frame index (for logging / response matching).
    glo_inst_id     Global instance ID in the reconstruction map.
    slot_idx        Ring buffer slot holding the RGB image (= f_i % 4).
    request_idx     Index into ``inst_dict[glo_inst_id]['feat']`` where the
                    result feature vector should be stored.
    bbox            [x1, y1, x2, y2] in pixel coordinates. The ROI passed
                    to VLModel.encode_image_with_bbox().
    mask_len        Number of bytes in the binary obj_mask blob that
                    immediately follows the newline.

obj_mask:
    A boolean mask of shape (H, W), flattened to uint8 bytes and zlib-
    compressed. Decompressed by ``ipc_utils.decompress_mask()``.

Response (worker -> recon, via stdout)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
JSON line::

    {"glo_inst_id": 17, "f_i": 42, "request_idx": 5, "status": "ok",
     "feat_b64": "<base64-encoded float32 array>"}

On success:
    status = "ok", feat_b64 = base64(roi_feat.astype(np.float32).tobytes()).
    The feature vector dimension depends on the VL model
    (e.g. 1024 for siglip-l-16-384, 768 for clip-ViT-L/14).

On error:
    status = "error", error = "<exception message>".
    The reconstruction process leaves the placeholder as None in inst_dict
    and filters it out during post-processing.

Shutdown handshake
~~~~~~~~~~~~~~~~~~
Reconstruction sends::

    {"cmd": "shutdown"}

Worker replies and exits::

    {"cmd": "shutdown_ack"}

---------------------------------------------------------------------------
"""

import sys
import os
import json
import base64
import argparse
import logging
import time
import traceback

# ---------------------------------------------------------------------------
# Early validation: Python version
# ---------------------------------------------------------------------------
if sys.version_info < (3, 9):
    sys.exit(f"[FATAL] Perception worker requires Python >= 3.9, "
             f"got {sys.version_info.major}.{sys.version_info.minor}")

# ---------------------------------------------------------------------------
# Parse command-line arguments first (before heavy imports)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="OVI-MAP Perception Worker")
parser.add_argument("--shm-names", type=str, required=True,
                    help="Comma-separated shared memory block names, "
                         "e.g. 'ovi_rgb_0,ovi_rgb_1,ovi_rgb_2,ovi_rgb_3'")
parser.add_argument("--img-height", type=int, required=True,
                    help="Image height in pixels")
parser.add_argument("--img-width", type=int, required=True,
                    help="Image width in pixels")
parser.add_argument("--model-name", type=str, default="siglip-l-16-384",
                    help="VL model name (default: siglip-l-16-384)")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device for model inference (default: cuda)")
parser.add_argument("--scripts-dir", type=str, default=None,
                    help="Path to the scripts/ directory. "
                         "Defaults to the directory containing this script.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Configure logging (to stderr, so stdout stays clean for JSON protocol)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [perception] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

# ---------------------------------------------------------------------------
# Add scripts/ directory to Python path (for vl_models + ipc_utils imports)
# ---------------------------------------------------------------------------
if args.scripts_dir is None:
    args.scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, args.scripts_dir)

# ---------------------------------------------------------------------------
# Import dependencies (heavy — torch/transformers are loaded here)
# ---------------------------------------------------------------------------
try:
    import numpy as np

    # Suppress verbose transformers logging
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    from ipc_utils import attach_shm_slots, decompress_mask
    from vl_models import VLModel
except ImportError as e:
    logging.fatal("Failed to import required modules: %s", e)
    logging.fatal("Ensure the perception conda environment is active and "
                  "all packages are installed.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Shared memory: attach to ring buffer
# ---------------------------------------------------------------------------
def attach_rgb_slots():
    """Attach to all shared memory ring buffer slots."""
    names = [n.strip() for n in args.shm_names.split(",")]
    logging.info("Attaching to %d shared memory slots: %s", len(names), names)
    try:
        return attach_shm_slots(names, args.img_height, args.img_width)
    except FileNotFoundError as e:
        logging.fatal("Shared memory block not found: %s", e)
        logging.fatal("Make sure the reconstruction process created the "
                      "shared memory blocks before launching the worker.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# VL model initialization
# ---------------------------------------------------------------------------
def init_vl_model():
    """Load the VL model onto the specified device."""
    logging.info("Loading VL model '%s' on device '%s' (img=%dx%d)...",
                 args.model_name, args.device, args.img_height, args.img_width)
    t0 = time.time()
    try:
        model = VLModel(
            model_name=args.model_name,
            img_size=(args.img_height, args.img_width),
            device=args.device,
        )
    except Exception as e:
        logging.fatal("Failed to load VL model: %s", e)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    logging.info("VL model loaded in %.1f seconds", time.time() - t0)
    return model


# ---------------------------------------------------------------------------
# Request processing
# ---------------------------------------------------------------------------
def process_request(request, rgb_views, vl_model: VLModel):
    """Process a single feature extraction request.

    Args:
        request: dict with keys f_i, glo_inst_id, slot_idx, request_idx,
                 bbox, mask_len.
        rgb_views: dict mapping slot_idx -> (SharedMemory, np.ndarray view).
        vl_model: VLModel instance.

    Returns:
        tuple: (response_dict, gpu_time_seconds)
    """
    t0 = time.time()
    f_i = request["f_i"]
    glo_inst_id = request["glo_inst_id"]
    slot_idx = request["slot_idx"]
    request_idx = request["request_idx"]
    bbox = tuple(request["bbox"])  # [x1, y1, x2, y2]
    mask_len = request["mask_len"]

    # Read compressed mask from stdin (binary, exactly mask_len bytes)
    mask_data = sys.stdin.buffer.read(mask_len)
    if len(mask_data) != mask_len:
        raise IOError(f"Expected {mask_len} bytes of mask data, got {len(mask_data)}")
    # Decompress the obj_mask
    obj_mask = decompress_mask(mask_data, (args.img_height, args.img_width))

    # Read RGB image from shared memory
    _, rgb_view = rgb_views[slot_idx]
    # Copy to avoid holding a view that could be overwritten by the main process
    rgb_img = rgb_view.copy()

    # Extract VL feature
    roi_feat = vl_model.encode_image_with_bbox(rgb_img, obj_mask, bbox)

    gpu_time = time.time() - t0

    # Encode feature as base64
    feat_b64 = base64.b64encode(roi_feat.astype(np.float32).tobytes()).decode("ascii")

    return {
        "glo_inst_id": glo_inst_id,
        "f_i": f_i,
        "request_idx": request_idx,
        "status": "ok",
        "feat_b64": feat_b64,
    }, gpu_time


# ---------------------------------------------------------------------------
# Main event loop
# ---------------------------------------------------------------------------
def main():
    rgb_views = attach_rgb_slots()
    vl_model = init_vl_model()

    # Signal readiness to the main process
    # (the main process checks that the worker responds to the first request)
    logging.info("Perception worker ready, waiting for requests...")

    request_count = 0
    error_count = 0
    frame_times = {}  # f_i -> list of gpu_time per request

    try:
        for line in sys.stdin.buffer:
            line = line.decode("utf-8").strip()
            if not line:
                continue

            request = json.loads(line)

            # Handle shutdown command
            if request.get("cmd") == "shutdown":
                # Log per-frame GPU time summary
                if frame_times:
                    total_time = sum(sum(t) for t in frame_times.values())
                    total_reqs = sum(len(t) for t in frame_times.values())
                    num_frames = len(frame_times)
                    logging.info("=== GPU inference timing ===")
                    logging.info("Requests: %d (errors: %d)", request_count, error_count)
                    logging.info("Frames with features: %d", num_frames)
                    logging.info("Avg GPU time per request: %.1f ms", total_time / total_reqs * 1000)
                    logging.info("Avg GPU time per frame: %.1f ms (avg %.1f requests/frame)",
                                 total_time / num_frames * 1000,
                                 total_reqs / num_frames)
                logging.info("Received shutdown signal. "
                             "Processed %d requests (%d errors).",
                             request_count, error_count)
                response = json.dumps({"cmd": "shutdown_ack"})
                sys.stdout.write(response + "\n")
                sys.stdout.flush()
                break

            # Handle regular feature extraction request
            try:
                response, gpu_time = process_request(request, rgb_views, vl_model)
                frame_times.setdefault(request["f_i"], []).append(gpu_time)
                request_count += 1
            except Exception as e:
                logging.error("Error processing request for inst %s frame %s: %s",
                              request.get("glo_inst_id", "?"),
                              request.get("f_i", "?"), e)
                traceback.print_exc(file=sys.stderr)
                error_count += 1
                response = {
                    "glo_inst_id": request.get("glo_inst_id", 0),
                    "f_i": request.get("f_i", -1),
                    "request_idx": request.get("request_idx", -1),
                    "status": "error",
                    "error": str(e),
                }

            # Send response back to the main process
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

    except BrokenPipeError:
        logging.warning("Main process closed stdin (broken pipe). Exiting.")
    except KeyboardInterrupt:
        logging.info("Worker interrupted by signal.")
    finally:
        # Close shared memory attachments and unregister from resource tracker
        from multiprocessing import resource_tracker as _rt
        for slot_idx, (shm, _) in rgb_views.items():
            try:
                shm.close()
                try:
                    _rt.unregister(shm._name, "shared_memory")
                except (FileNotFoundError, KeyError):
                    pass
            except Exception:
                pass
        logging.info("Perception worker exiting.")


if __name__ == "__main__":
    main()
