#!/usr/bin/env python3
"""
Direct V4L2 Frame Capture without OpenCV (cv2)
Captures uncompressed YUYV frames directly from Linux kernel V4L2 buffers
and saves them to a 16-bit TIFF image using pure NumPy and tifffile.
"""

import sys
import subprocess
import numpy as np
import tifffile as tiff
import os

def capture_noncv2(device="/dev/video0", output_file="noncv2.tiff", width=1920, height=1080):
    raw_path = f"/tmp/v4l2_raw_{os.getpid()}.yuyv"
    print(f"[1/3] Streaming 1 uncompressed YUYV frame from {device} via v4l2-ctl (no cv2)...")
    
    cmd = [
        "v4l2-ctl",
        "-d", device,
        f"--set-fmt-video=width={width},height={height},pixelformat=YUYV",
        "--stream-mmap",
        "--stream-count=1",
        f"--stream-to={raw_path}"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error: v4l2-ctl failed (code {res.returncode}):\n{res.stderr.strip()}")
        if "busy" in res.stderr.lower():
            print("Note: The camera device is currently held open by another process (e.g. backend service).")
            print("To free it temporarily: curl -X POST http://localhost:8000/rig -H 'Content-Type: application/json' -d '{\"camera_mode\":\"mock\"}'")
        sys.exit(1)

    print(f"[2/3] Decoding raw YUYV bytes into RGB matrix using pure NumPy...")
    raw = np.fromfile(raw_path, dtype=np.uint8)
    os.remove(raw_path)

    expected_bytes = width * height * 2
    if len(raw) != expected_bytes:
        print(f"Error: Incomplete frame (got {len(raw)} bytes, expected {expected_bytes})")
        sys.exit(1)

    yuyv = raw.reshape((height, width, 2))
    y = yuyv[:, :, 0]
    u = yuyv[:, 0::2, 1].repeat(2, axis=1)
    v = yuyv[:, 1::2, 1].repeat(2, axis=1)

    # Convert BT.601 YUV -> RGB in pure float32 NumPy
    y_flt = y.astype(np.float32)
    u_flt = u.astype(np.float32) - 128.0
    v_flt = v.astype(np.float32) - 128.0

    r = np.clip(y_flt + 1.402 * v_flt, 0, 255).astype(np.uint8)
    g = np.clip(y_flt - 0.344136 * u_flt - 0.714136 * v_flt, 0, 255).astype(np.uint8)
    b = np.clip(y_flt + 1.772 * u_flt, 0, 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)

    # Scale to 16-bit
    rgb16 = (rgb.astype(np.uint16) * 257)

    print(f"[3/3] Writing {output_file} using tifffile...")
    tiff.imwrite(output_file, rgb16)

    unique_vals = len(np.unique(rgb16))
    print(f"Successfully saved {output_file}!")
    print(f"Statistics: shape={rgb16.shape}, min={rgb16.min()}, max={rgb16.max()}, mean={rgb16.mean():.1f}, std={rgb16.std():.2f}")
    print(f"Histogram analysis: {unique_vals} unique value(s) in frame.")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "noncv2.tiff"
    dev = sys.argv[2] if len(sys.argv) > 2 else "/dev/video0"
    capture_noncv2(device=dev, output_file=out)
