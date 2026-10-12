#!/usr/bin/env python3
"""
Batch Mosaic Generator
======================
Runs the star-aligned mosaic stitcher across multiple capture session directories,
generating an independent full-FOV, HDR-stretched mosaic for each session.
"""

import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from mosaic_stars import combine_images

TARGET_SESSIONS = [
    "/home/kio/projects/astrocam/captures/session_20261006_152650",
    "/home/kio/projects/astrocam/captures/session_20261006_205057",
    "/home/kio/projects/astrocam/captures/session_20261006_205523",
    "/home/kio/projects/astrocam/captures/session_20261006_205725",
    "/home/kio/projects/astrocam/captures/session_20261006_214855",
    "/home/kio/projects/astrocam/captures/session_20261006_224132",
]


def run_batch():
    results = []
    total_start = time.time()

    print("==================================================================")
    print(" AstroCam Batch Mosaic Processor")
    print(f" Processing {len(TARGET_SESSIONS)} capture sessions independently")
    print("==================================================================")

    for idx, session_path in enumerate(TARGET_SESSIONS, 1):
        s_name = os.path.basename(session_path)
        print(f"\n[{idx}/{len(TARGET_SESSIONS)}] Processing: {s_name}...")
        out_dir = os.path.join(session_path, "mosaic")

        try:
            res = combine_images(
                input_dir=session_path,
                output_dir=out_dir,
                fov_mode="full",
                feather_px=12,
                stretch_method="asinh",
                stretch_factor=50.0,
                black_percentile=0.5,
                save_individual_renders=True,
                verbose=True,
            )
            results.append({
                "session": s_name,
                "status": "SUCCESS",
                "canvas": f"{res['canvas_size'][0]}x{res['canvas_size'][1]}",
                "frames": f"{res['aligned_frames']}/{res['valid_files']}",
                "elapsed": f"{res['elapsed_sec']:.1f}s",
                "output_dir": out_dir,
            })
        except Exception as e:
            print(f"[!] Error processing {s_name}: {e}")
            results.append({
                "session": s_name,
                "status": f"ERROR: {e}",
                "canvas": "N/A",
                "frames": "N/A",
                "elapsed": "N/A",
                "output_dir": out_dir,
            })

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print(" BATCH MOSAIC PROCESSING SUMMARY")
    print("=" * 80)
    print(f"{'Session':<26} {'Status':<10} {'Canvas':<14} {'Frames':<12} {'Time':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['session']:<26} {r['status']:<10} {r['canvas']:<14} {r['frames']:<12} {r['elapsed']:<8}")
    print("=" * 80)
    print(f"Total time elapsed: {total_elapsed / 60.0:.2f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    run_batch()
