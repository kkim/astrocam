#!/usr/bin/env python3
"""
Batch Mosaic Generator
======================
Runs the star-aligned mosaic stitcher across capture session directories,
generating an independent full-FOV, HDR-stretched mosaic in `mosaic_v1` for each session.
"""

import os
import sys
import glob
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(script_dir, "../backend"))
sys.path.insert(0, script_dir)
sys.path.insert(0, backend_dir)

from mosaic_stars import combine_images
from mosaic_service import create_placeholder_mosaic

CAPTURES_DIR = os.path.abspath(os.path.join(script_dir, "../captures"))
MOSAIC_SUBDIR = "mosaic_v1"


def discover_sessions():
    """Finds all session directories (newest first by session timestamp) that previously had a mosaic."""
    all_sessions = sorted(
        glob.glob(os.path.join(CAPTURES_DIR, "session_*")),
        key=lambda p: os.path.basename(p),
        reverse=True,
    )
    # Prioritize sessions that previously had a mosaic built in `mosaic/`, newest first
    prior_mosaics = []
    for s_dir in all_sessions:
        if not os.path.isdir(s_dir):
            continue
        if os.path.exists(os.path.join(s_dir, "mosaic", "mosaic_linear_16bit.tif")):
            prior_mosaics.append(s_dir)
    return prior_mosaics


def run_batch(target_sessions=None):
    if target_sessions is None:
        target_sessions = discover_sessions()

    results = []
    total_start = time.time()

    print("==================================================================")
    print(" AstroCam Batch Mosaic Processor (mosaic_v1)")
    print(f" Processing {len(target_sessions)} capture sessions independently")
    print("==================================================================")

    for idx, session_path in enumerate(target_sessions, 1):
        s_name = os.path.basename(session_path)
        out_dir = os.path.join(session_path, MOSAIC_SUBDIR)
        real_marker = os.path.join(out_dir, "mosaic_linear_16bit.tif")
        proc_marker = os.path.join(out_dir, ".processing")
        progress_file = os.path.join(out_dir, ".progress.json")

        if os.path.exists(real_marker):
            print(f"\n[{idx}/{len(target_sessions)}] Skipping {s_name} (already has {MOSAIC_SUBDIR})")
            results.append({
                "session": s_name,
                "status": "SKIPPED",
                "canvas": "existing",
                "frames": "-",
                "elapsed": "0.0s",
                "output_dir": out_dir,
            })
            continue

        if os.path.exists(proc_marker):
            print(f"\n[{idx}/{len(target_sessions)}] Skipping {s_name} (currently processing)")
            continue

        # Wait if the Web UI is actively processing another session to avoid CPU/RAM contention
        while True:
            active_markers = [
                m for m in glob.glob(os.path.join(CAPTURES_DIR, "session_*", MOSAIC_SUBDIR, ".processing"))
                if os.path.exists(m) and (time.time() - os.path.getmtime(m)) < 900
            ]
            if not active_markers:
                break
            time.sleep(2.0)

        if os.path.exists(real_marker):
            print(f"\n[{idx}/{len(target_sessions)}] Skipping {s_name} (completed while waiting)")
            continue

        print(f"\n[{idx}/{len(target_sessions)}] Processing: {s_name}...")
        os.makedirs(out_dir, exist_ok=True)

        try:
            files = sorted([
                f for f in os.listdir(session_path)
                if not f.startswith(".") and f != "thumb.jpg" and f.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png"))
            ])
            if files:
                create_placeholder_mosaic(session_path, out_dir, files[0])

            with open(proc_marker, "w") as f:
                f.write(str(time.time()))

            res = combine_images(
                input_dir=session_path,
                output_dir=out_dir,
                fov_mode="full",
                feather_px=12,
                stretch_method="asinh",
                stretch_factor=60.0,
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
        finally:
            for p in (proc_marker, progress_file):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

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
