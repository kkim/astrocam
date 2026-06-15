import numpy as np
import cv2
import sys
import os
import time
import glob
from mock_rig import MockAstroRig
from panorama import PanoramaManager

def test_mock_panorama():
    print("Testing PanoramaManager with Mock Rig and Drift...")
    
    # 1. Initialize Mock Rig
    rig = MockAstroRig()
    
    # Let it generate a few frames first
    time.sleep(0.5)
    
    # Change duty cycle to 85.5% to simulate a slight drift rate of 10 pixels/sec
    rig.set_motor_speed(85.5)
    
    # 2. Instantiate PanoramaManager
    manager = PanoramaManager(rig)
    
    # Keep track of captures directory before starting
    captures_dir = "/home/kio/projects/astrocam/captures"
    if not os.path.exists(captures_dir):
        os.makedirs(captures_dir)
    before_files = set(glob.glob(os.path.join(captures_dir, "panorama_*.jpg")))
    
    # 3. Start Panorama (5 frames, auto_align=True)
    frames_to_run = 5
    success = manager.start(frames=frames_to_run, drift_step=10.0, auto_align=True)
    assert success, "Failed to start panorama"
    
    # 4. Poll and monitor progress
    print("Polling panorama status...")
    timeout = 20.0 # max 20 seconds
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < timeout:
        status = manager.get_status()
        last_status = status
        print(f"Status: Active={status['active']}, Frame={status['current']}/{status['total']}, "
              f"Offsets: dx={status['offset_x']}, dy={status['offset_y']}, rot={status['offset_angle']}°")
        
        if not status['active'] and status['current'] == status['total']:
            break
        time.sleep(0.5)
        
    status = manager.get_status()
    assert status['current'] == frames_to_run, f"Expected {frames_to_run} frames, but only got {status['current']}"
    assert not status['active'], "Panorama is still active after completion"
    
    # Verify that a new panorama JPEG was saved to captures directory
    after_files = set(glob.glob(os.path.join(captures_dir, "panorama_*.jpg")))
    new_files = after_files - before_files
    print(f"Created new files: {new_files}")
    assert len(new_files) > 0, "No new panorama JPEG was generated!"
    
    # Clean up the test output file
    for file_path in new_files:
        print(f"Verifying and cleaning up {file_path}...")
        img = cv2.imread(file_path)
        assert img is not None, f"Generated file {file_path} is not a valid image"
        assert img.shape[0] > 0 and img.shape[1] > 0, "Generated image is empty"
        os.remove(file_path)
        
    rig.close()
    print("✓ Mock Panorama test passed successfully!")

if __name__ == "__main__":
    try:
        test_mock_panorama()
        print("\nALL PANORAMA TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
