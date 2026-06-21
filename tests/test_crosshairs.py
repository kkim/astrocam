import numpy as np
import cv2
import time
from mock_rig import MockAstroRig
from pipeline import AstroPipeline

def test_crosshair_detection_and_drawing():
    # 1. Initialize Mock Rig and Pipeline
    rig = MockAstroRig()
    # Wait for mock frame generation to start
    time.sleep(0.2)
    
    pipeline = AstroPipeline(rig)
    try:
        # Give pipeline time to process at least one frame
        time.sleep(0.5)
        
        # Verify initially auto_tracking is disabled and no reference stars are detected
        assert not pipeline.auto_tracking
        assert len(pipeline.reference_stars) == 0
        
        # Capture frame without crosshairs
        frame_before = pipeline.get_frame()
        assert frame_before is not None
        
        # 2. Enable Auto-tracking (AUTO button hit)
        pipeline.set_auto_tracking(True)
        assert pipeline.auto_tracking
        
        # Wait for the next processing step to populate reference frame and stars
        time.sleep(0.5)
        
        # Verify 5 reference stars are detected
        assert len(pipeline.reference_stars) > 0
        assert len(pipeline.reference_stars) <= 5
        
        # Capture frame with crosshairs
        frame_after = pipeline.get_frame()
        assert frame_after is not None
        
        # Since cyan crosshairs (0, 255, 255) are drawn, the frame bytes should differ
        assert frame_before != frame_after
        
        # 3. Disable Auto-tracking
        pipeline.set_auto_tracking(False)
        assert not pipeline.auto_tracking
        assert len(pipeline.reference_stars) == 0
        
    finally:
        pipeline.close()
        rig.close()
