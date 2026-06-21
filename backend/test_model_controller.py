import numpy as np
from pipeline import update_duty_cycle

def test_update_duty_cycle_basic():
    # Gain vector: g = [1.0, 0.0]
    g = np.array([1.0, 0.0])
    
    # u_k = 80.0%
    # Observed v_k = [1.5, 0.0] px/s (drifts ahead)
    # The ideal u should be: 80.0 - 1.5 = 78.5%
    history_u = [80.0]
    history_v = [np.array([1.5, 0.0])]
    
    next_u = update_duty_cycle(history_v, history_u, g)
    assert abs(next_u - 78.5) < 1e-6, f"Expected 78.5, got {next_u}"
    print("test_update_duty_cycle_basic passed.")

def test_update_duty_cycle_multi_step():
    g = np.array([2.0, 0.0])
    
    # Step 1: u = 80.0%, v = [1.0, 0.0] -> ideal u = 80.0 - 1.0/2.0 = 79.5%
    # Step 2: u = 79.0%, v = [-0.5, 0.0] -> ideal u = 79.0 - (-0.5/2.0) = 79.25%
    # Mean of ideals = (79.5 + 79.25) / 2 = 79.375%
    history_u = [80.0, 79.0]
    history_v = [np.array([1.0, 0.0]), np.array([-0.5, 0.0])]
    
    next_u = update_duty_cycle(history_v, history_u, g)
    assert abs(next_u - 79.375) < 1e-6, f"Expected 79.375, got {next_u}"
    print("test_update_duty_cycle_multi_step passed.")

def test_update_duty_cycle_zero_gain():
    g = np.array([0.0, 0.0])
    history_u = [80.0]
    history_v = [np.array([1.5, 0.0])]
    
    next_u = update_duty_cycle(history_v, history_u, g)
    assert next_u == 80.0, f"Expected 80.0, got {next_u}"
    print("test_update_duty_cycle_zero_gain passed.")

if __name__ == "__main__":
    test_update_duty_cycle_basic()
    test_update_duty_cycle_multi_step()
    test_update_duty_cycle_zero_gain()
    print("All unit tests passed successfully!")
