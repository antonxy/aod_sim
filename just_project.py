import nidaq_pattern
import hex_grid
import matplotlib.pyplot as plt
import time
import sys

pattern_deg = hex_grid.projection_hex_pattern_deg(40, 20, 20, orientation_rad = 0.1)
pattern_rate_Hz = 40000.0
exposure_time_sec = pattern_deg.shape[1] / pattern_rate_Hz
print(f"Exposure time {exposure_time_sec*1e3} ms")
sys.stdout.flush()


nidaq_pattern.project_patterns(pattern_deg[0, :, :], pattern_rate_Hz, loop=True)