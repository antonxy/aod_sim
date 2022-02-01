import nidaq_pattern
import hex_grid
import matplotlib.pyplot as plt
import time
import sys
import tifffile
import numpy as np

import pco


with pco.Camera(debuglevel='error', interface="Camera Link Silicon Software") as cam:

    cam.default_configuration()

    for dist in [40, 45, 50, 55, 60]:
        pattern_deg = hex_grid.projection_hex_pattern_deg(dist, 20, 10, orientation_rad = 0.0)
        pattern_rate_Hz = 40000.0
        exposure_time_sec = pattern_deg.shape[1] / pattern_rate_Hz
        print(f"Exposure time {exposure_time_sec*1e3} ms")
        sys.stdout.flush()

        cam.configuration = {
            'exposure time': exposure_time_sec,
            'roi': (1, 1, 1008, 1008),
            #'trigger': 'auto sequence',
            'trigger': 'external exposure start & software trigger',
            'acquire': 'auto',
        }

        cam.record(number_of_images=7, mode='sequence non blocking')

        nidaq_pattern.project_patterns(pattern_deg, pattern_rate_Hz)
        while True:
            running = cam.rec.get_status()['is running']
            if not running:
                break
            time.sleep(0.001)

        images, metadatas = cam.images()
        images = np.stack(images)

        tifffile.imwrite(f"dist_{dist}.tiff", images)

#        plt.imshow(images[0])
#        plt.colorbar()
#        plt.show()
