import nidaq_pattern
import hex_grid
import matplotlib.pyplot as plt

import pco

pattern_deg = hex_grid.projection_hex_pattern_deg(0.1, 10, 5, orientation_rad = 0.1)
pattern_rate_Hz = 10000.0
exposure_time_sec = pattern_deg.shape[1] / pattern_rate_Hz

with pco.Camera(debuglevel='verbose') as cam:

    cam.default_configuration()
    cam.configuration = {
        'exposure time': exposure_time_sec,
        'roi': (1, 1, 1008, 1008),
        'trigger': 'external exposure start & software trigger',
        'acquire': 'auto',
    }

    cam.record(number_of_images=7, mode='sequence non blocking')

    nidaq_pattern.project_patterns(pattern_deg, pattern_rate_Hz)

    images, metadatas = cam.images()

    plt.imshow(images[0])
    plt.show()
