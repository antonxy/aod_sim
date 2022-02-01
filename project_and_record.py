import nidaq_pattern
import hex_grid
import matplotlib.pyplot as plt
import time
import sys
import tifffile
import numpy as np
from pathlib import Path
import pco

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--num_x', type=int, default=20)
    parser.add_argument('--num_y', type=int, default=10)
    parser.add_argument('--orientation_deg', type=float, default=0.0)
    parser.add_argument('--dist_deg', type=float, nargs='+', required=True, help="Reasonable value ~ 0.038")
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    path = Path(args.filename)

    with pco.Camera(debuglevel='error', interface="Camera Link Silicon Software") as cam:

        cam.default_configuration()

        for dist in args.dist_deg:
            pattern_deg = hex_grid.projection_hex_pattern_deg(dist, args.num_x, args.num_y, orientation_rad = np.deg2rad(args.orientation_deg))
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

            dist_str = f"{dist:.4f}".replace('.', '_')
            tifffile.imwrite(f"{Path.joinpath(path.parent, path.stem)}_dist{dist_str}{path.suffix}", images)

            if args.show:
                plt.imshow(images[0])
                plt.colorbar()
                plt.show()

if __name__ == "__main__":
    main()
