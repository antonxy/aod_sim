from hexSimProcessor import HexSimProcessor
import tifffile
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-N', type=int)
    parser.add_argument('--offset', type=int, nargs='+')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--filter', action='store_true')
    args = parser.parse_args()

    print(f"Reconstruct file {args.filename}")
    frames = tifffile.imread(args.filename)
    assert frames.shape[0] == 7

    N = args.N
    assert len(args.offset) == 2
    frames = frames[:, args.offset[1]:args.offset[1]+N, args.offset[0]:args.offset[0]+N]

    p = HexSimProcessor()
    p.N = N
    p.debug = args.debug
    p.use_filter = args.filter
    p.calibrate(frames)
    reconstruct = p.reconstruct_fftw(frames)

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(reconstruct)
    ax[1].imshow(scipy.ndimage.zoom(np.sum(frames, axis=0), (2, 2), order=1))
    plt.show()



if __name__ == "__main__":
    main()
