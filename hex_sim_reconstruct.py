from hexSimProcessor import HexSimProcessor
import tifffile
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    print(f"Reconstruct file {args.filename}")
    frames = tifffile.imread(args.filename)
    assert frames.shape[0] == 7

    reconstruct(frames[:, :64, :64])

if __name__ == "__main__":
    main()
