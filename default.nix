let
    pkgs = import <nixpkgs> {};
    stdenv = pkgs.stdenv;
    pypkgs = pkgs.python39Packages;
in rec {
    noise = pkgs.mkShell rec {
        buildInputs = [
            pkgs.python3
            pypkgs.nidaqmx
            pypkgs.numpy
            pypkgs.matplotlib
            pypkgs.tifffile
            pypkgs.ipython
        ];
    };
}
