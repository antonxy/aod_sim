let
    pkgs = import <nixpkgs> {};
    stdenv = pkgs.stdenv;
    pypkgs = pkgs.python39Packages;
in rec {
    simShell = pkgs.mkShell rec {
        buildInputs = [
            pkgs.python3
            pypkgs.nidaqmx
            pypkgs.numpy
            pypkgs.matplotlib
            pypkgs.tifffile
            pypkgs.ipython
            pypkgs.scipy
            pypkgs.pyside2
            pypkgs.scikitimage
        ];

        #LD_LIBRARY_PATH = "${pypkgs.pyside.dev}/lib";
        QT_QPA_PLATFORM_PLUGIN_PATH = "${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins/platforms";

    };
}
