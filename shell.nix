let
  pkgs = import <nixpkgs> { };

  pyrtools = pkgs.python39.pkgs.buildPythonPackage rec {
    pname = "pyrtools";
    version = "1.0.0";

    src = pkgs.python39.pkgs.fetchPypi {
      inherit pname version;
      sha256 = "+9Pb9W22posmfMLuUD+5KLiE8gW+6RDgwpAesCw16Lg=";
    };

    doCheck = false;

    # dependencies for yellowbrick
    buildInputs = with pkgs.python39Packages; [
      matplotlib
      pillow
      requests
      scipy
      tqdm
    ];
  };
in pkgs.mkShell {
  name = "LenaIsLove";
  buildInputs = with pkgs; [
    python39
    python39Packages.ipython
    python39Packages.jupyter
    python39Packages.matplotlib
    python39Packages.networkx
    python39Packages.numpy
    python39Packages.opencv4
    python39Packages.pillow
    python39Packages.scikit-learn
    python39Packages.scikitimage
    python39Packages.scipy
    python39Packages.tqdm

    pyrtools
  ];
  shellHook = "jupyter-lab";
}
