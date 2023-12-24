{ pkgs }: {
  deps = [
    pkgs.tomcat10
    pkgs.sqlite.bin
    pkgs.zlib
    pkgs.tk
    pkgs.tcl
    pkgs.openjpeg
    pkgs.libxcrypt
    pkgs.libwebp
    pkgs.libtiff
    pkgs.libjpeg
    pkgs.libimagequant
    pkgs.lcms2
    pkgs.freetype
    pkgs.python38Full
  ];
  env = {
    PYTHONBIN = "${pkgs.python38Full}/bin/python3.8";
    
  PYTHON_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
        pkgs.zlib
        pkgs.tk
        pkgs.tcl
        pkgs.openjpeg
        pkgs.libxcrypt
        pkgs.libwebp
        pkgs.libimagequant
      pkgs.freetype
    ];LANG = "en_US.UTF-8";
  };
}