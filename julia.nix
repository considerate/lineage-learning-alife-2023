{ pkgs, lib, stdenv, ... }:
let
  d = version: "v${lib.concatStringsSep "." (lib.take 2 (lib.splitString "." version))}";
  extraLibs = [
    pkgs.glib.dev
    pkgs.xorg.libX11
    pkgs.xorg.libXcursor
    pkgs.xorg.libXrandr
    pkgs.xorg.libXt
    pkgs.xorg.libXrender
    pkgs.xorg.libXext
    pkgs.xorg.libXinerama
    pkgs.glfw
    pkgs.mesa

    # pkgs.cairo.dev
    pkgs.dbus.dev
    pkgs.iana-etc
    pkgs.dbus.lib

    # from davinci
    pkgs.udev
    pkgs.librsvg
    pkgs.libGLU
    pkgs.libGL
    # pkgs.xorg.libICE
    # pkgs.xorg.libSM
    # pkgs.xorg.libXxf86vm
    pkgs.alsa-lib
    pkgs.expat
    pkgs.zlib
    pkgs.libuuid
    pkgs.bzip2
    pkgs.libtool
    pkgs.ocl-icd
    # dev environment
    pkgs.cudaPackages.cudatoolkit
    pkgs.cudaPackages.cudnn
    (pkgs.runCommand "openblas64_" { } ''
      mkdir -p $out/lib/
      ln -s ${pkgs.openblasCompat}/lib/libopenblas.so $out/lib/libopenblas64_.so.0
    '')
  ];
in
stdenv.mkDerivation rec {
  name = "julia-env";
  version = pkgs.julia.version;
  nativeBuildInputs = [ pkgs.makeWrapper ];
  phases = [ "installPhase" ];
  installPhase = ''
    export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
    export LD_LIBRARY_PATH=${lib.makeLibraryPath extraLibs}
    makeWrapper ${pkgs.julia}/bin/julia $out/bin/julia \
        --prefix LD_LIBRARY_PATH : "$LD_LIBRARY_PATH" \
        --prefix LD_LIBRARY_PATH ":" "${pkgs.linuxPackages.nvidia_x11}/lib" \
        --prefix LD_LIBRARY_PATH ":" "/run/opengl-driver/lib" \
        --set CUDA_PATH "${pkgs.cudaPackages.cudatoolkit}" \
        --set JULIA_PKGDIR $JULIA_PKGDIR
  '';
}
