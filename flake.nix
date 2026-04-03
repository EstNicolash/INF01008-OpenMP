{
  description = "Development environment for Parallel Linear Regression with OpenMP (INF01008)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Compiler and Build Tools
            gcc
            cmake
            gnumake
            pkg-config

            # Testing Frameworks
            catch2_3

            # Debugging and Profiling
            gdb
            valgrind
            linuxPackages_latest.perf

            #python
            python314
            python314Packages.numpy

            #utils
            unzip
            curl
          ];

          shellHook = ''
            echo "--- Parallel Programming Environment Active ---"
            echo "GCC Version: $(gcc --version | head -n 1)"
            echo "Quick Start: cmake -S . -B build && cmake --build build"

            # Set default OpenMP threads to the number of logical cores
            export OMP_NUM_THREADS=$(nproc)
          '';
        };
      }
    );
}
