# INF01008
Trabalho de Programação Paralela



##  How to Build

This project uses a parameterized CMake configuration. You must specify which implementation folder you want to compile using the `-DIMPL` flag.

### 1. Configure the Build
From the project root, run the following command to generate the build files for a specific version (e.g., your parallel version):

```bash
# General syntax: cmake -S . -B build -DIMPL=src/<folder_name>
cmake -S . -B build -DIMPL=src/impl_parallel_nan
```
### 2. Compile
```bash
cmake --build build
```

### 3. Release Mode
```bash
cmake -S . -B build -DIMPL=src/impl_parallel_nan -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
