# Parallel Sobel Edge Detection (pacuanCUDA)

A small assignment for **IF3130 – Parallel and Distributed Systems** (ITB, K-03, team *pacuanCUDA*).

## What this project does

The project implements the **Sobel edge detection** algorithm on JPEG images and compares one serial baseline against four parallel implementations:

| Version | Technology | Kind of parallelism |
|---|---|---|
| `serial` | Plain C++ | Baseline, single thread |
| `open_mp` | OpenMP | Shared-memory multithreading |
| `open_mpi` | Open MPI | Distributed-memory, multi-node (run on a VM cluster) |
| `avx2` | AVX2 intrinsics (+FMA) | SIMD, data-level parallelism on one core |
| `cuda` | NVIDIA CUDA | GPU, massively parallel threads |

Every version follows the same pipeline:

1. **Input** – load a JPEG as grayscale with OpenCV.
2. **Processing** – convolve the image with the Sobel `Gx` and `Gy` kernels and compute the gradient magnitude per pixel.
3. **Output** – write the result as a JPEG and print a timing report (input / processing / output time, in ms) to stdout.

The output style is selected by the integer `n`:

| `n` | Mode | Result |
|---|---|---|
| `0` | Gradient magnitude | Grayscale edge-strength image |
| `1` | Binary threshold | Black/white edges (threshold 128) |
| `≥ 2` | Multi-level threshold | Edge strength quantized into `n` levels (thresholds at `255·i/n`) |

Each version's folder has its own report README covering theory, implementation, correctness checks, timing tables, speedup and efficiency.

## Usage

All programs need OpenCV (`OPENCV_INCLUDE` and `OPENCV_LIB` environment variables are expected by most Makefiles). Build with `make` inside each folder.

```sh
./serial/serial   n input.jpg output.jpg > output.txt
./avx2/avx2       n input.jpg output.jpg > output.txt
./cuda/cuda       n input.jpg output.jpg > output.txt
./open_mp/open_mp thread_count n input.jpg output.jpg > output.txt
mpirun -np <procs> --hostfile hostfile ./mpi n input.jpg output.jpg > output.txt
```

## Repository layout

### `serial/`
Single-threaded reference implementation (`serial.cpp`) and its `Makefile`. It is the baseline for correctness and for all speedup/efficiency calculations. `output/` holds its result images and timing `.txt` files. `README.md` is the report and the template that the other reports copy.

### `open_mp/`
OpenMP version (`open_mp.cpp`, `Makefile`). The Sobel loops are parallelized with `#pragma omp parallel for schedule(static) collapse(2)`, and the thread count is the first CLI argument. `output/` has results for several thread counts (e.g. 4 and 8). `README.md` is the report.

### `open_mpi/`
Distributed Open MPI version.
- `src/open_mpi.cpp` – the MPI source.
- `Makefile` – the cluster manager, run on the host. It starts and stops the VMs (`up`, `down`, `clean`, `status`), opens SSH sessions (`ssh-master`, `ssh-worker1..3`), and syncs, builds, distributes and pulls results (`sync`, `remote-build`, `distribute`, `pull`).
- `flake.nix` / `flake.lock` – Nix flake that builds a NixOS VM cluster of one master and three workers, with OpenCV and Open MPI.
- `mpi_cluster_key`, `mpi_cluster_key.pub` – SSH key pair for passwordless access between cluster nodes. It is for this throwaway VM cluster only.
- `tc_results/` – committed output images and timing files for 2 and 3 processes.
- `README.md` – the report.

Generated and untracked files (`result*`, `*.qcow2`, `*.log`, `.pid-*`, `results/`) are VM images, logs, and build or pull artifacts.

### `avx2/`
SIMD version using AVX2 and FMA intrinsics (`src/avx2.cpp`, `Makefile` with `-mavx2 -mfma -march=native`). It uses 32-byte aligned image buffers and processes several pixels per instruction. It targets AVX2 only, not other parallel frameworks. `output/` holds its results, and `README.md` is the report.

### `cuda/`
GPU version (`cuda.cu`, built with `nvcc` through the `Makefile`). A `sobelKernel` computes one output pixel per GPU thread, and host code handles memory transfers and launch. `output/` holds results, and `README.md` is the report.

### `test_cases/`
Shared inputs and demo outputs.
- Top level: the five input images `snake.jpg`, `lion.jpg`, `view.jpg`, `fish.jpg`, `birds.jpg`. They cover high-frequency detail, smooth gradients, straight lines, clutter and sharp colour boundaries, with `n` values chosen to match.
- `serial/`, `avx2/`, `cuda/`, `open_mp/`: demo-run outputs (image and timing `.txt`) for each implementation on `fish` and `view`. `open_mp/` has runs for 4, 8 and 16 threads.

## Team

| Name | Student ID |
|---|---|
| Frederiko Eldad Mugiyono | 13523147 |
| Naufarrel Zhafif Abhista | 13523149 |
| I Made Wiweka Putera | 13523160 |
