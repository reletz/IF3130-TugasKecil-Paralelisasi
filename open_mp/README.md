# Parallelization Report — Sobel Edge Detection with OpenMP

## Team Information
- **Team ID: pacuanCUDA**  
- **Class: K-03**  

### Members
| Name      | Student ID |
|---|---|
| Frederiko Eldad Mugiyono | 13523147 |
| Naufarrel Zhafif Abhista | 13523149 |
| I Made Wiweka Putera     | 13523160 |

## List of Contents
0. [Prerequisites](#0-prerequisites)
1. [Introduction](#1-introduction)  
2. [Theory: Parallelizable Operations](#2-theory-parallelizable-operations)  
3. [Code Changes and Implementation](#3-code-changes-and-implementation)  
4. [Results and Evaluation](#4-results-and-evaluation)  
   - [Correctness](#41-correctness)  
   - [Performance Comparison](#42-performance-comparison)  
   - [Speedup and Efficiency](#43-speedup-and-efficiency)  
5. [Discussion](#5-discussion)  
6. [Conclusion](#6-conclusion)  
7. [Additional Notes (Optional)](#7-additional-notes-optional)  
8. [References](#8-references)  
9. [How to Run](#9-how-to-run)  

## 0. Prerequisites

We use OpenMP as a parallelization library for this project

Setup steps:
1. Install OpenMP. On Arch Linux:
```bash
sudo pacman -S openmp
```
Or on Debian/Ubuntu:
```bash
sudo apt-get install openmp
```

Compile command:
```bash
g++ open_mp.cpp -o open_mp -fopenmp -I$OPENCV_INCLUDE/opencv4 -L$OPENCV_LIB -lopencv_core -lopencv_imgproc -lo
pencv_highgui -lopencv_imgcodecs
```

Run command:
```bash
./open_mp <thread_number> <n> input.jpg output.jpg > output.txt
```

## 1. Introduction
This assignment aims to accelerate the Sobel edge detection algorithm using OpenMP. The provided serial code, serial.cpp, will be converted into a parallel version, open_mp.cpp, which uses shared‑memory multithreading to distribute the Sobel filter work across CPU cores and reduce overall processing time.

## 2. Theory: Parallelizable Operations
The Sobel algorithm essentially applies a 3×3 convolution kernel to every pixel of the image. The calculation of the gradient value for a single pixel depends only on itself and its 8 neighboring pixels. This makes the Sobel operation an ideal candidate for data parallelism because:

- **Pixel Independence**: The computation for each pixel (away from the borders) is independent of the others. This allows multiple threads to process different pixels simultaneously without requiring synchronization or locks.
- **Loop-Level Parallelism**: The nested loops iterating over image rows and columns can be parallelized using OpenMP directives. OpenMP's `#pragma omp parallel for` with `collapse(2)` enables efficient distribution of both outer (y) and inner (x) loop iterations across available CPU threads.
- **Shared Memory Access**: Since OpenMP operates on shared memory, all threads can directly access the input image data without communication overhead, unlike distributed memory systems.

**I/O operations** (reading and saving the image) are kept serial and handled by the main thread. Parallelizing I/O would add complexity and is generally only effective for very large datasets. 

## 3. Code Changes and Implementation
## 3. Code Changes and Implementation

### 3.1 Parallelization Strategy
The strategy employed is **loop-level parallelization** using OpenMP, where the nested loops iterating over image pixels are distributed across multiple threads on a shared-memory system.

- **Task Division**: The workload is divided automatically by OpenMP among available threads. Each thread processes a subset of pixels (rows and columns) independently. The number of threads can be controlled via the command-line argument or environment variable `OMP_NUM_THREADS`.

- **Shared Memory Access**: Unlike distributed memory systems (e.g., MPI), all threads in OpenMP share the same memory space. This means the input image data is accessible to all threads without explicit communication. Each thread reads from the shared input image and writes to its assigned portion of the output image without conflicts, as each pixel's output location is determined by its (x, y) coordinates.

- **Boundary Handling**: The Sobel kernel requires a 3×3 neighborhood, so pixels at the image borders (first and last rows/columns) are skipped in the parallel loop to avoid out-of-bounds access. This is handled by starting the loop at `y=1` and `x=1`, and ending before `in.h-1` and `in.w-1`.

- **No Explicit Synchronization Needed**: Since each thread writes to a distinct region of the output image (determined by pixel coordinates), there are no race conditions or data dependencies between threads during the Sobel computation. The OpenMP runtime handles thread creation, workload distribution, and implicit synchronization at the end of the parallel region.

The execution flow is as follows:
1. **Thread Initialization**: OpenMP runtime creates a team of threads based on the specified thread count (via `omp_set_num_threads()` or `OMP_NUM_THREADS` environment variable).
2. **Work Distribution**: The OpenMP runtime automatically divides the nested loop iterations (rows and columns) into chunks and assigns them to available threads. With `schedule(static)` and `collapse(2)`, the iteration space is distributed evenly across threads for balanced workload.
3. **Parallel Computation**: Each thread independently processes its assigned subset of pixels. Threads read from the shared input image and write to distinct locations in the output image without conflicts. No inter-thread communication or synchronization is needed during computation.
4. **Implicit Synchronization**: An implicit barrier at the end of the `#pragma omp parallel for` region ensures all threads complete their work before proceeding. This guarantees the output image is fully processed before the main thread continues.

We tested the implementation with thread counts of 2, 4, 8, and 16 to evaluate scalability and speedup.

### 3.2 Code Modifications

The primary change involves adding OpenMP directives to parallelize the nested loops in the `sobel()` function. The serial version processes pixels sequentially, while the parallel version distributes the workload across multiple threads.

#### Before: serial.cpp

The serial version has a straightforward nested loop structure that processes each pixel one at a time in a single thread.

```cpp
// From: serial.cpp

Image sobel(const Image &in, int mode, const std::vector<int>& thresholds) {
    int Gx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    int Gy[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};
    Image out=in;

    // Serial nested loop - processes pixels one by one
    for(int y=1; y<in.h-1; y++){
        for(int x=1; x<in.w-1; x++){
            int gx=0,gy=0;
            for(int ky=-1;ky<=1;ky++){
                for(int kx=-1;kx<=1;kx++){
                    int val=in.at(x+kx,y+ky);
                    gx+=Gx[ky+1][kx+1]*val;
                    gy+=Gy[ky+1][kx+1]*val;
                }
            }
            int mag=(int)std::sqrt(gx*gx+gy*gy);
            // ... thresholding logic ...
            out.at(x,y)=final_val;
        }
    }
    return out;
}
```

#### After: open_mp.cpp

The parallel version adds an OpenMP directive to distribute the nested loop iterations across multiple threads. Key changes include:

##### 1. OpenMP Parallel Directive

```cpp
// From: open_mp.cpp

Image sobel(const Image &in, int mode, const std::vector<int>& thresholds) {
    int Gx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    int Gy[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};
    Image out=in;

    // Pre-compute threshold levels outside the parallel region
    // (moved from inside loop to avoid redundant computation)
    std::vector<int> levels;
    int thresh_size = thresholds.size();
    if (mode == 2){
        for(int i=0; i<thresh_size; i++){
            levels.push_back((255/(thresh_size+1))*(i+1));
        }
    }

    // Parallel nested loop with OpenMP
    #pragma omp parallel for schedule(static) collapse(2)
    for(int y=1; y<in.h-1; y++){
        for(int x=1; x<in.w-1; x++){
            int gx=0,gy=0;
            for(int ky=-1;ky<=1;ky++){
                for(int kx=-1;kx<=1;kx++){
                    int val=in.at(x+kx,y+ky);
                    gx+=Gx[ky+1][kx+1]*val;
                    gy+=Gy[ky+1][kx+1]*val;
                }
            }
            int mag=(int)std::sqrt(gx*gx+gy*gy);
            // ... thresholding logic ...
            out.at(x,y)=final_val;
        }
    }
    return out;
}
```

**Explanation of OpenMP directive components:**
- `#pragma omp parallel for`: Creates a team of threads and distributes loop iterations among them.
- `schedule(static)`: Divides iterations into equal-sized chunks and assigns them to threads in a round-robin fashion. This provides good load balance when all iterations have similar computational cost.
- `collapse(2)`: Merges the two nested loops (y and x) into a single iteration space, allowing OpenMP to distribute work more effectively across threads. This is especially beneficial when the outer loop has fewer iterations than the number of threads.

##### 2. Thread Count Control

```cpp
// From: open_mp.cpp

int main(int argc,char*argv[]){
    // ... argument parsing ...
    
    int num_threads = std::stoi(argv[1]); // Get thread count from command line
    omp_set_num_threads(num_threads);     // Set the number of OpenMP threads
    
    // ... rest of the program ...
}
```

##### 3. Optimization: Pre-computing Threshold Levels

In the serial version, threshold levels for multi-level thresholding (mode 2) were computed inside the nested loop, causing redundant calculations. In the parallel version, these levels are pre-computed once before entering the parallel region:

```cpp
// Moved outside the parallel region for efficiency
std::vector<int> levels;
int thresh_size = thresholds.size();
if (mode == 2){
    for(int i=0; i<thresh_size; i++){
        levels.push_back((255/(thresh_size+1))*(i+1));
    }
}
```

This optimization reduces unnecessary computation and ensures all threads can read from the same shared `levels` vector without synchronization overhead.

##### 4. No Explicit Synchronization Required

The implicit barrier at the end of the `#pragma omp parallel for` region ensures all threads complete their work before the main thread proceeds to save the output image. This makes the code simpler and easier to maintain compared to distributed memory parallelization.


## 4. Results and Evaluation

### 4.1 Correctness
- Did the parallel version produce the same output image as the serial version?  
- Show example input and output images for both versions.  

Example (replace with actual images):

| Input Image | Serial Output | Parallel Output |
|-------------|---------------|-----------------|
| ![input](path/to/input.jpg) | ![serial](path/to/serial_output.jpg) | ![parallel](path/to/parallel_output.jpg) |



### 4.2 Performance Comparison

#### Serial Version
| Image Name | Input Time (ms) | Processing Time (ms) | Output Time (ms) | Total Time (ms) |
|------------|-----------------|-----------------------|------------------|-----------------|
| image1.jpg |                 |                       |                  |                 |
| image2.jpg |                 |                       |                  |                 |
| image3.jpg |                 |                       |                  |                 |

#### Parallel Version
| Image Name | Core Number | Input Time (ms) | Processing Time (ms) | Output Time (ms) | Total Time (ms) |
|------------|-------------|-----------------|-----------------------|------------------|-----------------|
| image1.jpg | 2           |                 |                       |                  |                 |
| image1.jpg | 4           |                 |                       |                  |                 |
| image1.jpg | 8           |                 |                       |                  |                 |
| image2.jpg | 2           |                 |                       |                  |                 |
| image2.jpg | 4           |                 |                       |                  |                 |
| image2.jpg | 8           |                 |                       |                  |                 |



### 4.3 Speedup and Efficiency
- **Speedup** = Serial Time / Parallel Time  
- **Efficiency** = Speedup / Number of Processes  



## 5. Discussion
- What worked well in your parallelization approach?  
- What challenges did you face (data distribution, communication, synchronization)?  
- Did you notice any overhead, and how did it affect performance?  



## 6. Conclusion
Summarize your findings:  
- Was parallelization effective?  
- Did it improve performance significantly?  
- Any tradeoffs between computation speed and communication overhead?  



## 7. Additional Notes (Optional)
- Suggestions for further improvement (e.g., using hybrid MPI + OpenMP).  
- Observations for different input sizes (small vs large images).  
- Any optimizations beyond the basic parallelization.  



## 8. References
List any references you used (books, lecture notes, research papers, or online sources).



## 9. How to Run
How to run the programs (compiling and running process must atleast)