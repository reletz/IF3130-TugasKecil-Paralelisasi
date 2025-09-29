# Parallelization Report — Sobel Edge Detection with OpenMPI

## Team Information
- **Team ID:**  
- **Class:**  

### Members
| Name      | Student ID |
|-----------|------------|
| Name 1    | ID1        |
| Name 2    | ID2        |
| Name 3    | ID3        |
| ...       | ...        |



## List of Contents
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
Kita pakai NixOS karena males copas config satu-satu ke tiap VM-nya.

Jadi caranya gini:
- Install dulu Nix Package Manager
```
sudo pacman -Sy nix
```

atau

```
sudo apt-get install nix
```

Jangan lupa masukin ke Path

- Udah sih good to go harusnya tinggal jalanin make-make nya.

## 1. Introduction
Provide a short description of the purpose of this assignment, the Sobel edge detection algorithm in the given code, and the goal of applying OpenMPI parallelization.



## 2. Theory: Parallelizable Operations
Explain which operations or functions in the program can be parallelized and why.  
Examples:
- Pixel-by-pixel convolution in the Sobel operator is independent and can be distributed across processes.  
- Gradient magnitude and thresholding can also be computed in parallel since each pixel is independent.  
- Image I/O is not parallelized as it does not benefit significantly from MPI.  


## 3. Code Changes and Implementation
### 3.1 Parallelization Strategy
Describe how the workload is divided among MPI processes (e.g., row-wise distribution, block distribution). Mention how many processes you tested.

### 3.2 Code Modifications
Document the changes you made to the code. Use **before vs after** snippets and provide explanations.

Example:

```cpp
// Serial version loop
for (int y = 1; y < in.h - 1; y++) {
    for (int x = 1; x < in.w - 1; x++) {
        ...
    }
}
```
With this
``` cpp
// Parallel version with MPI (row distribution example)
int rowsPerProcess = in.h / world_size;
int startRow = rank * rowsPerProcess;
int endRow   = (rank == world_size - 1) ? in.h - 1 : (startRow + rowsPerProcess);

for (int y = startRow; y < endRow; y++) {
    for (int x = 1; x < in.w - 1; x++) {
        ...
    }
}
```
You may also include screenshots of modified sections and highlight the differences.


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