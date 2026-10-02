# TO BE COPIED INTO EACH README

## 4. Results and Evaluation

### 4.0 Test Cases (Self-made)

1. Test Case 1: High-Frequency Detail (Binary Threshold)
    - Image: snake.jpg
    - n value: 1
    - Rationale: This test evaluates how well the algorithm identifies fine, complex edges. The scales of the snake provide high-frequency details. Using n=1 (binary threshold) will create a stark, high-contrast output, making it easy to see if the main patterns of the scales are correctly detected. It's a good test for correctness.

2. Test Case 2: Smooth Gradients and Broad Edges (Gradient Magnitude)
    - Image: lion.jpg
    - n value: 0
    - Rationale: The lion's mane and the out-of-focus background have smooth transitions and less defined edges. Using n=0 (gradient magnitude) is perfect for this scenario. It will show the intensity of the edges, allowing us to see how the filter responds to both the sharp edges of the lion's face and the softer edges of its fur.

3. Test Case 3: Structural and Geometric Lines (Multi-level Threshold)
    - Image: view.jpg
    - n value: 4
    - Rationale: This image is dominated by strong, straight lines from the fence and the clear horizon. Using a multi-level threshold (n=4) will test the algorithm's ability to quantize edge strengths into different levels. We expect the strong lines of the fence to be in the highest intensity buckets, while the softer edges of the hills and clouds will fall into lower-intensity gray levels.

4. Test Case 4: Repetitive Patterns and Clutter (Binary Threshold)
    - Image: fish.jpg
    - n value: 1
    - Rationale: This image contains many similar, overlapping objects, creating a cluttered scene. A binary threshold (n=1) will test the filter's ability to separate individual objects. The goal is to see if the outlines of each fish can be distinguished, or if they blend into a single noisy mass. This is a good stress test for edge separation.

5. Test Case 5: Vibrant Colors and Sharp Boundaries (High Multi-level Threshold)
    - Image: birds.jpg
    - n value: 128
    - Rationale: The birds have very distinct, sharp boundaries between different colored feathers. The grayscale conversion will turn these color boundaries into sharp intensity changes. Using a high number of levels (n=128) will test the multi-level thresholding logic more thoroughly, creating a more nuanced "posterized" effect on the detected edges. It checks if the thresholding logic scales correctly with a higher number of bins.

### 4.1 Correctness

| Input Image | Serial Output | Parallel Output |
|-------------|---------------|-----------------|
| ![input](../test_cases/snake.jpg) | ![serial](../serial/output/snake_binary.jpg) | ![parallel](path/to/parallel_output.jpg) |
| ![input](../test_cases/lion.jpg) | ![serial](../serial/output/lion_gradient.jpg) | ![parallel](path/to/parallel_output.jpg) |
| ![input](../test_cases/view.jpg) | ![serial](../serial/output/view_multi.jpg) | ![parallel](path/to/parallel_output.jpg) |
| ![input](../test_cases/fish.jpg) | ![serial](../serial/output/fish_binary.jpg) | ![parallel](path/to/parallel_output.jpg) |
| ![input](../test_cases/birds.jpg) | ![serial](../serial/output/birds_multi128.jpg) | ![parallel](path/to/parallel_output.jpg) |



### 4.2 Performance Comparison

#### Serial Version
| Image Name | Input Time (ms) | Processing Time (ms) | Output Time (ms) | Total Time (ms) |
|------------|-----------------|-----------------------|------------------|-----------------|
| snake.jpg | 8                | 52                      |  9                | 69                |
| lion.jpg | 6                | 48                      |  9                | 63                |
| view.jpg | 19                | 215                      | 17                 | 251                |
| fish.jpg | 86                | 452                      | 80                 | 618                |
| birds.jpg | 1                | 41                     |  0                | 42                |

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


## TEST CASE DEMO

fish dan view

SERIAL
./serial/serial 0 test_cases/fish.jpg ./test_cases/serial/fish.jpg > ./test_cases/serial/fish.txt && ./serial/serial 128 test_cases/view.jpg ./test_cases/serial/view.jpg > ./test_cases/serial/view.txt

MPI
mpirun --hostfile hostfile -np 2 --mca btl_tcp_if_include eth1 ./src/mpi 0 test_cases/fish.jpg tc_results/fish2.jpg > tc_results/fish2.txt
mpirun --hostfile hostfile -np 3 --mca btl_tcp_if_include eth1 ./src/mpi 0 test_cases/fish.jpg tc_results/fish3.jpg > tc_results/fish3.txt
mpirun --hostfile hostfile -np 4 --mca btl_tcp_if_include eth1 ./src/mpi 0 test_cases/fish.jpg tc_results/fish4.jpg > tc_results/fish4.txt

mpirun --hostfile hostfile -np 2 --mca btl_tcp_if_include eth1 ./src/mpi 128 test_cases/view.jpg tc_results/view2.jpg > tc_results/view2.txt
mpirun --hostfile hostfile -np 3 --mca btl_tcp_if_include eth1 ./src/mpi 128 test_cases/view.jpg tc_results/view3.jpg > tc_results/view3.txt
mpirun --hostfile hostfile -np 4 --mca btl_tcp_if_include eth1 ./src/mpi 128  test_cases/view.jpg tc_results/view4.jpg > tc_results/view4.txt

OPEN MP
./open_mp/open_mp 4 0 test_cases/fish.jpg test_cases/open_mp/fish4.jpg > test_cases/open_mp/fish4.txt
./open_mp/open_mp 8 0 test_cases/fish.jpg test_cases/open_mp/fish8.jpg > test_cases/open_mp/fish8.txt
./open_mp/open_mp 16 0 test_cases/fish.jpg test_cases/open_mp/fish16.jpg > test_cases/open_mp/fish16.txt

./open_mp/open_mp 4 128 test_cases/view.jpg test_cases/open_mp/view4.jpg > test_cases/open_mp/view4.txt
./open_mp/open_mp 8 128 test_cases/view.jpg test_cases/open_mp/view8.jpg > test_cases/open_mp/view8.txt
./open_mp/open_mp 16 128 test_cases/view.jpg test_cases/open_mp/view16.jpg > test_cases/open_mp/view16.txt

AVX2
./avx2/avx2 0 test_cases/fish.jpg ./test_cases/avx2/fish.jpg > ./test_cases/avx2/fish.txt
./avx2/avx2 128 test_cases/view.jpg ./test_cases/avx2/view.jpg > ./test_cases/avx2/view.txt

CUDA
./cuda/cuda 0 test_cases/fish.jpg ./test_cases/cuda/fish.jpg > ./test_cases/cuda/fish.txt
./cuda/cuda 128 test_cases/view.jpg ./test_cases/cuda/view.jpg > ./test_cases/cuda/view.txt