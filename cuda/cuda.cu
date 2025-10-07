#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// ========================== Image Struct ==========================
struct Image {
    int w, h;
    std::vector<unsigned char> p;
    unsigned char& at(int x, int y) { return p[y * w + x]; }
    const unsigned char& at(int x, int y) const { return p[y * w + x]; }
};

// ========================== Image I/O ==========================
Image loadJPG(const std::string &f) {
    cv::Mat mat = cv::imread(f, cv::IMREAD_GRAYSCALE);
    if(mat.empty()) throw std::runtime_error("Failed to load image");
    Image img{mat.cols, mat.rows, std::vector<unsigned char>(mat.cols * mat.rows)};
    for(int y=0; y<mat.rows; y++)
        for(int x=0; x<mat.cols; x++)
            img.at(x,y) = mat.at<uchar>(y,x);
    return img;
}

void saveJPG(const Image &img, const std::string &f) {
    cv::Mat mat(img.h, img.w, CV_8UC1);
    for(int y=0; y<img.h; y++)
        for(int x=0; x<img.w; x++)
            mat.at<uchar>(y,x) = img.at(x,y);
    cv::imwrite(f, mat);
}

// ========================== Sobel Kernel ==========================
__global__ void sobelKernel(
    const unsigned char *in, unsigned char *out,
    int w, int h, int mode,
    const int *thresholds, int thresh_size,
    const int *levels, int level_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= w - 1 || y >= h - 1) return;

    int Gx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    int Gy[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};

    int sx=0, sy=0;
    for(int ky=-1; ky<=1; ky++)
        for(int kx=-1; kx<=1; kx++){
            int px=in[(y+ky)*w+(x+kx)];
            sx+=px*Gx[ky+1][kx+1];
            sy+=px*Gy[ky+1][kx+1];
        }

    int g_squared = sx*sx + sy*sy;

    if (mode == 0) { // Gradient magnitude
        int g = (int)sqrtf((float)g_squared);
        out[y*w+x] = (g > 255) ? 255 : g;
    }
    else if (mode == 1) { // Binary threshold
        int threshold_squared = thresholds[0] * thresholds[0];
        out[y*w+x] = (g_squared > threshold_squared) ? 0 : 255;
    }
    else { // Multi-level thresholds
        int idx = 0;
        while(idx < thresh_size && g_squared > thresholds[idx] * thresholds[idx]) idx++;
        out[y*w+x] = levels[idx];
    }
}

// ========================== Main ==========================
int main(int argc,char*argv[]){
    if(argc != 4){
        std::cerr<<"Usage: ./cuda n input.jpg output.jpg > output.txt\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    std::string inputFile  = argv[2];
    std::string outputFile = argv[3];

    int mode;
    std::vector<int> thresholds;
    std::vector<int> levels;

    if (n == 0) {
        mode = 0; // gradient magnitude
    } else if (n == 1) {
        mode = 1; // binary
        thresholds.push_back(128);
    } else {
        mode = 2; // multi-level
        for (int i = 1; i < n; i++) thresholds.push_back((255 * i) / n);
        int bins = thresholds.size() + 1;
        levels.resize(bins);
        for (int i=0; i<bins; i++) levels[i] = (255 * i) / (bins - 1);
    }

    // ---------------------- Timing: Input ----------------------
    auto t_start_input = std::chrono::high_resolution_clock::now();
    Image img = loadJPG(inputFile);
    auto t_end_input = std::chrono::high_resolution_clock::now();

    // ---------------------- Timing: Host→Device Copy ----------------------
    auto t_start_h2d = std::chrono::high_resolution_clock::now();
    unsigned char *d_in, *d_out;
    size_t imgSize = img.w * img.h * sizeof(unsigned char);
    cudaMalloc(&d_in, imgSize);
    cudaMalloc(&d_out, imgSize);
    cudaMemcpy(d_in, img.p.data(), imgSize, cudaMemcpyHostToDevice);
    auto t_end_h2d = std::chrono::high_resolution_clock::now();

    // ---------------------- Prepare Thresholds ----------------------
    int *d_thresh = nullptr, *d_levels = nullptr;
    if (!thresholds.empty()) {
        cudaMalloc(&d_thresh, thresholds.size() * sizeof(int));
        cudaMemcpy(d_thresh, thresholds.data(), thresholds.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    if (!levels.empty()) {
        cudaMalloc(&d_levels, levels.size() * sizeof(int));
        cudaMemcpy(d_levels, levels.data(), levels.size() * sizeof(int), cudaMemcpyHostToDevice);
    }

    // ---------------------- CUDA Config ----------------------
    dim3 block(16, 16);
    dim3 grid((img.w + block.x - 1) / block.x, (img.h + block.y - 1) / block.y);

    // ---------------------- Timing: Kernel ----------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sobelKernel<<<grid, block>>>(d_in, d_out, img.w, img.h, mode,
                                 d_thresh, thresholds.size(), d_levels, levels.size());
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);

    // ---------------------- Timing: Device→Host Copy ----------------------
    auto t_start_d2h = std::chrono::high_resolution_clock::now();
    Image res = img;
    cudaMemcpy(res.p.data(), d_out, imgSize, cudaMemcpyDeviceToHost);
    auto t_end_d2h = std::chrono::high_resolution_clock::now();

    // ---------------------- Timing: Output ----------------------
    auto t_start_output = std::chrono::high_resolution_clock::now();
    saveJPG(res, outputFile);
    auto t_end_output = std::chrono::high_resolution_clock::now();

    // ---------------------- Summary ----------------------
    auto tInput   = std::chrono::duration<double, std::milli>(t_end_input - t_start_input).count();
    auto tHtoD    = std::chrono::duration<double, std::milli>(t_end_h2d - t_start_h2d).count();
    auto tDtoH    = std::chrono::duration<double, std::milli>(t_end_d2h - t_start_d2h).count();
    auto tOutput  = std::chrono::duration<double, std::milli>(t_end_output - t_start_output).count();


    std::cout << "================ Sobel Edge Detection ================\n";
    std::cout << "Program Type : CUDA\n";
    std::cout << "------------------------------------------------------\n";
    std::cout << "Mode         : " << mode << "\n";
    std::cout << "Grid Size    : " << grid.x << "x" << grid.y << "\n";
    std::cout << "Block Size   : " << block.x << "x" << block.y << "\n";
    if (!thresholds.empty()) {
        std::cout << "Threshold(s) : ";
        for (auto t : thresholds) std::cout << t << " ";
        std::cout << "\n";
    }
    std::cout << "------------------------------------------------------\n";
    std::cout << "Timing (ms)\n";
    std::cout << "  Input (Load)     : " << tInput  << "\n";
    std::cout << "  Copy HtoD        : " << tHtoD   << "\n";
    std::cout << "  Processing (GPU) : " << kernelTime << "\n";
    std::cout << "  Copy DtoH        : " << tDtoH   << "\n";
    std::cout << "  Output (Save)    : " << tOutput << "\n";
    std::cout << "======================================================\n";

    // ---------------------- Cleanup ----------------------
    cudaFree(d_in);
    cudaFree(d_out);
    if (d_thresh) cudaFree(d_thresh);
    if (d_levels) cudaFree(d_levels);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
