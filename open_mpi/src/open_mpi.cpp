#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <mpi.h>

// image struct
struct Image {
    int w, h;
    std::vector<unsigned char> p;
    unsigned char& at(int x, int y) { return p[y * w + x]; }
    const unsigned char& at(int x, int y) const { return p[y * w + x]; }
};

// ini buat iamge load dan save (yang diproses jpg), ini mau di serial atau paralel gaadabedanya
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

// implementasi algorimta nya
Image sobel(const Image &in, int mode, const std::vector<int>& thresholds) {
    int Gx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    int Gy[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};
    Image out=in;

    std::vector<int> levels;
    int thresh_size = thresholds.size();
    if (mode == 2){
        int bins = thresholds.size() + 1;
        levels.resize(bins);
        for(int i=0; i<bins; i++){
            levels[i] = (255 * i) / (bins - 1);
        }
    }

    for(int y=1;y<in.h-1;y++){
        for(int x=1;x<in.w-1;x++){
            int sx=0, sy=0;
            for(int ky=-1; ky<=1; ky++)
                for(int kx=-1; kx<=1; kx++){
                    int px=in.at(x+kx,y+ky);
                    sx += px * Gx[ky+1][kx+1];
                    sy += px * Gy[ky+1][kx+1];
                }
            int g_squared = sx*sx + sy*sy;

            if (mode == 0) { 
                int g = std::sqrt(g_squared);
                out.at(x,y) = (g > 255) ? 255 : g;
            }
            else if (mode == 1) { 
                int threshold_squared = thresholds[0] * thresholds[0];
                out.at(x,y) = (g_squared > threshold_squared) ? 0 : 255;
            }
            else {
                int idx = 0;
                while(idx < thresholds.size() && g_squared > thresholds[idx] * thresholds[idx]) idx++;
                out.at(x,y) = levels[idx];
            }
        }
    }
    return out;
}


int main(int argc,char*argv[]){
    if(argc<4){
        std::cerr<<"Usage: ./main n input.jpg output.jpg > output.txt\n";
        return 1;
    }

    int rank, size, imageWidth;

    int n = std::stoi(argv[1]);
    std::string inputFile  = argv[2];
    std::string outputFile = argv[3];

    if (n < 0) {
        std::cerr<<"Error: n must be greater than or equal to 0\n";
        return 1;
    }

    
    int mode;
    std::vector<int> thresholds;
    
    if (n == 0) {
        mode = 0; // gradient magnitude
    } else if (n == 1) {
        mode = 1; // Binary threshold
        thresholds.push_back(128);
    } else {
        mode = 2; // Multi-level thresholds
        for (int i = 1; i < n; i++) {
            int threshold = (255 * i) / n;
            thresholds.push_back(threshold);
        }
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Image img, finalImg;
    auto t0 = std::chrono::high_resolution_clock::now();
    long long tInput = 0;

    if (rank == 0) {
        img=loadJPG(inputFile);
        auto t1 = std::chrono::high_resolution_clock::now();
        tInput = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    }

    imageWidth = rank == 0 ? img.w : 0;
    MPI_Bcast(&imageWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int threshold_count = rank == 0 ? thresholds.size() : 0;
    MPI_Bcast(&threshold_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (threshold_count > 0) {
        MPI_Bcast(thresholds.data(), threshold_count, MPI_INT, 0, MPI_COMM_WORLD);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    int total_rows = (rank == 0) ? img.h : 0;
    MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int rows_per_slave = total_rows / size;
    int remainder_rows = total_rows % size;
    if (rank == 0) {
        finalImg = Image{ imageWidth, total_rows, std::vector<unsigned char>(imageWidth * total_rows) };
    }
    int my_start_row = rank * rows_per_slave + std::min(rank, remainder_rows);
    int my_num_rows = rows_per_slave + (rank < remainder_rows ? 1 : 0);
    int recv_start_row = my_start_row == 0 ? 0 : my_start_row - 1;
    int recv_end_row = (my_start_row + my_num_rows == total_rows) ? total_rows : my_start_row + my_num_rows + 1;
    int recv_num_rows = recv_end_row - recv_start_row;
    int pixels_to_send = recv_num_rows * imageWidth;
    std::vector<unsigned char> local_slices(pixels_to_send);
    if (rank != 0) {
        //slave do work here
        MPI_Recv(local_slices.data(), pixels_to_send, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else { // master
        int row_start = 0;
        for (int slave = 0; slave < size; ++slave) {
            int rows_for_slave = rows_per_slave + (slave < remainder_rows ? 1 : 0);
            int send_start = row_start == 0 ? 0 : row_start - 1;
            int send_end = (row_start + rows_for_slave == total_rows) ? total_rows : row_start + rows_for_slave + 1;
            int send_cnt = (send_end - send_start) * imageWidth;

            if (slave == 0) {
                std::copy(&img.p[send_start * imageWidth], &img.p[send_start * imageWidth] + send_cnt, local_slices.begin());
            } else {
                MPI_Send(&img.p[send_start * imageWidth], send_cnt, MPI_UNSIGNED_CHAR, slave, 0, MPI_COMM_WORLD);
            }
            row_start += rows_for_slave;
        }
    }

    Image local_chunk{imageWidth, recv_num_rows, local_slices};
    Image processed_chunk = sobel(local_chunk, mode, thresholds);

    int res_pixel_offset = my_start_row == 0 ? 0 : imageWidth;
    int res_pixel_count = my_num_rows * imageWidth;

    std::vector<int> recvcount(size);
    std::vector<int> displacement(size);

    if (rank == 0) {
        int row_start = 0;
        for (int slave = 0; slave < size; ++slave) {
            int rows_for_slave = rows_per_slave + (slave < remainder_rows ? 1 : 0);
            recvcount[slave] = rows_for_slave * imageWidth;
            displacement[slave] = row_start * imageWidth;
            row_start += rows_for_slave;
        }
    }

    MPI_Gatherv(processed_chunk.p.data() + res_pixel_offset,
                res_pixel_count,
                MPI_UNSIGNED_CHAR,
                finalImg.p.data(),
                recvcount.data(),
                displacement.data(),
                MPI_UNSIGNED_CHAR,
                0,
                MPI_COMM_WORLD);

    auto t3 = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        auto t4 = std::chrono::high_resolution_clock::now();
        saveJPG(finalImg, outputFile);
        auto t5 = std::chrono::high_resolution_clock::now();
        auto tProc   = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();
        auto tOutput = std::chrono::duration_cast<std::chrono::milliseconds>(t5-t4).count();

        std::cout << "================ Sobel Edge Detection ================\n";
        std::cout << "Program Type : Paralel (Open MPI)\n";
        std::cout << "------------------------------------------------------\n";
        std::cout << "Mode         : " << mode << "\n";
        if (!thresholds.empty()) {
            std::cout << "Threshold(s) : ";
            for (auto t : thresholds) std::cout << t << " ";
            std::cout << "\n";
        }
        std::cout << "------------------------------------------------------\n";
        std::cout << "Timing (ms)\n";
        std::cout << "  Input      : " << tInput  << "\n";
        std::cout << "  Processing : " << tProc   << "\n";
        std::cout << "  Output     : " << tOutput << "\n";
        std::cout << "======================================================\n";
    }

    MPI_Finalize();
    return 0;
}
