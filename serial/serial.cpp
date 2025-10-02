#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

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

    for(int y=1;y<in.h-1;y++){
        for(int x=1;x<in.w-1;x++){
            int sx=0, sy=0;
            for(int ky=-1; ky<=1; ky++)
                for(int kx=-1; kx<=1; kx++){
                    int px=in.at(x+kx,y+ky);
                    sx += px * Gx[ky+1][kx+1];
                    sy += px * Gy[ky+1][kx+1];
                }
            int g = std::sqrt(sx*sx + sy*sy);

            if(mode == 0) { 
                out.at(x,y) = (g > 255) ? 255 : g;
            }
            else if(mode == 1) { 
                out.at(x,y) = (g > thresholds[0]) ? 0 : 255;
            }
            else {
                int bins = thresholds.size() + 1;
                std::vector<int> levels(bins);
                for(int i=0; i<bins; i++){
                    levels[i] = (255 * i) / (bins - 1);
                }

                int idx = 0;
                while(idx < thresholds.size() && g > thresholds[idx]) idx++;
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

    auto t0 = std::chrono::high_resolution_clock::now();
    Image img=loadJPG(inputFile);
    auto t1 = std::chrono::high_resolution_clock::now();
    Image res=sobel(img,mode,thresholds);
    auto t2 = std::chrono::high_resolution_clock::now();
    saveJPG(res,outputFile);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto tInput  = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    auto tProc   = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    auto tOutput = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();

    std::cout << "================ Sobel Edge Detection ================\n";
    std::cout << "Program Type : Serial\n";
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
