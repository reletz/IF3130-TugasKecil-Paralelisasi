#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <immintrin.h>
#include <memory>

struct AlignedDeleter {
    void operator()(unsigned char* p) const {
        _mm_free(p);
    }
};

struct Image {
    int w, h;
    std::unique_ptr<unsigned char[], AlignedDeleter> p;
    unsigned char& at(int x, int y) { return p.get()[y * w + x]; }
    const unsigned char& at(int x, int y) const { return p.get()[y * w + x]; }
    Image (int width, int height) : w(width), h(height) {
        p.reset((unsigned char*)_mm_malloc(w * h, 32));
        if (!p) throw std::runtime_error("Memory allocation failed");
    }
};

// // image struct
// struct Image {
//     int w, h;
//     std::vector<unsigned char> p;
//     unsigned char& at(int x, int y) { return p[y * w + x]; }
//     const unsigned char& at(int x, int y) const { return p[y * w + x]; }
// };

// ini buat iamge load dan save (yang diproses jpg), ini mau di serial atau paralel gaadabedanya
Image loadJPG(const std::string &f) {
    cv::Mat mat = cv::imread(f, cv::IMREAD_GRAYSCALE);
    if(mat.empty()) throw std::runtime_error("Failed to load image");
    Image img(mat.cols, mat.rows);
    
    // Image img{mat.cols, mat.rows, std::vector<unsigned char>(mat.cols * mat.rows)};
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
template<int mode>
Image sobel(const Image &in, const std::vector<int>& thresholds) {
    int Gx_scalar[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    int Gy_scalar[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};
    __m256i Gx[9] = {
        _mm256_set1_epi32(-1), _mm256_set1_epi32(0), _mm256_set1_epi32(1),
        _mm256_set1_epi32(-2), _mm256_set1_epi32(0), _mm256_set1_epi32(2),
        _mm256_set1_epi32(-1), _mm256_set1_epi32(0), _mm256_set1_epi32(1)
    };

    __m256i Gy[9] = {
        _mm256_set1_epi32(1), _mm256_set1_epi32(2), _mm256_set1_epi32(1),
        _mm256_set1_epi32(0), _mm256_set1_epi32(0), _mm256_set1_epi32(0),
        _mm256_set1_epi32(-1), _mm256_set1_epi32(-2), _mm256_set1_epi32(-1)
    };

    Image out(in.w, in.h);

    std::vector<__m256i> level_vecs;
    if constexpr (mode == 2) {
        int bins = thresholds.size() + 1;
        level_vecs.resize(bins);
        for(int i = 0; i < bins; i++){
            level_vecs[i] = _mm256_set1_epi32((255 * i) / (bins - 1));
        }
    }

    for(int y=1;y<in.h-1;y++){
        // __m256i top_rowc1 = _mm256_loadu_si256((__m256i*)&in.at(0,y-1));
        // __m256i top_rowc2 = _mm256_loadu_si256((__m256i*)&in.at(8,y-1));

        // __m256i mid_rowc1 = _mm256_loadu_si256((__m256i*)&in.at(0,y));
        // __m256i mid_rowc2 = _mm256_loadu_si256((__m256i*)&in.at(8,y));

        // __m256i bot_rowc1 = _mm256_loadu_si256((__m256i*)&in.at(0,y+1));
        // __m256i bot_rowc2 = _mm256_loadu_si256((__m256i*)&in.at(8,y+1));
        int x =1;

        for(;x<=in.w-9;x+=8){
            __m128i v_tl_8b = _mm_loadl_epi64((__m128i const*)&in.at(x - 1, y - 1));
            __m128i v_tc_8b = _mm_loadl_epi64((__m128i const*)&in.at(x,     y - 1));
            __m128i v_tr_8b = _mm_loadl_epi64((__m128i const*)&in.at(x + 1, y - 1));
            
            __m128i v_ml_8b = _mm_loadl_epi64((__m128i const*)&in.at(x - 1, y));
            __m128i v_mc_8b = _mm_loadl_epi64((__m128i const*)&in.at(x,     y));
            __m128i v_mr_8b = _mm_loadl_epi64((__m128i const*)&in.at(x + 1, y));

            __m128i v_bl_8b = _mm_loadl_epi64((__m128i const*)&in.at(x - 1, y + 1));
            __m128i v_bc_8b = _mm_loadl_epi64((__m128i const*)&in.at(x,     y + 1));
            __m128i v_br_8b = _mm_loadl_epi64((__m128i const*)&in.at(x + 1, y + 1));

            __m256i v_tl = _mm256_cvtepu8_epi32(v_tl_8b);
            __m256i v_tc = _mm256_cvtepu8_epi32(v_tc_8b);
            __m256i v_tr = _mm256_cvtepu8_epi32(v_tr_8b);

            __m256i v_ml = _mm256_cvtepu8_epi32(v_ml_8b);
            __m256i v_mc = _mm256_cvtepu8_epi32(v_mc_8b);
            __m256i v_mr = _mm256_cvtepu8_epi32(v_mr_8b);
            
            __m256i v_bl = _mm256_cvtepu8_epi32(v_bl_8b);
            __m256i v_bc = _mm256_cvtepu8_epi32(v_bc_8b);
            __m256i v_br = _mm256_cvtepu8_epi32(v_br_8b);

            __m256i sx_vec = _mm256_setzero_si256();
            __m256i sy_vec = _mm256_setzero_si256();

            sx_vec = _mm256_add_epi32(sx_vec, _mm256_mullo_epi32(v_tl, Gx[0]));
            sy_vec = _mm256_add_epi32(sy_vec, _mm256_mullo_epi32(v_tl, Gy[0]));
            sx_vec = _mm256_add_epi32(sx_vec, _mm256_mullo_epi32(v_tc, Gx[1]));
            sy_vec = _mm256_add_epi32(sy_vec, _mm256_mullo_epi32(v_tc, Gy[1]));
            sx_vec = _mm256_add_epi32(sx_vec, _mm256_mullo_epi32(v_tr, Gx[2]));
            sy_vec = _mm256_add_epi32(sy_vec, _mm256_mullo_epi32(v_tr, Gy[2]));

            sx_vec = _mm256_add_epi32(sx_vec, _mm256_mullo_epi32(v_ml, Gx[3]));
            sy_vec = _mm256_add_epi32(sy_vec, _mm256_mullo_epi32(v_ml, Gy[3]));
            sx_vec = _mm256_add_epi32(sx_vec, _mm256_mullo_epi32(v_mc, Gx[4]));
            sy_vec = _mm256_add_epi32(sy_vec, _mm256_mullo_epi32(v_mc, Gy[4]));
            sx_vec = _mm256_add_epi32(sx_vec, _mm256_mullo_epi32(v_mr, Gx[5]));
            sy_vec = _mm256_add_epi32(sy_vec, _mm256_mullo_epi32(v_mr, Gy[5]));

            sx_vec = _mm256_add_epi32(sx_vec, _mm256_mullo_epi32(v_bl, Gx[6]));
            sy_vec = _mm256_add_epi32(sy_vec, _mm256_mullo_epi32(v_bl, Gy[6]));
            sx_vec = _mm256_add_epi32(sx_vec, _mm256_mullo_epi32(v_bc, Gx[7]));
            sy_vec = _mm256_add_epi32(sy_vec, _mm256_mullo_epi32(v_bc, Gy[7]));
            sx_vec = _mm256_add_epi32(sx_vec, _mm256_mullo_epi32(v_br, Gx[8]));
            sy_vec = _mm256_add_epi32(sy_vec, _mm256_mullo_epi32(v_br, Gy[8]));

            __m256i sx_sq = _mm256_mullo_epi32(sx_vec, sx_vec);
            __m256i sy_sq = _mm256_mullo_epi32(sy_vec, sy_vec);
            __m256i g = _mm256_add_epi32(sx_sq, sy_sq);

            // __m256 g_sqf = _mm256_cvtepi32_ps(g);
            // __m256 g_f = _mm256_sqrt_ps(g_sqf);
            // __m256i g_vec = _mm256_cvttps_epi32(g_f);
            __m256i g_vec = _mm256_cvttps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(g)));

            if constexpr (mode == 0){
                __m256i val_255 = _mm256_set1_epi32(255);
                g_vec = _mm256_min_epi32(g_vec, val_255);
            }
            else if constexpr (mode == 1){
                __m256i threshold_vec = _mm256_set1_epi32(thresholds[0]);
                __m256i zeros = _mm256_setzero_si256();
                __m256i val_255 = _mm256_set1_epi32(255);
                __m256i mask = _mm256_cmpgt_epi32(g_vec, threshold_vec);
                g_vec = _mm256_blendv_epi8(val_255, zeros, mask);
            }
            else if constexpr (mode == 2){
                __m256i result = level_vecs[0];
                for (size_t i = 0; i < thresholds.size(); ++i) {
                    __m256i threshold_vec = _mm256_set1_epi32(thresholds[i]);
                    __m256i mask = _mm256_cmpgt_epi32(g_vec, threshold_vec);
                    result = _mm256_blendv_epi8(g_vec, level_vecs[i+1], mask);
                }
                g_vec = result;
            }

            __m256i zeros_256 = _mm256_setzero_si256();
            __m256i res_16bit_lanes = _mm256_packus_epi32(g_vec, zeros_256);
            __m256i permuted = _mm256_permute4x64_epi64(res_16bit_lanes, 0xD8); // 11011000
            __m128i res_16bit = _mm256_castsi256_si128(permuted);

            __m128i zeros_128 = _mm_setzero_si128();
            __m128i finalP = _mm_packus_epi16(res_16bit, zeros_128);
            _mm_storel_epi64((__m128i*)&out.at(x, y), finalP);

            // top_rowc1 = top_rowc2;
            // mid_rowc1 = mid_rowc2;
            // bot_rowc1 = bot_rowc2;
            // top_rowc2 = _mm256_loadu_si256((__m256i*)&in.at(x + 16 -1, y - 1));
            // mid_rowc2 = _mm256_loadu_si256((__m256i*)&in.at(x + 16 -1, y));
            // bot_rowc2 = _mm256_loadu_si256((__m256i*)&in.at(x + 16 -1, y + 1));
        }
        for(;x<in.w-1;x++){
            int sx=0, sy=0;
            for(int ky=-1; ky<=1; ky++)
                for(int kx=-1; kx<=1; kx++){
                    int px=in.at(x+kx,y+ky);
                    sx += px * Gx_scalar[ky+1][kx+1];
                    sy += px * Gy_scalar[ky+1][kx+1];
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
    for (int y = 0; y < in.h; ++y) {
        out.at(0, y) = in.at(0, y);
        out.at(in.w - 1, y) = in.at(in.w - 1, y);
    }
    for (int x = 1; x < in.w - 1; ++x) {
        out.at(x, 0) = in.at(x, 0);
        out.at(x, in.h - 1) = in.at(x, in.h - 1);
    }
    // for(int y=1;y<in.h-1;y++){
    //     for(int x=1;x<in.w-1;x++){
    //         int sx=0, sy=0;
    //         for(int ky=-1; ky<=1; ky++)
    //             for(int kx=-1; kx<=1; kx++){
    //                 int px=in.at(x+kx,y+ky);
    //                 sx += px * Gx[ky+1][kx+1];
    //                 sy += px * Gy[ky+1][kx+1];
    //             }
    //         int g = std::sqrt(sx*sx + sy*sy);

    //         if(mode == 0) { 
    //             out.at(x,y) = (g > 255) ? 255 : g;
    //         }
    //         else if(mode == 1) { 
    //             out.at(x,y) = (g > thresholds[0]) ? 0 : 255;
    //         }
    //         else {
    //             int bins = thresholds.size() + 1;
    //             std::vector<int> levels(bins);
    //             for(int i=0; i<bins; i++){
    //                 levels[i] = (255 * i) / (bins - 1);
    //             }

    //             int idx = 0;
    //             while(idx < thresholds.size() && g > thresholds[idx]) idx++;
    //             out.at(x,y) = levels[idx];
    //         }
    //     }
    // }
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
    Image res(img.w, img.h);
    switch (mode) {
        case 0:
            res = sobel<0>(img, thresholds);
            break;
        case 1:
            res = sobel<1>(img, thresholds);
            break;
        case 2:
            res = sobel<2>(img, thresholds);
            break;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    saveJPG(res,outputFile);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto tInput  = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    auto tProc   = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    auto tOutput = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();

    std::cout << "================ Sobel Edge Detection ================\n";
    std::cout << "Program Type : AVX2\n";
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
