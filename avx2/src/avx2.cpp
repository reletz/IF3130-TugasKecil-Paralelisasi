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

// ini buat iamge load dan save (yang diproses jpg), ini mau di serial atau paralel gaadabedanya
Image loadJPG(const std::string &f) {
    cv::Mat mat = cv::imread(f, cv::IMREAD_GRAYSCALE);
    if(mat.empty()) throw std::runtime_error("Failed to load image");
    Image img(mat.cols, mat.rows);
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

    alignas(32) __m256i level_vecs[256];
    if constexpr (mode == 2) {
        int bins = thresholds.size() + 1;
        for(int i = 0; i < bins; i++){
            level_vecs[i] = _mm256_set1_epi32((255 * i) / (bins - 1));
        }
    }

    auto levels_mode = [&](__m256i g_vec) -> __m256i {
        if constexpr (mode == 0){
            __m256i gg_vec = _mm256_cvttps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(g_vec)));
            __m256i val_255 = _mm256_set1_epi32(255);
            return _mm256_min_epi32(gg_vec, val_255);
        }
        else if constexpr (mode == 1){
            // __m256i threshold_vec = _mm256_set1_epi32(thresholds[0]);
            int threshold_squared = thresholds[0] * thresholds[0];
            __m256i threshold_vec = _mm256_set1_epi32(threshold_squared);
            __m256i zeros = _mm256_setzero_si256();
            __m256i val_255 = _mm256_set1_epi32(255);
            __m256i mask = _mm256_cmpgt_epi32(g_vec, threshold_vec);
            return _mm256_blendv_epi8(val_255, zeros, mask);
        }
        else if constexpr (mode == 2){
            __m256i result = level_vecs[0];
            for (size_t i = 0; i < thresholds.size(); ++i) {
                // __m256i threshold_vec = _mm256_set1_epi32(thresholds[i]);
                int threshold_squared = thresholds[i] * thresholds[i];
                __m256i threshold_vec = _mm256_set1_epi32(threshold_squared);
                __m256i mask = _mm256_cmpgt_epi32(g_vec, threshold_vec);
                result = _mm256_blendv_epi8(result, level_vecs[i+1], mask);
            }
            return result;
        }
    };

    auto load_8bytes_to_epi32 = [](const unsigned char* ptr) -> __m256i {
        alignas(32) uint64_t temp[4] = {0};
        temp[0] = *reinterpret_cast<const uint64_t*>(ptr);
        __m256i loaded = _mm256_loadu_si256((__m256i*)temp);
        return _mm256_cvtepu8_epi32(_mm256_castsi256_si128(loaded));
    };

    auto store_8bytes_from_epi32 = [](unsigned char* ptr, __m256i vec) {
        __m256i zeros = _mm256_setzero_si256();
        __m256i packed_16 = _mm256_packus_epi32(vec, zeros);
        packed_16 = _mm256_permute4x64_epi64(packed_16, 0xD8);
        __m256i packed_8 = _mm256_packus_epi16(packed_16, zeros);
        alignas(32) uint64_t temp[4];
        _mm256_store_si256((__m256i*)temp, packed_8);
        *reinterpret_cast<uint64_t*>(ptr) = temp[0];
    };

    int num_blocks = (in.w - 2) / 8;
    int remainder = (in.w - 2) % 8;

    for(int y=1; y<in.h-1; y++){
        int x = 1;
        // Main loop: process 8 pixels at a time
        for(int block=0; block<num_blocks; block++, x+=8){
            // Load 9 neighbors for 8 pixels using AVX2 only
            __m256i v_tl = load_8bytes_to_epi32(&in.at(x - 1, y - 1));
            __m256i v_tc = load_8bytes_to_epi32(&in.at(x,     y - 1));
            __m256i v_tr = load_8bytes_to_epi32(&in.at(x + 1, y - 1));
            
            __m256i v_ml = load_8bytes_to_epi32(&in.at(x - 1, y));
            __m256i v_mc = load_8bytes_to_epi32(&in.at(x,     y));
            __m256i v_mr = load_8bytes_to_epi32(&in.at(x + 1, y));
            
            __m256i v_bl = load_8bytes_to_epi32(&in.at(x - 1, y + 1));
            __m256i v_bc = load_8bytes_to_epi32(&in.at(x,     y + 1));
            __m256i v_br = load_8bytes_to_epi32(&in.at(x + 1, y + 1));

            // Compute Sobel gradients
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

            // Compute gradient magnitude squared
            __m256i sx_sq = _mm256_mullo_epi32(sx_vec, sx_vec);
            __m256i sy_sq = _mm256_mullo_epi32(sy_vec, sy_vec);
            __m256i g_squared = _mm256_add_epi32(sx_sq, sy_sq);
            
            // Apply mode-specific processing
            __m256i g_vec = levels_mode(g_squared);

            store_8bytes_from_epi32(&out.at(x, y), g_vec);
        }
        
        // Handle remaining pixels (< 8) using masked operations
        if (remainder > 0) {
            // Create mask for remaining pixels
            alignas(32) int mask_data[8] = {0};
            for (int i = 0; i < remainder; i++) {
                mask_data[i] = -1;
            }
            __m256i mask = _mm256_load_si256((__m256i*)mask_data);
            
            // Load with boundary check (load full 8 but only use 'remainder' pixels)
            __m256i v_tl = load_8bytes_to_epi32(&in.at(x - 1, y - 1));
            __m256i v_tc = load_8bytes_to_epi32(&in.at(x,     y - 1));
            __m256i v_tr = load_8bytes_to_epi32(&in.at(x + 1, y - 1));
            
            __m256i v_ml = load_8bytes_to_epi32(&in.at(x - 1, y));
            __m256i v_mc = load_8bytes_to_epi32(&in.at(x,     y));
            __m256i v_mr = load_8bytes_to_epi32(&in.at(x + 1, y));
            
            __m256i v_bl = load_8bytes_to_epi32(&in.at(x - 1, y + 1));
            __m256i v_bc = load_8bytes_to_epi32(&in.at(x,     y + 1));
            __m256i v_br = load_8bytes_to_epi32(&in.at(x + 1, y + 1));

            // Compute Sobel gradients
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

            // Compute gradient magnitude squared
            __m256i sx_sq = _mm256_mullo_epi32(sx_vec, sx_vec);
            __m256i sy_sq = _mm256_mullo_epi32(sy_vec, sy_vec);
            __m256i g_squared = _mm256_add_epi32(sx_sq, sy_sq);
            
            // Apply mode-specific processing
            __m256i g_vec = levels_mode(g_squared);

            // Masked store for remainder pixels
            alignas(32) int result_temp[8];
            _mm256_store_si256((__m256i*)result_temp, g_vec);
            for (int i = 0; i < remainder; i++) {
                out.at(x + i, y) = static_cast<unsigned char>(result_temp[i]);
            }
        }
    }

    // Border handling
    __m256i zeros = _mm256_setzero_si256();
    
    // Top and bottom borders
    int x = 0;
    for (; x <= in.w - 32; x += 32) {
        _mm256_storeu_si256((__m256i*)&out.at(x, 0), zeros);
        _mm256_storeu_si256((__m256i*)&out.at(x, in.h - 1), zeros);
    }
    // Remaining top/bottom pixels with masked store
    if (x < in.w) {
        int rem = in.w - x;
        alignas(32) int mask_data[8] = {0};
        for (int i = 0; i < std::min(8, rem); i++) {
            mask_data[i] = -1;
        }
        __m256i mask = _mm256_load_si256((__m256i*)mask_data);
        
        for (; x < in.w; x += 8) {
            int to_write = std::min(8, in.w - x);
            alignas(32) unsigned char zero_bytes[32] = {0};
            for (int i = 0; i < to_write; i++) {
                out.at(x + i, 0) = 0;
                out.at(x + i, in.h - 1) = 0;
            }
        }
    }
    
    // Left and right borders (single pixels per row)
    for (int y = 1; y < in.h - 1; ++y) {
        out.at(0, y) = 0;
        out.at(in.w - 1, y) = 0;
    }
    
    return out;
}


int main(int argc,char*argv[]){
    if(argc<4){
        std::cerr<<"Usage: ./avx2 n input.jpg output.jpg > output.txt\n";
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
