#include <iostream>
#include <immintrin.h>
#include <random>
#include <chrono>
#include <vector>

using namespace std;

// Basic convolution implementation
void conv2d_basic(const float* input, const float* kernel, float* output,
                 int height, int width, int kernel_size, int stride) {
    int out_height = (height - kernel_size) / stride + 1;
    int out_width = (width - kernel_size) / stride + 1;
    
    for(int oh = 0; oh < out_height; oh++) {
        for(int ow = 0; ow < out_width; ow++) {
            float sum = 0;
            for(int kh = 0; kh < kernel_size; kh++) {
                for(int kw = 0; kw < kernel_size; kw++) {
                    int ih = oh * stride + kh;
                    int iw = ow * stride + kw;
                    sum += input[ih * width + iw] * kernel[kh * kernel_size + kw];
                }
            }
            output[oh * out_width + ow] = sum;
        }
    }
}

// SIMD convolution implementation
void conv2d_simd(const float* input, const float* kernel, float* output,
                int height, int width, int kernel_size, int stride) {
    int out_height = (height - kernel_size) / stride + 1;
    int out_width = (width - kernel_size) / stride + 1;
    
    for(int oh = 0; oh < out_height; oh++) {
        for(int ow = 0; ow < out_width; ow += 4) {
            // Handle boundary case
            if(ow + 4 > out_width) {
                for(int w = ow; w < out_width; w++) {
                    float sum = 0;
                    for(int kh = 0; kh < kernel_size; kh++) {
                        for(int kw = 0; kw < kernel_size; kw++) {
                            int ih = oh * stride + kh;
                            int iw = w * stride + kw;
                            sum += input[ih * width + iw] * kernel[kh * kernel_size + kw];
                        }
                    }
                    output[oh * out_width + w] = sum;
                }
                continue;
            }
            
            __m128 sum = _mm_setzero_ps();
            for(int kh = 0; kh < kernel_size; kh++) {
                for(int kw = 0; kw < kernel_size; kw++) {
                    __m128 in = _mm_loadu_ps(&input[(oh * stride + kh) * width + (ow * stride + kw)]);
                    __m128 k = _mm_set1_ps(kernel[kh * kernel_size + kw]);
                    sum = _mm_add_ps(sum, _mm_mul_ps(in, k));
                }
            }
            _mm_storeu_ps(&output[oh * out_width + ow], sum);
        }
    }
}

void weight_stationary_convolution(const float* input, const float* kernel, float* output,
                                 int height, int width, int kernel_size, int stride) {
    int out_height = (height - kernel_size) / stride + 1;
    int out_width = (width - kernel_size) / stride + 1;

    // Initialize output to zero
    std::memset(output, 0, out_height * out_width * sizeof(float));

    // For each kernel weight
    for(int kh = 0; kh < kernel_size; kh++) {
        for(int kw = 0; kw < kernel_size; kw++) {
            // Load single kernel weight into all 8 SIMD lanes
            __m256 kernel_val = _mm256_set1_ps(kernel[kh * kernel_size + kw]);
            
            // For each output row
            for(int oh = 0; oh < out_height; oh++) {
                int ih = oh * stride + kh;  // Input row
                
                // Process 8 output columns at once
                for(int ow = 0; ow < out_width; ow += 8) {
                    // Handle boundary case
                    if(ow + 8 > out_width) {
                        // Process remaining elements one by one
                        for(int w = ow; w < out_width; w++) {
                            int iw = w * stride + kw;
                            output[oh * out_width + w] +=
                                input[ih * width + iw] * kernel[kh * kernel_size + kw];
                        }
                        continue;
                    }

                    // Load 8 input values
                    __m256 in = _mm256_loadu_ps(&input[ih * width + ow * stride + kw]);
                    
                    // Load current output values
                    __m256 out = _mm256_loadu_ps(&output[oh * out_width + ow]);
                    
                    // Multiply input with kernel weight
                    __m256 product = _mm256_mul_ps(in, kernel_val);
                    
                    // Add to output
                    out = _mm256_add_ps(out, product);
                    
                    // Store back to output
                    _mm256_storeu_ps(&output[oh * out_width + ow], out);
                }
            }
        }
    }
}

// Verify results between basic and SIMD implementations
bool verify_results(const float* output1, const float* output2, int size) {
    const float epsilon = 1e-5;
    for(int i = 0; i < size; i++) {
        if(abs(output1[i] - output2[i]) > epsilon) {
            cout << "Mismatch at index " << i << ": " 
                 << output1[i] << " vs " << output2[i] << endl;
            return false;
        }
    }
    return true;
}

// Driver function to test and compare implementations
void test_convolution(int height, int width, int kernel_size, int stride, int num_runs) {
    // Allocate memory
    vector<float> input(height * width);
    vector<float> kernel(kernel_size * kernel_size);
    int out_height = (height - kernel_size) / stride + 1;
    int out_width = (width - kernel_size) / stride + 1;
    vector<float> output_basic(out_height * out_width);
    vector<float> output_simd(out_height * out_width);
    
    // Generate random input data
    generate_random_data(input.data(), height * width);
    generate_random_data(kernel.data(), kernel_size * kernel_size);
    
    // Timing variables
    chrono::duration<double, milli> basic_time(0);
    chrono::duration<double, milli> simd_time(0);
    
    // Run multiple times for accurate timing
    for(int run = 0; run < num_runs; run++) {
        // Time basic implementation
        auto start = chrono::high_resolution_clock::now();
        conv2d_basic(input.data(), kernel.data(), output_basic.data(),
                    height, width, kernel_size, stride);
        auto end = chrono::high_resolution_clock::now();
        basic_time += end - start;
        
        // Time SIMD implementation
        start = chrono::high_resolution_clock::now();
        conv2d_simd(input.data(), kernel.data(), output_simd.data(),
                   height, width, kernel_size, stride);
        end = chrono::high_resolution_clock::now();
        simd_time += end - start;
    }
    
    // Verify results
    bool results_match = verify_results(output_basic.data(), output_simd.data(),
                                      out_height * out_width);
    
    // Print results
    cout << "Input size: " << height << "x" << width << endl;
    cout << "Kernel size: " << kernel_size << "x" << kernel_size << endl;
    cout << "Stride: " << stride << endl;
    cout << "Output size: " << out_height << "x" << out_width << endl;
    cout << "Average time over " << num_runs << " runs:" << endl;
    cout << "Basic implementation: " << basic_time.count()/num_runs << " ms" << endl;
    cout << "SIMD implementation: " << simd_time.count()/num_runs << " ms" << endl;
    cout << "Speedup: " << basic_time.count()/simd_time.count() << "x" << endl;
    cout << "Results match: " << (results_match ? "Yes" : "No") << endl;
}

int main() {
    // Test cases
    cout << "Small input test:" << endl;
    test_convolution(32, 32, 3, 1, 100);
    
    cout << "\nMedium input test:" << endl;
    test_convolution(128, 128, 3, 1, 100);
    
    cout << "\nLarge input test:" << endl;
    test_convolution(512, 512, 3, 1, 10);
    
    cout << "\nDifferent stride test:" << endl;
    test_convolution(128, 128, 3, 2, 100);
    
    cout << "\nLarger kernel test:" << endl;
    test_convolution(128, 128, 5, 1, 100);
    
    return 0;
}