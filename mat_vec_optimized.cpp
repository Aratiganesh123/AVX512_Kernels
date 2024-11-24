#include<iostream>
#include <immintrin.h>
#include<random>
#include<chrono>
#include <immintrin.h>
using namespace std;

void transpose_block(float* src, float* dst, int N, int K) {
    for(int i = 0; i < K; i += 8) {
        for(int j = 0; j < N; j += 8) {
            for(int ii = 0; ii < 8 && i + ii < K; ii++) {
                for(int jj = 0; jj < 8 && j + jj < N; jj++) {
                    dst[j*K + i + ii*N + jj] = src[(i+ii)*N + j + jj];
                }
            }
        }
    }
}

inline float hsum_avx(__m256 sum) {
    __m128 vlow  = _mm256_castps256_ps128(sum);
    __m128 vhigh = _mm256_extractf128_ps(sum, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    sums        = _mm_add_ss(sums, shuf);
    return      _mm_cvtss_f32(sums);
}
void matvec_mul1(float* mat , float* vec, float* result, int M, int N )
{
    for(int i = 0 ; i < M ; i++)
    {
        __m256 sum = _mm256_setzero_ps();
        for(int j = 0 ; j < N ; j+=8)
        {
            __m256 mv = _mm256_loadu_ps(&mat[i*N + j]);
            __m256 vv = _mm256_loadu_ps(&vec[j]);
            sum = _mm256_fmadd_ps(mv ,vv , sum);
        }

            result[i] = hsum_avx(sum);
        }
}


void matvec_mul2(float* mat , float* vec, float* result, int M, int N )
{

    float* mat_trans = new float[M*N];

    transpose_block(mat , mat_trans , N, M);
    for(int i = 0 ; i < M ; i++)
    {
        __m256 sum = _mm256_setzero_ps();
        for(int j = 0 ; j < N ; j+=8)
        {
            __m256 mv = _mm256_loadu_ps(&mat[i*N + j]);
            __m256 vv = _mm256_loadu_ps(&vec[j]);
            sum = _mm256_fmadd_ps(mv ,vv , sum);
        }

            result[i] = hsum_avx(sum);
        }

        delete [] mat_trans;
}

    


void matvec_ref(float* mat, float* vec, float* result, int M, int N) {
    for(int i = 0; i < M; i++) {
        float sum = 0.0f;
        for(int j = 0; j < N; j++) {
            sum += mat[i*N + j] * vec[j];
        }
        result[i] = sum;
    }
}

int main() {
    // Matrix dimensions
    int M = 1<<20;  // rows
    int N = 1<<20;  // cols

    // Ensure N is multiple of 8 for AVX
    N = (N + 7) & ~7;

    // Allocate memory
    float* mat = new float[M*N];
    float* vec = new float[N];
    float* result1 = new float[M];
    float* result2 = new float[M];
    float* result_ref = new float[M];

    if (!mat || !vec || !result1 || !result2 || !result_ref) {
        cout << "Memory allocation failed!" << endl;
        return 1;
    }

    // Initialize with random values
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(1.0f, 100.0f);

    for(int i = 0; i < M*N; i++) {
        mat[i] = dis(gen);
    }
    for(int i = 0; i < N; i++) {
        vec[i] = dis(gen);
    }

    // Compute reference result
    matvec_ref(mat, vec, result_ref, M, N);

    // Benchmark parameters
    const int iterations = 10;
    double ops = 2.0 * M * N;  // multiply-add = 2 ops

    // Benchmark regular version
    auto start = chrono::high_resolution_clock::now();
    
    for(int iter = 0; iter < iterations; iter++) {
        matvec_mul1(mat, vec, result1, M, N);
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::microseconds>(end - start).count();

    // Benchmark transpose version
    start = chrono::high_resolution_clock::now();
    
    for(int iter = 0; iter < iterations; iter++) {
        matvec_mul2(mat, vec, result2, M, N);
    }
    
    end = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::microseconds>(end - start).count();

    // Calculate GFLOPS
    double gflops1 = (ops * iterations) / (duration1 * 1e3);  // GFLOPS for version 1
    double gflops2 = (ops * iterations) / (duration2 * 1e3);  // GFLOPS for version 2

    // Verify results
    float max_diff1 = 0.0f;
    float max_diff2 = 0.0f;
    for(int i = 0; i < M; i++) {
        max_diff1 = max(max_diff1, abs(result1[i] - result_ref[i]));
        max_diff2 = max(max_diff2, abs(result2[i] - result_ref[i]));
    }

    // Print results
    cout << "\nRegular Version:" << endl;
    cout << "Time: " << duration1/iterations << " microseconds" << endl;
    cout << "GFLOPS: " << gflops1 << endl;
    cout << "Max Difference: " << max_diff1 << endl;

    cout << "\nTranspose Version:" << endl;
    cout << "Time: " << duration2/iterations << " microseconds" << endl;
    cout << "GFLOPS: " << gflops2 << endl;
    cout << "Max Difference: " << max_diff2 << endl;

    // Memory cleanup
    delete[] mat;
    delete[] vec;
    delete[] result1;
    delete[] result2;
    delete[] result_ref;

    return 0;
}

