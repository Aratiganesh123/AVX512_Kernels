#include<iostream>
#include <immintrin.h>
#include<random>
#include<chrono>
#include <immintrin.h>
using namespace std;


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

void matmul_regular(float* A , float* B , float* C , int M , int N , int K)
{
    for(int i = 0 ; i < M ; i++)
    {
        for(int j = 0 ; j < N ; j++)
        {
            __m256 sum = _mm256_setzero_ps();
            for(int k = 0 ; k < K ; k+=8)
            {   
                __m256 vecA = _mm256_loadu_ps(&A[i* K + k]);
                __m256 vecB = _mm256_set_ps(
                    B[(k + 7)* N + j],
                    B[(k + 6)* N + j],
                    B[(k + 5)* N + j],
                    B[(k + 4)* N + j],
                    B[(k + 3)* N + j],
                    B[(k + 2)* N + j],
                    B[(k + 1)* N + j],
                    B[(k)* N + j]
                );

                sum = _mm256_fmadd_ps(vecA, vecB, sum);
                
            }   
            float val = hsum_avx(sum);
            C[i*N + j] = val;
        }
    }
}

void transpose_block(float* src, float* dst, int N, int K) {
    for(int i = 0; i < K; i += 8) {
        for(int j = 0; j < N; j += 8) {
            for(int ii = 0; ii < 8 && i + ii < K; ii++) {
                for(int jj = 0; jj < 8 && j + jj < N; jj++) {
                    dst[(j + jj)*K + i + ii] = src[(i+ii)*N + j + jj];
                }
            }
        }
    }
}

void matmul_tranpose(float* A, float* B, float* C, int M, int N, int K) {
    // Allocate space for transposed B
    float* B_trans = new float[K*N];
    
    // Transpose B once
    transpose_block(B, B_trans, N, K);

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            __m256 sum = _mm256_setzero_ps();
            
            for(int k = 0; k < K; k += 8) {
                __m256 vecA = _mm256_loadu_ps(&A[i*K + k]);
                __m256 vecB = _mm256_loadu_ps(&B_trans[j*K + k]);
                sum = _mm256_fmadd_ps(vecA, vecB, sum);
            }
            C[i*N + j] = hsum_avx(sum);
        }
    }

    delete[] B_trans;
}

void matmul_nonstrided(float* A, float* B, float* C, int M, int N, int K)
{
    for(int i = 0 ; i < M ;i++)
    {
        for(int j = 0 ; j < N ; j+=8)
        {
            __m256 sum = _mm256_setzero_ps();
            for(int k = 0 ; k < K ; k++)
            {
                __m256 a = _mm256_broadcast_ss(&A[i*K + k]);
                __m256 b = _mm256_loadu_ps(&B[k*N + j]);

                sum = _mm256_fmadd_ps(a , b , sum);
            }

            _mm256_storeu_ps(&C[i*N + j], sum);
        }
    }

}

void matmul_reference(float* A, float* B, float* C, int M, int N, int K) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0;
            for(int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

void verify_result(float* C, float* C_ref, int M, int N, const string& name) {
    float max_diff = 0.0f;
    for(int i = 0; i < M*N; i++) {
        max_diff = max(max_diff, abs(C[i] - C_ref[i]));
    }
    cout << name << " max difference from reference: " << max_diff << endl;
    if(max_diff < 1e-3) {
        cout << name << " verification PASSED!" << endl;
    } else {
        cout << name << " verification FAILED!" << endl;
    }
}

int main() {
    int N = 1 << 10;  // 1024
    int M = 1 << 10;  // 1024
    int K = 1 << 10;  // 1024

    float* A = new float[M*K];
    float* B = new float[K*N];
    float* C1 = new float[M*N];  // For regular
    float* C2 = new float[M*N];  // For transpose
    float* C3 = new float[M*N];  // For nonstrided
    float* C_ref = new float[M*N];  // For reference

    if (!A || !B || !C1 || !C2 || !C3 || !C_ref) {
        cout << "Memory allocation failed!" << endl;
        return 1;
    }

    // Initialize matrices
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(1.0f, 100.0f);

    for(int i = 0; i < M*K; i++) A[i] = dis(gen);
    for(int i = 0; i < K*N; i++) B[i] = dis(gen);

    // Compute reference result
    cout << "Computing reference result..." << endl;
    matmul_reference(A, B, C_ref, M, N, K);

    // Benchmark parameters
    const int iterations = 3;
    double gflops = 2.0 * M * N * K * 1e-9; // multiply-add = 2 ops

    // 1. Regular version
    {
        auto start = chrono::high_resolution_clock::now();
        for(int iter = 0; iter < iterations; iter++) {
            matmul_regular(A, B, C1, M, N, K);
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
        double seconds = duration / 1e6 / iterations;
        
        cout << "\nRegular Version:" << endl;
        cout << "Average time: " << duration/iterations << " microseconds" << endl;
        cout << "GFLOPS: " << gflops/seconds << endl;
        verify_result(C1, C_ref, M, N, "Regular");
    }

    // 2. Transpose version
    {
        auto start = chrono::high_resolution_clock::now();
        for(int iter = 0; iter < iterations; iter++) {
            matmul_tranpose(A, B, C2, M, N, K);
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
        double seconds = duration / 1e6 / iterations;
        
        cout << "\nTranspose Version:" << endl;
        cout << "Average time: " << duration/iterations << " microseconds" << endl;
        cout << "GFLOPS: " << gflops/seconds << endl;
        verify_result(C2, C_ref, M, N, "Transpose");
    }

    // 3. Non-strided version
    {
        auto start = chrono::high_resolution_clock::now();
        for(int iter = 0; iter < iterations; iter++) {
            matmul_nonstrided(A, B, C3, M, N, K);
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
        double seconds = duration / 1e6 / iterations;
        
        cout << "\nNon-strided Version:" << endl;
        cout << "Average time: " << duration/iterations << " microseconds" << endl;
        cout << "GFLOPS: " << gflops/seconds << endl;
        verify_result(C3, C_ref, M, N, "Non-strided");
    }

    // Memory cleanup
    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
    delete[] C3;
    delete[] C_ref;

    return 0;
}