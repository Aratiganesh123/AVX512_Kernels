#include<iostream>
#include <immintrin.h>
#include<random>
#include<chrono>
using namespace std;


//Naive - for optimizations you could prefetch from 16 cache lines away or implement some loop unrolling mech
void vector_add(float*a , float* b , float* c , int n)
{
    int i; 
    for(i = 0 ; i < n - 8 ; i+=8)
    {
        __m256 va = _mm256_loadu_ps(&a[i]); 
        __m256 vb = _mm256_loadu_ps(&b[i]);  
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&c[i], vc);
    }

    for(; i < n ; i++)
    {
        c[i] = a[i] + b[i];
    }
}

float horizontal_sum(__m256 v){
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v,1);
    vlow = _mm_add_ps(vlow , vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sum = _mm_add_ps(shuf , sum);
    shuf = _mm_movehl_ps(shuf , sum);
    sum = _mm_add_ss(sum, shuf);    

    return _mm_cvtss_f32(sum);
}

float reduction(float*a ,  int n)
{
    __m256 sum_ans = _mm256_setzero_ps();

    int i = 0 ;
    for(; i <= n - 8 ; i+=8)
    {
        __m256 v = _mm256_loadu_ps(&a[i]);
        sum_ans = _mm256_add_ps(sum_ans , v);
    }

    float sum = horizontal_sum(sum_ans);

    for(; i < n ; i++)
    {
        sum+=a[i];
    }
    return sum;
}

void relu(float* input , float* output , int n)
{
    __m256 zeros = _mm256_setzero_ps();
    int i = 0 ;
    for(; i <= n - 8 ; i+=8)
    {
        __m256 x = _mm256_loadu_ps(&input[i]);

        __m256 result = _mm256_max_ps(zeros , x);
        _mm256_storeu_ps(&output[i] , result);
    }

    for(; i < n ; i++)
    {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }
}

void relu_inplace(float* data , int n)
{
    __m256 zeros = _mm256_setzero_ps();

    int i = 0 ;
    for(; i <= n - 32 ; i+=32)
    {
        __m256 x1 = _mm256_loadu_ps(&data[i]);
        __m256 x2 = _mm256_loadu_ps(&data[i+8]);
        __m256 x3 = _mm256_loadu_ps(&data[i+16]);
        __m256 x4 = _mm256_loadu_ps(&data[i+24]);

        x1 = _mm256_max_ps(zeros, x1);
        x2 = _mm256_max_ps(zeros, x2);
        x3 = _mm256_max_ps(zeros, x3);
        x4 = _mm256_max_ps(zeros, x4);

        _mm256_storeu_ps(&data[i], x1);
        _mm256_storeu_ps(&data[i + 8], x2);
        _mm256_storeu_ps(&data[i + 16], x3);
        _mm256_storeu_ps(&data[i + 24], x4);

    }

    for(; i < n; i++) {
        data[i] = (data[i] > 0) ? data[i] : 0;
    }
}


int main()
{

    int n = 4093;
    float* A = new float[n];
    float* B = new float[n];
    float* C = new float[n];

        if (!A || !B || !C) {
        std::cout << "Memory allocation failed!" << std::endl;
        return 1;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f,100.0f);

    for(int i = 0 ; i < n ; i++)
    {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }


    //multiple of 2
    const int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for(int iter = 0; iter < iterations; iter++) {
        vector_add(A, B, C, n);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Average time per iteration: " 
              << (duration / static_cast<double>(iterations)) 
              << " microseconds" << std::endl;



    for(int i = 0; i < n; i++) {
        float expected = A[i] + B[i];
        if (abs(C[i] - expected) > 1e-5) {
            std::cout << "Mismatch at " << i << ": "
                     << C[i] << " != " << expected << std::endl;
        }
    }


    //Call reductiion

    auto t1 = std::chrono::high_resolution_clock::now();
    
    for(int iter = 0; iter < iterations; iter++) {
        reduction(A,  n);
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    
    std::cout << "Average time per iteration: " 
              << (duration2 / static_cast<double>(iterations)) 
              << " microseconds" << std::endl;


    auto t3 = std::chrono::high_resolution_clock::now();
    
    for(int iter = 0; iter < iterations; iter++) {
        relu_inplace(A,   n);
    }
    
    auto t4 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    
    std::cout << "Average time per iteration: " 
              << (duration3 / static_cast<double>(iterations)) 
              << " microseconds" << std::endl;


    // Free aligned memory
    delete[] A;
    delete[] B;
    delete[] C;


}