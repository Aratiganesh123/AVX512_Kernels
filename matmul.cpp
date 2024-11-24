#include<iostream>
#include<random>
#include <chrono>

using namespace std;

//Too much cache pollution and the matrix B is not row major. The simplest optimization would be to move up the the C up to remove the cache pollution
void naiveImplementation(float* A , float* B , float* C , int M , int N , int K)
{
    for(int i = 0 ; i < M ; i++)
    {
        for(int j = 0 ; j < N ; j++)
        {
            for(int k = 0 ; k < K ; k++)
            {
                C[i* N + j] += A[i*K + k] * B[k*N + j];
            }
        }
    }
}

//Can help with Spatial locality - especially when the matrix size grows
//Simple Loop Reordeing to make it column wise 
void loopReorder(float* A , float* B , float* C , int M , int N , int K)
{
    for(int i = 0 ; i < M ; i++)
    {
         for(int k = 0 ; k < K ; k++)
        {
            for(int j = 0 ; j < N ; j++)
            {
                C[i* N + j] += A[i*K + k] * B[k*N + j];
            }
        }
    }
}

//More temporal locality - reuse of repeated similar elements
//It is not cache agnostic
//3 levels of cache would need diff kinds of optimization
template<int T>
void naiveTiled(float* const A , float* const B , float* const C ,  const int M , const int N , const int K )
{
    for(int m = 0 ; m <  M / T ; m+= T)
    {
        for(int n = 0 ; n < N/T ; n+=T)
        {
            for(int k = 0 ; k < K/T ; k+=T)
            {
               for(int mt = m ; mt < m + T && mt < M ; mt++)
               {
                    for(int nt = n ; nt < n + T  && nt < N ; nt++)
                    {
                        float sum = C[mt * N + nt];
                        for(int kt = k ; kt < k + T && kt < K ; kt++)
                        {
                                sum += A[mt*K + kt] * B[kt*K + nt];
                        }

                        C[mt * N + nt] = sum;
                    }
               }
            }
        }
    }
}



int main()
{
    int M = 1 << 10;
    int N = 1 << 10;
    int K = 1 << 10;
    float* A = new float[M*K];
    float* B = new float[N*K];
    float* C = new float[M*N];


    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(1.0f,100.0f);

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {  // K for matrix A
            A[i * K + j] = dist(gen);
        }
    }

    for(int i = 0; i <  K; i++) {
        for(int j = 0; j < N; j++) {  // K for matrix A
            B[i * K + j] = dist(gen);
        }
    }

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            C[i * N + j] = 0;
        }
    }
    //Naive Implemntation 
    auto start = std::chrono::high_resolution_clock::now();
    naiveImplementation(A , B , C , M , N , K);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Naive Implementation Time : " << duration.count() << " microseconds" << endl;


    //Loop Reorder
    start = std::chrono::high_resolution_clock::now();
    loopReorder(A , B , C , M , N , K);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Loop Reorder Implementation Time : " << duration.count() << " microseconds" << endl;



    //Tiled Implementation
    start = std::chrono::high_resolution_clock::now();
    naiveTiled<32>(A , B , C , M , N , K);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Naive Tiled Implementation Time : " << duration.count() << " microseconds" << endl;

    delete[] A;
    delete[] B;
    delete[] C;



}