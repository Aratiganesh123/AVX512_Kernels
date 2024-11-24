#include <iostream>
#include <immintrin.h>
#include <random>
#include <chrono>
#include <vector>

//B[B,H,N,D]
//K[B,H,D,N]
void attention_kernel(float* Q, float* K , float* V, int N , int D , int B)
{
    constexpr int simd_width = 8;  // AVX-256
   std::vector<float> scores(N * N);
   float scale = 1.0f / std::sqrt(static_cast<float>(D));
    for(int b = 0 ; b < B ; b++)
    {
        for(int n = 0 ; n < N ; n++)
        {
            for(int i = 0 ; i < N ; i++)
            {
                for(int j = 0 ; j < N ; j++)
                {
                    __m256 sum = __m256_setzero_ps();
                    for(int k = 0 ; k < K ; k+=simd_width)
                    {
                        __m256 query = _mm256_loadu_ps(&Q[b*H*N + h* N + i*D + k]);
                        __m256 key =  _mm256_loadu_ps(&K[b*H*D + h* D + k*N + j]);
                        sum = _mm256_fmadd_ps(query , key , sum);
                    }


                    float sum_arr[8];
                    __mm256_storeu_ps(sum_arr , sum);
                    for (int k = 0; k < 8; k++) {
                       dot += sum[k];
                   }
                    scores[i*N + j] = dot*scale;
                }

                //vectorized softmax
                float max_val = -std::numeric_limits<float>::infinity();
                for(int k = 0 ; k < N ; k++)
                {
                     max_val = std::max(max_val, scores[i * N + j]);
                }

                __m256 max_vec = _mm256_set1_ps(max_val);
                __m256 sum_vec = _mm256_setzero_ps();

                for(int K = 0 ; k < N ; k+=simd_width)
                {
                    __m256 score_vec = _mm256_loadu_ps(&scores[i * N + j]);
                    __m256 exp_input = _mm256_sub_ps(score_vec, max_vec);
                    __m256 exp_result = exp256_ps(exp_input);
                    _mm256_storeu_ps(&scores[i * N + j], exp_result);
                    sum_vec = _mm256_add_ps(sum_vec, exp_result);

                }

                float sum_arr[8];
                _mm256_storeu_ps(sum_arr, sum_vec);
                float sum = 0.0f;
                for (int k = 0; k < 8; k++) {
                    sum += sum_arr[k];
                }
                __m256 inv_sum = _mm256_set1_ps(1.0f / sum);
                
                for (int j = 0; j < N; j += simd_width) {
                    __m256 softmax_vec = _mm256_loadu_ps(&scores[i * N + j]);
                    softmax_vec = _mm256_mul_ps(softmax_vec, inv_sum);
                    _mm256_storeu_ps(&scores[i * N + j], softmax_vec);
                }
                



            }
        }
    }


}


inline __m256 exp256_ps(__m256 x) {
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 exp_c1 = _mm256_set1_ps(0.693147180559945f);
    const __m256 exp_c2 = _mm256_set1_ps(0.240226506959101f);
    const __m256 exp_p0 = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 exp_p1 = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 exp_p2 = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 exp_p3 = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 exp_p4 = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 exp_p5 = _mm256_set1_ps(5.0000001201E-1f);
    
  
    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);
    
 
    __m256 fx = _mm256_mul_ps(x, log2e);
    fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));

    __m256i emm0 = _mm256_cvttps_epi32(fx);
    __m256 tmp = _mm256_cvtepi32_ps(emm0);
    

    __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, _mm256_set1_ps(1.0f));
    fx = _mm256_sub_ps(tmp, mask);
    
    tmp = _mm256_mul_ps(fx, exp_c1);
    __m256 z = _mm256_mul_ps(fx, exp_c2);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);
    z = _mm256_mul_ps(x, x);
    
    __m256 y = exp_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, exp_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, exp_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, exp_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, exp_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, exp_p5);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    
 
    emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(emm0);
    
    y = _mm256_mul_ps(y, pow2n);
    return y;
}
