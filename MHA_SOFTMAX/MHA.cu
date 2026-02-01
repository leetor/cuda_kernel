#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

#define Bc 32
#define Br 32

__global__ void flash_attn_2_kernel(
    const float* Q, const float* K, const float* V, float* output,
    int N, int d_model, int h, int d_k, float scale) {

    int head_idx = blockIdx.y;             
    int row_start = blockIdx.x * Br;   


    int d_k_padded = d_k + 1; 
    extern __shared__ float sram[];
    float* sQ = sram;
    float* sK = &sram[Br * d_k_padded];
    float* sV = &sram[Br* d_k_padded + Bc * d_k];

    float m_prev[Br]; 
    float l_prev[Br]; 
    float O_row[Br][128]; 

    int tid = threadIdx.x;

    if (tid < Br && (row_start + tid) < N) {
        m_prev[tid] = -1e20f;
        l_prev[tid] = 0.0f;
        for (int d = 0; d < d_k; d++) O_row[tid][d] = 0.0f;

        for (int d = 0; d < d_k; d++) {
            sQ[tid * d_k_padded + d] = Q[(row_start + tid) * d_model + head_idx * d_k + d];
        }
    }
    __syncthreads();

    for (int j = 0; j < (N + Bc - 1) / Bc; j++) {
        int col_start = j * Bc;
        if (tid < Bc) {
            for (int d = 0; d < d_k; d++) {
                int k_row = col_start + tid;
                if (k_row < N) {
                    sK[tid * d_k + d] = K[k_row * d_model + head_idx * d_k + d];
                    sV[tid * d_k + d] = V[k_row * d_model + head_idx * d_k + d];
                } else {
                    sK[tid * d_k + d] = 0.0f;
                    sV[tid * d_k + d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (tid < Br && (row_start + tid) < N) {
            float m_curr = -1e20f;
            float S[Bc];

            for (int k = 0; k < Bc; k++) {
                if (col_start + k >= N) continue;
                float score = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    score += sQ[tid * d_k_padded + d] * sK[k * d_k + d];
                }
                score *= scale;
                S[k] = score;
                if (score > m_curr) m_curr = score;
            }

            float m_new = fmaxf(m_prev[tid], m_curr);
            float exp_prev = expf(m_prev[tid] - m_new);
            
            float l_curr = 0.0f;
            for (int k = 0; k < Bc; k++) {
                if (col_start + k >= N) continue;
                S[k] = expf(S[k] - m_new);
                l_curr += S[k];
            }

            float l_new = l_prev[tid] * exp_prev + l_curr;

            float rescale_factor = exp_prev; 
            for (int d = 0; d < d_k; d++) {
                float v_sum = 0.0f;
                for (int k = 0; k < Bc; k++) {
                    if (col_start + k >= N) continue;
                    v_sum += S[k] * sV[k * d_k + d];
                }
                O_row[tid][d] = O_row[tid][d] * rescale_factor + v_sum;
            }

            m_prev[tid] = m_new;
            l_prev[tid] = l_new;
        }
        __syncthreads();
    }

    if (tid < Br && (row_start + tid) < N) {
        for (int d = 0; d < d_k; d++) {
            output[(row_start + tid) * d_model + head_idx * d_k + d] = O_row[tid][d] / l_prev[tid];
        }
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
    int d_k = d_model / h;
    float scale = 1.0f / sqrtf((float)d_k);


    dim3 grid((N + Br - 1) / Br, h);
    dim3 block(std::max(Br, Bc)); 

    size_t shared_mem_size = (Br * (d_k + 1) + 2 * Bc * d_k) * sizeof(float);

    flash_attn_2_kernel<<<grid, block, shared_mem_size>>>(
        Q, K, V, output, N, d_model, h, d_k, scale
    );

    cudaDeviceSynchronize();
}