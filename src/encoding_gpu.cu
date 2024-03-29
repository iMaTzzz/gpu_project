#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>
#include <assert.h>

/* Constantes utilisées dans l'algorithme de Loeffler pour appliquer la DCT*/
#define VALUE_0_765366865 0.765366865        // sqrt(2) * (sin(3pi/8) - cos(3pi/8))
#define VALUE_MINUS_1_847759065 -1.847759065 // -sqrt(2) * (cos(3pi/8) + sin(3pi/8))
#define VALUE_MINUS_1_175875602 -1.175875602 // -(cos(pi/16) + sin(pi/16))
#define VALUE_0_541196100 0.541196100        // sqrt(2) * cos(3pi/8)
#define VALUE_MINUS_0_275899379 -0.275899379 // sin(3pi/16) - cos(3pi/16)
#define VALUE_MINUS_1_387039845 -1.387039845 // -(cos(3pi/16) + sin(3pi/16)
#define VALUE_1_414213562 1.414213562        // sqrt(2)
#define VALUE_0_831469612 0.831469612        // cos(3pi/16)
#define VALUE_0_980785280 0.980785280        // cos(pi/16)
#define VALUE_MINUS_0_785694958 -0.785694958 // sin(pi/16) - cos(pi/16)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__constant__ uint8_t cuda_matrix_zig_zag[8][8] = {
    {0, 1, 5, 6, 14, 15, 27, 28},
    {2, 4, 7, 13, 16, 26, 29, 42},
    {3, 8, 12, 17, 25, 30, 41, 43},
    {9, 11, 18, 24, 31, 40, 44, 53},
    {10, 19, 23, 32, 39, 45, 52, 54},
    {20, 22, 33, 38, 46, 51, 55, 60},
    {21, 34, 37, 47, 50, 56, 59, 61},
    {35, 36, 48, 49, 57, 58, 62, 63}};

__constant__ uint8_t cuda_quantification_table_Y[] = {
    0x05, 0x03, 0x03, 0x05, 0x07, 0x0c, 0x0f, 0x12,
    0x04, 0x04, 0x04, 0x06, 0x08, 0x11, 0x12, 0x11,
    0x04, 0x04, 0x05, 0x07, 0x0c, 0x11, 0x15, 0x11,
    0x04, 0x05, 0x07, 0x09, 0x0f, 0x1a, 0x18, 0x13,
    0x05, 0x07, 0x0b, 0x11, 0x14, 0x21, 0x1f, 0x17,
    0x07, 0x0b, 0x11, 0x13, 0x18, 0x1f, 0x22, 0x1c,
    0x0f, 0x13, 0x17, 0x1a, 0x1f, 0x24, 0x24, 0x1e,
    0x16, 0x1c, 0x1d, 0x1d, 0x22, 0x1e, 0x1f, 0x1e};

__constant__ uint8_t cuda_quantification_table_CbCr[] = {
    0x05, 0x05, 0x07, 0x0e, 0x1e, 0x1e, 0x1e, 0x1e,
    0x05, 0x06, 0x08, 0x14, 0x1e, 0x1e, 0x1e, 0x1e,
    0x07, 0x08, 0x11, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
    0x0e, 0x14, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
    0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
    0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
    0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
    0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e};

__global__ void encoding_gpu(int16_t *d_mcus_array, uint8_t luminance)
{
    /******** DCT + zig zag ********/
    // temporary data structure used by all threads within a block
    __shared__ int32_t tmp_shared_block[8][8];
    __shared__ int16_t output_shared_block[64];
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t block_offset = blockIdx.x * blockDim.x * blockDim.y;

    // We only compute 1D DCT on 8 threads per block, the remaining 56 threads remain idle
    int32_t a0, a1, a2, a3, a4, a5, a6, a7;
    int32_t b0, b1, b2, b3, b4, b5, b6, b7;
    int32_t c4, c5, c6, c7;
    int32_t tmp0, tmp1, tmp2;
    if (tx == 0 && ty < 8) {
        /* Row-wise DCT */
        // We add block_offset to move to the correct block and the threadIdx.y corresponds to one of the 8 rows. 
        // Each thread is assigned to a row in a block which corresponds to their id
        uint32_t offset = block_offset + ty * 8;
        // Stage 1 contains 8 adds (+ 4 offsets)
        a0 = (int32_t)(d_mcus_array[offset + 0] + d_mcus_array[offset + 7] - 256);
        a1 = (int32_t)(d_mcus_array[offset + 1] + d_mcus_array[offset + 6] - 256);
        a2 = (int32_t)(d_mcus_array[offset + 2] + d_mcus_array[offset + 5] - 256);
        a3 = (int32_t)(d_mcus_array[offset + 3] + d_mcus_array[offset + 4] - 256);
        a4 = (int32_t)(d_mcus_array[offset + 3] - d_mcus_array[offset + 4]);
        a5 = (int32_t)(d_mcus_array[offset + 2] - d_mcus_array[offset + 5]);
        a6 = (int32_t)(d_mcus_array[offset + 1] - d_mcus_array[offset + 6]);
        a7 = (int32_t)(d_mcus_array[offset + 0] - d_mcus_array[offset + 7]);

        // Stage 2 contains 6 mult + 10 adds
        b0 = a0 + a3;
        b1 = a1 + a2;
        b2 = a1 - a2;
        b3 = a0 - a3;
        tmp0 = VALUE_0_831469612 * (a4 + a7);
        b4 = VALUE_MINUS_0_275899379 * a7 + tmp0;
        b7 = VALUE_MINUS_1_387039845 * a4 + tmp0;
        tmp1 = VALUE_0_980785280 * (a5 + a6);
        b5 = VALUE_MINUS_0_785694958 * a6 + tmp1;
        b6 = VALUE_MINUS_1_175875602 * a5 + tmp1;

        // Stage 3 contains 3 mult + 9 adds
        tmp2 = VALUE_0_541196100 * (b2 + b3);
        tmp_shared_block[ty][0] = b0 + b1;
        tmp_shared_block[ty][2] = VALUE_0_765366865 * b3 + tmp2;
        tmp_shared_block[ty][4] = b0 - b1;
        tmp_shared_block[ty][6] = VALUE_MINUS_1_847759065 * b2 + tmp2;
        c4 = b4 + b6;
        c5 = b7 - b5;
        c6 = b4 - b6;
        c7 = b5 + b7;

        // Stage 4 contains 2 mults + 2 adds
        tmp_shared_block[ty][1] = c4 + c7;
        tmp_shared_block[ty][3] = c5 * VALUE_1_414213562;
        tmp_shared_block[ty][5] = c6 * VALUE_1_414213562;
        tmp_shared_block[ty][7] = c7 - c4;
    }

    // synchronize to ensure all threads have completed the row-wise DCT before doing the column-wise DCT
    __syncthreads();

    if (tx == 0 && ty < 8) {
        /* Column-wise DCT */
        // ty == column => Each thread is assigned to a column in a block which corresponds to their id
        // Stage 1 contains 8 adds
        a0 = tmp_shared_block[0][ty] + tmp_shared_block[7][ty];
        a1 = tmp_shared_block[1][ty] + tmp_shared_block[6][ty];
        a2 = tmp_shared_block[2][ty] + tmp_shared_block[5][ty];
        a3 = tmp_shared_block[3][ty] + tmp_shared_block[4][ty];
        a4 = tmp_shared_block[3][ty] - tmp_shared_block[4][ty];
        a5 = tmp_shared_block[2][ty] - tmp_shared_block[5][ty];
        a6 = tmp_shared_block[1][ty] - tmp_shared_block[6][ty];
        a7 = tmp_shared_block[0][ty] - tmp_shared_block[7][ty];

        // Stage 2 contains 6 mult + 10 adds
        b0 = a0 + a3;
        b1 = a1 + a2;
        b2 = a1 - a2;
        b3 = a0 - a3;
        tmp0 = VALUE_0_831469612 * (a4 + a7);
        b4 = VALUE_MINUS_0_275899379 * a7 + tmp0;
        b7 = VALUE_MINUS_1_387039845 * a4 + tmp0;
        tmp1 = VALUE_0_980785280 * (a5 + a6);
        b5 = VALUE_MINUS_0_785694958 * a6 + tmp1;
        b6 = VALUE_MINUS_1_175875602 * a5 + tmp1;

        // Stage 3 contains 3 mult + 9 adds
        tmp2 = VALUE_0_541196100 * (b2 + b3);
        output_shared_block[cuda_matrix_zig_zag[0][ty]] = (int16_t)((b0 + b1) >> 3);
        output_shared_block[cuda_matrix_zig_zag[2][ty]] = (int16_t)(((int32_t)(VALUE_0_765366865 * b3 + tmp2)) >> 3);
        output_shared_block[cuda_matrix_zig_zag[4][ty]] = (int16_t)((b0 - b1) >> 3);
        output_shared_block[cuda_matrix_zig_zag[6][ty]] = (int16_t)(((int32_t)(VALUE_MINUS_1_847759065 * b2 + tmp2)) >> 3);
        c4 = b4 + b6;
        c5 = b7 - b5;
        c6 = b4 - b6;
        c7 = b5 + b7;

        // Stage 4 contains 2 mults + 2 adds + 8 normalized shifts (multiply by 8)
        output_shared_block[cuda_matrix_zig_zag[1][ty]] = (int16_t)((c4 + c7) >> 3);
        output_shared_block[cuda_matrix_zig_zag[3][ty]] = (int16_t)(((int32_t)(c5 * VALUE_1_414213562)) >> 3);
        output_shared_block[cuda_matrix_zig_zag[5][ty]] = (int16_t)(((int32_t)(c6 * VALUE_1_414213562)) >> 3);
        output_shared_block[cuda_matrix_zig_zag[7][ty]] = (int16_t)((c7 - c4) >> 3);
    }
    // Wait for every threads before we can move onto the quantify step since 56 threads were idle
    __syncthreads();


    /******** quantify ********/
    uint32_t thread_id_in_block = threadIdx.x * 8 + threadIdx.y;

    if (tx < 8 && ty < 8)
    {
        if (luminance) {
            output_shared_block[thread_id_in_block] /= cuda_quantification_table_Y[thread_id_in_block];
        }
        else {
            output_shared_block[thread_id_in_block] /= cuda_quantification_table_CbCr[thread_id_in_block];
        }
    }
    // Copy the output located in the shared block into the mcus_line_array
    uint32_t index_in_mcus_array = blockIdx.x * blockDim.x * blockDim.y + thread_id_in_block;
    d_mcus_array[index_in_mcus_array] = output_shared_block[thread_id_in_block];
}

extern "C"
void encoding(int16_t *h_mcus_array, int16_t *d_mcus_array, uint32_t nb_mcus, size_t array_size, bool luminance)
{
    // Copy data from the host to the device (CPU -> GPU)
    gpuErrchk(cudaMemcpy(d_mcus_array, h_mcus_array, array_size, cudaMemcpyHostToDevice));

    const dim3 block_size(8, 8);
    const dim3 grid_size(nb_mcus);
    // 64 threads per block, 1 block per MCUs
    // Starting kernel
    encoding_gpu<<<grid_size, block_size>>>(d_mcus_array, (uint8_t)luminance);
    gpuErrchk(cudaPeekAtLastError());

    // Copy data from the device to host (GPU -> CPU)
    // Acts a synchronization making sure all threads are done
    gpuErrchk(cudaMemcpy(h_mcus_array, d_mcus_array, array_size, cudaMemcpyDeviceToHost));
}
