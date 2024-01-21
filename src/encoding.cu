#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>
#include <assert.h>

/* Constantes utilisées dans les deux versions des algorithmes de Loeffler */
#define VALUE_0_765366865 0.765366865
#define VALUE_1_847759065 1.847759065
#define VALUE_1_175875602 1.175875602
#define VALUE_0_390180644 0.390180644
#define VALUE_1_961570560 1.961570560
#define VALUE_0_899976223 0.899976223
#define VALUE_1_501321110 1.501321110
#define VALUE_0_298631336 0.298631336
#define VALUE_2_562915447 2.562915447
#define VALUE_3_072711026 3.072711026
#define VALUE_2_053119869 2.053119869
#define VALUE_0_707106781 0.707106781
#define VALUE_0_382683433 0.382683433
#define VALUE_0_541196100 0.541196100
#define VALUE_1_306562965 1.306562965
#define VALUE_0_275899379 0.275899379
#define VALUE_1_387039845 1.387039845
#define VALUE_1_414213562 1.414213562
#define VALUE_0_555570233 0.555570233
#define VALUE_0_831469612 0.831469612
#define VALUE_0_195090322 0.195090322
#define VALUE_0_980785280 0.980785280
#define VALUE_1_662939225 1.662939225

__constant__ uint8_t cuda_matrix_zig_zag[8][8] = {
  {0, 1, 5, 6, 14, 15, 27, 28},
  {2, 4, 7, 13, 16, 26, 29, 42},
  {3, 8, 12, 17, 25, 30, 41, 43},
  {9, 11, 18, 24, 31, 40, 44, 53},
  {10, 19, 23, 32, 39, 45, 52, 54},
  {20, 22, 33, 38, 46, 51, 55, 60},
  {21, 34, 37, 47, 50, 56, 59, 61},
  {35, 36, 48, 49, 57, 58, 62, 63}
};

__constant__ uint8_t cuda_quantification_table_Y[] = {
  0x05, 0x03, 0x03, 0x05, 0x07, 0x0c, 0x0f, 0x12,
  0x04, 0x04, 0x04, 0x06, 0x08, 0x11, 0x12, 0x11,
  0x04, 0x04, 0x05, 0x07, 0x0c, 0x11, 0x15, 0x11,
  0x04, 0x05, 0x07, 0x09, 0x0f, 0x1a, 0x18, 0x13,
  0x05, 0x07, 0x0b, 0x11, 0x14, 0x21, 0x1f, 0x17,
  0x07, 0x0b, 0x11, 0x13, 0x18, 0x1f, 0x22, 0x1c,
  0x0f, 0x13, 0x17, 0x1a, 0x1f, 0x24, 0x24, 0x1e,
  0x16, 0x1c, 0x1d, 0x1d, 0x22, 0x1e, 0x1f, 0x1e
};

__constant__ uint8_t cuda_quantification_table_CbCr[] = {
  0x05, 0x05, 0x07, 0x0e, 0x1e, 0x1e, 0x1e, 0x1e,
  0x05, 0x06, 0x08, 0x14, 0x1e, 0x1e, 0x1e, 0x1e,
  0x07, 0x08, 0x11, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
  0x0e, 0x14, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
  0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
  0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
  0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
  0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e
};

__global__ void encoding_gpu(int16_t *mcus_line_array, uint32_t nb_mcu_line, uint8_t luminance)
{
  /******** DCT ********/
  // temporary data structure used by all threads within a block
  __shared__ int32_t shared_block[8][8];

  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.x * blockDim.y + threadIdx.y;

  // check if within bounds
  if (x < (nb_mcu_line - 1) * 8 + 8 && y < (nb_mcu_line - 1) * 8 + 8) {
    int32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
    const uint8_t mcus_line_array_width = 8;

    /***** perform row-wise DCT computation *****/
    tmp0 = (int32_t) (mcus_line_array[y * mcus_line_array_width + 0] + mcus_line_array[y * mcus_line_array_width + 7] - 256);
    tmp1 = (int32_t) (mcus_line_array[y * mcus_line_array_width + 1] + mcus_line_array[y * mcus_line_array_width + 6] - 256);
    tmp2 = (int32_t) (mcus_line_array[y * mcus_line_array_width + 2] + mcus_line_array[y * mcus_line_array_width + 5] - 256);
    tmp3 = (int32_t) (mcus_line_array[y * mcus_line_array_width + 3] + mcus_line_array[y * mcus_line_array_width + 4] - 256);

    tmp4 = tmp0 + tmp3;
    tmp5 = tmp1 + tmp2;
    tmp6 = tmp0 - tmp3;
    tmp7 = tmp1 - tmp2;

    shared_block[y][0] = tmp4 + tmp5;
    shared_block[y][4] = tmp4 - tmp5;

    tmp8 = (tmp6 + tmp7) * VALUE_0_541196100;

    shared_block[y][2] = tmp8 + tmp6 * VALUE_0_765366865;
    shared_block[y][6] = tmp8 - tmp7 * VALUE_1_847759065;

    tmp0 = (int32_t) (mcus_line_array[y * mcus_line_array_width + 0] - mcus_line_array[y * mcus_line_array_width + 7]);
    tmp1 = (int32_t) (mcus_line_array[y * mcus_line_array_width + 1] - mcus_line_array[y * mcus_line_array_width + 6]);
    tmp2 = (int32_t) (mcus_line_array[y * mcus_line_array_width + 2] - mcus_line_array[y * mcus_line_array_width + 5]);
    tmp3 = (int32_t) (mcus_line_array[y * mcus_line_array_width + 3] - mcus_line_array[y * mcus_line_array_width + 4]);

    tmp6 = tmp0 + tmp2;
    tmp7 = tmp1 + tmp3;

    tmp8 = (tmp6 + tmp7) * VALUE_1_175875602;
    tmp6 = tmp6 * (- VALUE_0_390180644) + tmp8;
    tmp7 = tmp7 * (- VALUE_1_961570560) + tmp8;

    tmp8 = (tmp0 + tmp3) * (- VALUE_0_899976223);
    tmp0 = tmp0 * VALUE_1_501321110 + tmp8 + tmp6;
    tmp3 = tmp3 * VALUE_0_298631336 + tmp8 + tmp7;

    tmp8 = (tmp1 + tmp2) * (- VALUE_2_562915447);
    tmp1 = tmp1 * VALUE_3_072711026 + tmp8 + tmp7;
    tmp2 = tmp2 * VALUE_2_053119869 + tmp8 + tmp6;

    shared_block[y][1] = tmp0;
    shared_block[y][3] = tmp1;
    shared_block[y][5] = tmp2;
    shared_block[y][7] = tmp3;

    // synchronize to ensure all threads have completed the row-wise DCT before doing the column-wise DCT
    __syncthreads();

    /***** perform column-wise DCT using the data in shared_block *****/
    tmp0 = shared_block[0][x] + shared_block[7][x];
    tmp1 = shared_block[1][x] + shared_block[6][x];
    tmp2 = shared_block[2][x] + shared_block[5][x];
    tmp3 = shared_block[3][x] + shared_block[4][x];

    tmp4 = tmp0 + tmp3;
    tmp5 = tmp1 + tmp2;
    tmp6 = tmp0 - tmp3;
    tmp7 = tmp1 - tmp2;

    tmp0 = shared_block[0][x] - shared_block[7][x];
    tmp1 = shared_block[1][x] - shared_block[6][x];
    tmp2 = shared_block[2][x] - shared_block[5][x];
    tmp3 = shared_block[3][x] - shared_block[4][x];

    mcus_line_array[cuda_matrix_zig_zag[0][x]] = ((int16_t) (tmp4 + tmp5) >> 3);
    mcus_line_array[cuda_matrix_zig_zag[4][x]] = ((int16_t) (tmp4 - tmp5) >> 3);

    tmp8 = (tmp6 + tmp7) * VALUE_0_541196100;

    mcus_line_array[cuda_matrix_zig_zag[2][x]] = ((int16_t) (tmp8 + tmp6 * VALUE_0_765366865) >> 3);
    mcus_line_array[cuda_matrix_zig_zag[6][x]] = ((int16_t) (tmp8 - tmp7 * VALUE_1_847759065) >> 3);

    tmp6 = tmp0 + tmp2;
    tmp7 = tmp1 + tmp3;

    tmp8 = (tmp6 + tmp7) * VALUE_1_175875602;
    tmp6 = tmp6 * (- VALUE_0_390180644) + tmp8;
    tmp7 = tmp7 * (- VALUE_1_961570560) + tmp8;

    tmp8 = (tmp0 + tmp3) * (- VALUE_0_899976223);
    tmp0 = tmp0 * VALUE_1_501321110 + tmp8 + tmp6;
    tmp3 = tmp3 * VALUE_0_298631336 + tmp8 + tmp7;

    tmp8 = (tmp1 + tmp2) * (- VALUE_2_562915447);
    tmp1 = tmp1 * VALUE_3_072711026 + tmp8 + tmp7;
    tmp2 = tmp2 * VALUE_2_053119869 + tmp8 + tmp6;

    mcus_line_array[cuda_matrix_zig_zag[1][x]] = (int16_t) (tmp0 >> 3);
    mcus_line_array[cuda_matrix_zig_zag[3][x]] = (int16_t) (tmp1 >> 3);
    mcus_line_array[cuda_matrix_zig_zag[5][x]] = (int16_t) (tmp2 >> 3);
    mcus_line_array[cuda_matrix_zig_zag[7][x]] = (int16_t) (tmp3 >> 3);
  }

  /******** zigzag ********/

  /******** quantify ********/
  uint32_t index_in_mcu_line_array = blockIdx.x * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
  uint32_t thread_id_in_block = threadIdx.y * blockDim.x + threadIdx.x;

  if (index_in_mcu_line_array < (nb_mcu_line - 1) * 64 && thread_id_in_block < 64) {
    if (luminance) mcus_line_array[index_in_mcu_array] /= cuda_quantification_table_Y[thread_id_in_block];
    else mcus_line_array[index_in_mcu_array] /= cuda_quantification_table_CbCr[thread_id_in_block];
  }
}

extern "C"
void encoding(int16_t *h_mcus_line_array, uint32_t nb_mcu_line, bool luminance)
{
  // Give size to allocate on GPU
  const int array_size = nb_mcu_line * 64 * sizeof(int16_t);

  // Allocate memory on the device
  int16_t *d_mcus_line_array;
  cudaMalloc(&d_mcus_line_array, array_size);

  // Copy data from the host to the device (CPU -> GPU)
  cudaMemcpy(d_mcus_line_array, h_mcus_line_array, array_size, cudaMemcpyHostToDevice);

  const dim3 block_size(8, 8);
  const dim3 grid_size(nb_mcu_line);
  encoding_gpu<<<grid_size, block_size>>>(d_mcus_line_array, nb_mcu_line, (uint8_t) luminance);

  // Copy data from the device to host (GPU -> CPU)
  cudaMemcpy(h_mcus_line_array, d_mcus_line_array, array_size, cudaMemcpyDeviceToHost);

  cudaFree(d_mcus_line_array);
}

