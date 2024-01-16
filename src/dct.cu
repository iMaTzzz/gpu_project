#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "zigzag.h"
#include <iostream>

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

/*
Algorithme de Loeffler : C'est un algorithme permettant de calculer la 1D-DCT sur 8 pixels.
Ainsi, pour calculer la 2D-DCT pour un bloc 8x8, il suffit d'appliquer l'algorithme de Loeffler
à chaque ligne puis à chaque colonne et on divise chaque coefficient par 8. On obtient donc les
coefficients voulus.
*/

__global__ void dct_kernel(uint8_t *bloc_spatiale, uint16_t *output_mcu_array) {
  // temporary data structure used by all threads within a block
  __shared__ int32_t shared_block[8][8];
  __shared__ uint8_t bloc_spatiale_width = 8;

  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  // check if within bounds
  if (x < 8 && y < 8) {
    int32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;

    /***** perform row-wise DCT computation *****/
    tmp0 = (int32_t) (bloc_spatiale[y * bloc_spatiale_width + 0] + bloc_spatiale[y * bloc_spatiale_width + 7] - 256);
    tmp1 = (int32_t) (bloc_spatiale[y * bloc_spatiale_width + 1] + bloc_spatiale[y * bloc_spatiale_width + 6] - 256);
    tmp2 = (int32_t) (bloc_spatiale[y * bloc_spatiale_width + 2] + bloc_spatiale[y * bloc_spatiale_width + 5] - 256);
    tmp3 = (int32_t) (bloc_spatiale[y * bloc_spatiale_width + 3] + bloc_spatiale[y * bloc_spatiale_width + 4] - 256);

    tmp4 = tmp0 + tmp3;
    tmp5 = tmp1 + tmp2;
    tmp6 = tmp0 - tmp3;
    tmp7 = tmp1 - tmp2;

    shared_block[y][0] = tmp4 + tmp5;
    shared_block[y][4] = tmp4 - tmp5;

    tmp8 = (tmp6 + tmp7) * VALUE_0_541196100;

    shared_block[y][2] = tmp8 + tmp6 * VALUE_0_765366865;
    shared_block[y][6] = tmp8 - tmp7 * VALUE_1_847759065;

    tmp0 = (int32_t) (bloc_spatiale[y * bloc_spatiale_width + 0] - bloc_spatiale[y * bloc_spatiale_width + 7]);
    tmp1 = (int32_t) (bloc_spatiale[y * bloc_spatiale_width + 1] - bloc_spatiale[y * bloc_spatiale_width + 6]);
    tmp2 = (int32_t) (bloc_spatiale[y * bloc_spatiale_width + 2] - bloc_spatiale[y * bloc_spatiale_width + 5]);
    tmp3 = (int32_t) (bloc_spatiale[y * bloc_spatiale_width + 3] - bloc_spatiale[y * bloc_spatiale_width + 4]);

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

    output_mcu_array[cuda_matrix_zig_zag[0][x]] = ((int16_t) (tmp4 + tmp5) >> 3);
    output_mcu_array[cuda_matrix_zig_zag[4][x]] = ((int16_t) (tmp4 - tmp5) >> 3);

    tmp8 = (tmp6 + tmp7) * VALUE_0_541196100;

    output_mcu_array[cuda_matrix_zig_zag[2][x]] = ((int16_t) (tmp8 + tmp6 * VALUE_0_765366865) >> 3);
    output_mcu_array[cuda_matrix_zig_zag[6][x]] = ((int16_t) (tmp8 - tmp7 * VALUE_1_847759065) >> 3);

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

    output_mcu_array[cuda_matrix_zig_zag[1][x]] = (int16_t) (tmp0 >> 3);
    output_mcu_array[cuda_matrix_zig_zag[3][x]] = (int16_t) (tmp1 >> 3);
    output_mcu_array[cuda_matrix_zig_zag[5][x]] = (int16_t) (tmp2 >> 3);
    output_mcu_array[cuda_matrix_zig_zag[7][x]] = (int16_t) (tmp3 >> 3);
  }
}

void gpu_dct_loeffler(uint8_t **bloc_spatiale, int16_t *h_mcu_array)
{
  const uint8_t n_rows = 8;
  const uint8_t n_cols = 8;
  const uint8_t array_size = n_rows * n_cols;

  // flatten 2D bloc_spatiale to 1D array for better performances in gpu
  const uint8_t flattened_bloc_spatiale[array_size];
  for (uint8_t row = 0; row < n_rows; row++)
    memcpy(flattened_bloc_spatiale + row * n_cols, bloc_spatiale[row], n_cols * sizeof(uint8_t));

  // copy host flattened_bloc_spatiale to device
  uint8_t *d_bloc_spatiale;
  cudaMalloc(&d_bloc_spatiale, array_size * sizeof(uint8_t));
  cudaMemcpy(d_bloc_spatiale, flattened_bloc_spatiale, array_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

  // copy host mcu_array to device
  int16_t *d_mcu_array;
  cudaMalloc(&d_mcu_array, array_size * sizeof(int16_t));
  cudaMemcpy(d_mcu_array, h_mcu_array, array_size * sizeof(int16_t), cudaMemcpyHostToDevice);

  // call kernel
  const dim3 block_size(8, 8);
  // TODO: not sure of the grid_size (maybe just 1 bloc in grid?)
  const dim3 grid_size((n_cols + block_size.x - 1) / block_size.x, (n_rows + block_size.y - 1) / block_size.y);
  dct_kernel<<<grid_size, block_size>>>(d_bloc_spatiale, d_mcu_array);

  // Copy result of gpu computation back on host
  // cudaMemcpy will wait for kernel completion (acts as synchronization barrier)
  cudaMemcpy(h_mcu_array, d_mcu_array, array_size, cudaMemcpyDeviceToHost);

  // free allocated gpu memory
  cudaFree(d_bloc_spatiale);
  cudaFree(d_mcu_array);
}


/* Fonction qui applique la transformée 2D grâce à l'algorithme de Loeffler non optimisé mais précis */
void cpu_dct_loeffler(uint8_t **bloc_spatiale, int16_t *mcu_array)
{
  int32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
  int32_t tab_tmp[8][8];
  for (uint8_t row = 0; row < 8; row++) {
    tmp0 = (int32_t) (bloc_spatiale[row][0] + bloc_spatiale[row][7] - 256);
    tmp1 = (int32_t) (bloc_spatiale[row][1] + bloc_spatiale[row][6] - 256);
    tmp2 = (int32_t) (bloc_spatiale[row][2] + bloc_spatiale[row][5] - 256);
    tmp3 = (int32_t) (bloc_spatiale[row][3] + bloc_spatiale[row][4] - 256);

    tmp4 = tmp0 + tmp3;
    tmp5 = tmp1 + tmp2;
    tmp6 = tmp0 - tmp3;
    tmp7 = tmp1 - tmp2;

    tab_tmp[row][0] = tmp4 + tmp5;
    tab_tmp[row][4] = tmp4 - tmp5;

    tmp8 = (tmp6 + tmp7) * VALUE_0_541196100;

    tab_tmp[row][2] = tmp8 + tmp6 * VALUE_0_765366865;
    tab_tmp[row][6] = tmp8 - tmp7 * VALUE_1_847759065;

    tmp0 = (int32_t) (bloc_spatiale[row][0] - bloc_spatiale[row][7]);
    tmp1 = (int32_t) (bloc_spatiale[row][1] - bloc_spatiale[row][6]);
    tmp2 = (int32_t) (bloc_spatiale[row][2] - bloc_spatiale[row][5]);
    tmp3 = (int32_t) (bloc_spatiale[row][3] - bloc_spatiale[row][4]);

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

    tab_tmp[row][1] = tmp0;
    tab_tmp[row][3] = tmp1;
    tab_tmp[row][5] = tmp2;
    tab_tmp[row][7] = tmp3;
  }
  for (uint8_t column = 0; column < 8; column++) {
    tmp0 = tab_tmp[0][column] + tab_tmp[7][column];
    tmp1 = tab_tmp[1][column] + tab_tmp[6][column];
    tmp2 = tab_tmp[2][column] + tab_tmp[5][column];
    tmp3 = tab_tmp[3][column] + tab_tmp[4][column];

    tmp4 = tmp0 + tmp3;
    tmp5 = tmp1 + tmp2;
    tmp6 = tmp0 - tmp3;
    tmp7 = tmp1 - tmp2;

    tmp0 = tab_tmp[0][column] - tab_tmp[7][column];
    tmp1 = tab_tmp[1][column] - tab_tmp[6][column];
    tmp2 = tab_tmp[2][column] - tab_tmp[5][column];
    tmp3 = tab_tmp[3][column] - tab_tmp[4][column];

    mcu_array[matrix_zig_zag[0][column]] = ((int16_t) (tmp4 + tmp5) >> 3); 
    mcu_array[matrix_zig_zag[4][column]] = ((int16_t) (tmp4 - tmp5) >> 3); 

    tmp8 = (tmp6 + tmp7) * VALUE_0_541196100;
    
    mcu_array[matrix_zig_zag[2][column]] = ((int16_t) (tmp8 + tmp6 * VALUE_0_765366865) >> 3);
    mcu_array[matrix_zig_zag[6][column]] = ((int16_t) (tmp8 - tmp7 * VALUE_1_847759065) >> 3);

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

    mcu_array[matrix_zig_zag[1][column]] = (int16_t) (tmp0 >> 3);
    mcu_array[matrix_zig_zag[3][column]] = (int16_t) (tmp1 >> 3);
    mcu_array[matrix_zig_zag[5][column]] = (int16_t) (tmp2 >> 3);
    mcu_array[matrix_zig_zag[7][column]] = (int16_t) (tmp3 >> 3);
  }
}

void verify_result(int16_t mcu_array[64], int16_t mcu_array_copy[64])
{
  for (uint8_t i = 0; i < 64; i++)
    assert(mcu_array[i] == mcu_array_copy[i]);
}


/* Fonction qui applique la transformée 2D grâce à l'algorithme de Loeffler non optimisé mais précis */
extern "C"
void dct_loeffler(uint8_t **bloc_spatiale, int16_t *mcu_array)
{
  std::cout << "--- beginning of dct_loeffler --- " << std::endl;

  int16_t mcu_array_copy[64];
  for (uint8_t i = 0; i < 64; i++) {
    mcu_array_copy[i] = mcu_array[i];
  }
  std::cout << "--- after copy of mcu_array --- " << std::endl;

  cpu_dct_loeffler(bloc_spatiale, mcu_array);
  std::cout << "--- after cpu_dct_loeffler --- " << std::endl;

  gpu_dct_loeffler(bloc_spatiale, mcu_array_copy);
  std::cout << "--- after gpu_dct_loeffler --- " << std::endl;

  verify_result(mcu_array, mcu_array_copy);
  std::cout << "--- after verify_result --- " << std::endl;
}
