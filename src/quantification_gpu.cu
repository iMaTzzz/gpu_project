// Quantification step, GPU version
#include <stdint.h>
#include "qtables.h"
// for ceil function
#include <cmath> // for c++
// #include <math.h> // for c

__constant__ const uint8_t quantification_table_Y[64];
__constant__ const uint8_t quantification_table_CbCr[64];


__global__ void quantify_kernel(int16_t *array, bool luminance)
{
    uint32_t thread_id = blockIdx.x * blockDim.x + threadId.x;

    if (thread_id < 64) {
        if (luminance) {
            array[thread_id] = array[thread_id] / quantification_table_Y[thread_id]
        } else {
            array[thread_id] = array[thread_id] / quantification_table_CbCr[thread_id]
        }
    }
}

void quantify_gpu(int16_t *h_array, bool luminance)
{
    // Give size to allocate on GPU
    const int size = 64*sizeof(int16_t);

    // Allocate memory on the device
    int16_t *d_array;
    bool d_luminance;
    cudaMalloc(&d_array, size);
    cudaMalloc(&d_luminance, sizeof(bool));

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_luminance, luminance, sizeof(bool), cudaMemcpyHostToDevice);

    // TODO
    // Define block size and grid size
    const dim3 block_size(64);
    const dim3 grid_size((int)ceil(64 / block_size.x)); // 1 grid

    // Execute kernel
    quantify_kernel<<<grid_size, block_size>>>(d_array, d_luminance);

    // Copy result of computation back on host
    // cudaMemcpy is a synchronous call and will therefore wait for kernel completion (serves as synchronization barrier)
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    cudaFree(d_array);
}

void verify_result(int16_t *array_cpu, int16_t *array_gpu)
{
    for (int i = 0; i < 64; ++i) {
        assert(array_cpu[i] == array_gpu);
    }
}

/*
    Quantifier le bloc 8x8 (qui a été transformé en array zigzag)
    Entrées : array int16_t de taille 64 et un bool qui vérifier si on est dans le cas Y ou CbCr
    Sortie : Rien 
*/
void quantify_cpu(int16_t *array, bool luminance)
{
    /*
        diviser terme à terme chaque bloc 8x8 par une matrice de quantification (sous le forme zig-zag déjà) suivant Y ou CbCr
    */
    for (uint8_t i=0; i<64; i++){
        if (luminance){
            array[i] = (array[i] / quantification_table_Y[i]);
        }
        else{
            array[i] = (array[i] / quantification_table_CbCr[i]);
        }
    }
}

void quantify(int16_t *array, bool luminance)
{
    int16_t *array_copy = malloc(64*sizeof(int16_t));
    // Create a copy of the original array
    for (int i = 0; i < 64; ++i) {
        array_copy[i] = array[i]
    }
    
    // Run this on CPU
    quantify_cpu(array, luminance);

    // Run this on GPU
    quantify_gpu(array_copy, luminance);

    verify_result(array, array_copy);
}
