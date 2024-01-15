// Quantification step, GPU version
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

__managed__ static uint8_t quantification_table_Y[] = {
    0x05, 0x03, 0x03, 0x05, 0x07, 0x0c, 0x0f, 0x12,
    0x04, 0x04, 0x04, 0x06, 0x08, 0x11, 0x12, 0x11,
    0x04, 0x04, 0x05, 0x07, 0x0c, 0x11, 0x15, 0x11,
    0x04, 0x05, 0x07, 0x09, 0x0f, 0x1a, 0x18, 0x13,
    0x05, 0x07, 0x0b, 0x11, 0x14, 0x21, 0x1f, 0x17,
    0x07, 0x0b, 0x11, 0x13, 0x18, 0x1f, 0x22, 0x1c,
    0x0f, 0x13, 0x17, 0x1a, 0x1f, 0x24, 0x24, 0x1e,
    0x16, 0x1c, 0x1d, 0x1d, 0x22, 0x1e, 0x1f, 0x1e
};

__managed__ static uint8_t quantification_table_CbCr[] = {
    0x05, 0x05, 0x07, 0x0e, 0x1e, 0x1e, 0x1e, 0x1e,
    0x05, 0x06, 0x08, 0x14, 0x1e, 0x1e, 0x1e, 0x1e,
    0x07, 0x08, 0x11, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
    0x0e, 0x14, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
    0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
    0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
    0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e,
    0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e
};

__global__ void quantify_kernel_Y(int16_t *array)
{
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < 64) {
        array[thread_id] = array[thread_id] / quantification_table_Y[thread_id];
    }
}

__global__ void quantify_kernel_CbCr(int16_t *array)
{
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < 64) {
        array[thread_id] = array[thread_id] / quantification_table_CbCr[thread_id];
    }
}

void quantify_gpu(int16_t *h_array, bool luminance)
{
    // Give size to allocate on GPU
    const int size = 64*sizeof(int16_t);

    // Allocate memory on the device
    int16_t *d_array;
    cudaMalloc(&d_array, size);

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    // TODO
    // Define block size and grid size
    const dim3 block_size(64);
    const dim3 grid_size((int)ceil(64 / block_size.x)); // 1 grid

    // Execute kernel
    if (luminance) {
        quantify_kernel_Y<<<grid_size, block_size>>>(d_array);
    } else {
        quantify_kernel_CbCr<<<grid_size, block_size>>>(d_array);
    }

    // Copy result of computation back on host
    // cudaMemcpy will wait for kernel completion (acts as synchronization barrier)
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    cudaFree(d_array);
}

void verify_result(int16_t *array_cpu, int16_t *array_gpu)
{
    for (int i = 0; i < 64; ++i) {
        assert(array_cpu[i] == array_gpu[i]);
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

extern "C"
void quantify(int16_t *array, bool luminance)
{
    printf("On rentre dans quantify\n");
    int16_t *array_copy = (int16_t *)malloc(64*sizeof(int16_t));
    // Create a copy of the original array
    for (int i = 0; i < 64; ++i) {
        array_copy[i] = array[i];
    }
    printf("On crée une copie du tableau original\n");

    // Run this on CPU
    quantify_cpu(array, luminance);
    printf("Quantify_cpu fait\n");

    // Run this on GPU
    quantify_gpu(array_copy, luminance);
    printf("Quantify_gpu fait\n");

    verify_result(array, array_copy);
    printf("Check result fait\n");
}
