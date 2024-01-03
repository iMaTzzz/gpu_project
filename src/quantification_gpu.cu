// Quantification step, GPU version
#include <stdint.h>
#include <qtables.h>

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

int16_t *quantify_cuda(int16_t *array, bool luminance)
{
    // Définissez les dimensions du bloc et du grille
    const dim3 block_size(32, 8, 1);
    const dim3 grid_size(ceil(64 / block_size.x), 8, 1);

    // Initialisez le tableau de données quantifié
    int16_t *quantified_array = (int16_t *)malloc(64 * sizeof(int16_t));

    // Appelez la fonction CUDA
    quantify_kernel<<<grid_size, block_size>>>(quantified_array, luminance);

    // Attendez la fin de l'exécution de la fonction CUDA
    cudaDeviceSynchronize();

    // Renvoyez le tableau de données quantifié
    return quantified_array;
}