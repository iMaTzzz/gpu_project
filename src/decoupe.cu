#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "quantification.h"
#include "coding.h"
#include "dct.h"
#include "jpeg_writer.h"
#include "rgb_to_ycbcr.h"
#include "downsampling.h"
#include "bitstream.h"
#include "encoding.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
    Dans cette fonction qui s'occupe des images en noir et blanc,
    on traite chaque MCU intégralement, en effectuant les transformations successives,
    avant de passer à la suivante.
*/
extern "C"
void treat_image_grey(FILE *image, uint32_t width, uint32_t height, struct huff_table *ht_dc, struct huff_table *ht_ac, struct bitstream *stream)
{
    // TODO => Check memoire sur les mallocs/callocs
    uint32_t nb_mcu_column = height / 8; // Nombre de MCUs par colonne
    uint32_t nb_mcu_line = width / 8; // Nombre de MCUs par ligne

    uint8_t height_remainder = height % 8; // Nombre de lignes dépassant (en cas de troncature).
    uint8_t width_remainder = width % 8; // Nombre de colonnes dépassant (en cas de troncature).

    bool tronc_down = 0; // Indique si il y a une troncature en bas.
    bool tronc_right = 0; // Indique si il y a une troncature à droite.

    if (height_remainder != 0) {
        nb_mcu_column++; // On rajoute une ligne de MCUs.
        tronc_down = 1; // Il y a troncature en bas.
    }
    if (width_remainder != 0) {
        nb_mcu_line++; // On rajoute une colonne de MCUs.
        tronc_right = 1; // Il y a troncature à droite.
    }

    /* On alloue tous les espaces mémoire nécessaires. */
    uint16_t mcus_line_array_width = width_remainder == 0 ? width : width + 8 - width_remainder;
    uint16_t mcus_array_height = height_remainder == 0 ? height : height + 8 - height_remainder;
    size_t mcus_array_size = mcus_array_height * mcus_line_array_width * sizeof(int16_t);
    int16_t *mcus_array = (int16_t *) malloc(mcus_array_size);
    uint16_t nb_mcus_line_allocated = nb_mcu_column; // Nombre de ligne de MCUs alloué
    while (mcus_array == NULL) {
        printf("malloc for mcus line array failed\n");
        // We remove a line of mcus in the array until we find the maximum number of lines that can be malloc'ed
        mcus_array_size -= mcus_array_size / 2;
        mcus_array = (int16_t *) malloc(mcus_array_size);
        nb_mcus_line_allocated /= 2;
        if (nb_mcus_line_allocated % 2) {
            nb_mcus_line_allocated++;
        }
    }
    printf("Nombre de lignes de MCUs alloué: %u, Nombre de lignes de MCUs: %u\n", nb_mcus_line_allocated, nb_mcu_column);
    // Allocate memory on the device
    int16_t *d_mcus_array;
    gpuErrchk(cudaMalloc(&d_mcus_array, mcus_array_size));

    uint16_t *index = (uint16_t *) malloc(sizeof(uint16_t));
    int16_t *predicator = (int16_t *) calloc(1, sizeof(int16_t));

    uint32_t nb_mcus_allocated = nb_mcus_line_allocated * nb_mcu_line;
    printf("nb mcus allocated: %u\n", nb_mcus_allocated);

    for (uint16_t nb_alloc = 0; nb_alloc < nb_mcu_column / nb_mcus_line_allocated; ++nb_alloc) {
        // printf("Current number alloc: %u\n", nb_alloc);
        for (uint32_t mcu_line = 0; mcu_line < nb_mcus_line_allocated && nb_alloc * nb_mcus_line_allocated + mcu_line < nb_mcu_column; mcu_line++) {
            // printf("Current mcu line: %u\n", mcu_line);
            uint32_t global_mcu_line = mcu_line + nb_alloc * nb_mcus_line_allocated;
            uint32_t mcu_line_offset = mcu_line * 8 * mcus_line_array_width;
            // Troncature en bas possible que sur la dernière ligne de MCU
            if (tronc_down && global_mcu_line == nb_mcu_column - 1) {
                // On parcourt les lignes de la MCU jusqu'à la dernière présente dans l'image
                for (uint8_t line = 0; line < height_remainder; ++line) {
                    uint32_t column;
                    for (column = 0; column < width; ++column) {
                        mcus_array[mcu_line_offset + (column / 8) * 64 + line * 8 + column % 8] = fgetc(image);
                    }
                    // Troncature à droite possible que sur la dernière colonne de MCU
                    if (tronc_right) {
                        uint8_t column_index = column % 8;
                        for (uint8_t column_offset = column_index; column_offset < 8; ++column_offset) {
                            // On copie la valeur précédente pour remplir le reste de la ligne
                            uint32_t row_in_last_mcu = mcu_line_offset + (nb_mcu_line - 1) * 64 + line * 8;
                            mcus_array[row_in_last_mcu + column_offset] = mcus_array[row_in_last_mcu + column_offset - 1];
                        }
                    }

                }
                // Puis on copie la dernière ligne de pixels présente dans l'image dans les lignes manquantes
                for (uint8_t line_offset = height_remainder; line_offset < 8; ++line_offset) {
                    uint32_t column;
                    for (column = 0; column < width; ++column) {
                        uint32_t same_column_in_same_mcu = mcu_line_offset + (column / 8) * 64 + column % 8;
                        mcus_array[same_column_in_same_mcu + line_offset * 8] = mcus_array[same_column_in_same_mcu + (height_remainder - 1) * 8];
                    }
                    // Troncature à droite possible que sur la dernière colonne de MCU
                    if (tronc_right) {
                        uint8_t column_index = column % 8;
                        for (uint8_t column_offset = column_index; column_offset < 8; ++column_offset) {
                            uint32_t last_mcu_in_line = mcu_line_offset + (nb_mcu_line - 1) * 64;
                            // On copie la valeur précédente pour remplir le reste de la ligne
                            mcus_array[last_mcu_in_line + line_offset * 8 + column_offset] = mcus_array[last_mcu_in_line + (height_remainder - 1) * 8 + column_index - 1];
                        }
                    }
                }
            } else { // Pas de troncature vers le bas ou on ne se trouve pas sur la dernière ligne de MCU
                for (uint8_t line = 0; line < 8; ++line) {
                    uint32_t column;
                    for (column = 0; column < width; ++column) {
                        mcus_array[mcu_line_offset + (column / 8) * 64 + line * 8 + column % 8] = fgetc(image);
                    }
                    // Troncature à droite possible que sur la dernière colonne de MCU
                    if (tronc_right) {
                        uint8_t column_index = column % 8;
                        for (uint8_t column_offset = column_index; column_offset < 8; ++column_offset) {
                            // On copie la valeur précédente pour remplir le reste de la ligne
                            uint32_t row_in_last_mcu = mcu_line_offset + (nb_mcu_line - 1) * 64 + line * 8;
                            mcus_array[row_in_last_mcu + column_offset] = mcus_array[row_in_last_mcu + column_index - 1];
                        }
                    }
                }
            }
        }
        // TODO
        // Call GPU
        encoding(mcus_array, d_mcus_array, nb_mcus_allocated, mcus_array_size, true);
        // Take result from GPU
        // Call coding from results of GPU
        coding_mcus(mcus_array, nb_mcus_allocated, ht_dc, ht_ac, stream, predicator, index);
    }
    /* On libère tous les espaces mémoire alloués. */
    gpuErrchk(cudaFree(d_mcus_array));
    free(mcus_array);
    free(predicator);
    free(index);
}

extern "C"
void treat_image_color(FILE *image, uint32_t width, uint32_t height, struct huff_table *ht_dc_Y, 
                        struct huff_table *ht_ac_Y, struct huff_table *ht_dc_C, struct huff_table *ht_ac_C, 
                        struct bitstream *stream, uint8_t h1, uint8_t v1, uint8_t h2, uint8_t v2, uint8_t h3, uint8_t v3)
{
    // TODO
    uint8_t width_mcu = 8*h1; // Nombre de pixels sur une ligne d'une MCU
    uint8_t height_mcu = 8*v1;  // Nombre de pixels sur une colonne d'une MCU.

    uint32_t nb_mcu_column = height / height_mcu; // Nombre de MCUs par colonne
    uint32_t nb_mcu_line = width / width_mcu; // Nombre de MCUs par ligne

    uint8_t height_remainder = height % height_mcu; // Nombre de lignes dépassant (en cas de troncature).
    uint8_t width_remainder = width % width_mcu; // Nombre de colonnes dépassant (en cas de troncature).

    bool tronc_down = 0; // Indique si il y a une troncature en bas.
    bool tronc_right = 0; // Indique si il y a une troncature à droite.

    if (height_remainder != 0) {
        nb_mcu_column++; // On rajoute une ligne de MCUs.
        tronc_down = 1; // Il y a troncature en bas.
    }
    if (width_remainder != 0) {
        nb_mcu_line++; // On rajoute une colonne de MCUs.
        tronc_right = 1; // Il y a troncature à droite.
    }

    uint16_t mcus_line_array_width = width_remainder == 0 ? width : width + width_mcu - width_remainder;
    uint16_t mcus_array_height = height_remainder == 0 ? height : height + height_mcu - height_remainder;
    size_t mcus_array_size = mcus_array_height * mcus_line_array_width * 3 * sizeof(int16_t);
    int16_t *mcus_array = (int16_t *) malloc(mcus_array_size);
    uint16_t nb_mcus_line_allocated = nb_mcu_column; // Nombre de ligne de MCUs alloué
    while (mcus_array == NULL) {
        printf("malloc for mcus line array failed\n");
        // We remove a line of mcus in the array until we find the maximum number of lines that can be malloc'ed
        mcus_array_size -= mcus_array_size / 2;
        mcus_array = (int16_t *) malloc(mcus_array_size);
        nb_mcus_line_allocated /= 2;
        if (nb_mcus_line_allocated % 2) {
            nb_mcus_line_allocated++;
        }
    }

    // Allocate memory on the device
    int16_t *d_mcus_array;
    gpuErrchk(cudaMalloc(&d_mcus_array, mcus_array_size));

    uint16_t *index = (uint16_t *) malloc(sizeof(uint16_t));
    // On utilise un prédicateur par composante.
    int16_t *predicator_Y = (int16_t *) calloc(1, sizeof(int16_t));
    int16_t *predicator_Cb = (int16_t *) calloc(1, sizeof(int16_t));
    int16_t *predicator_Cr = (int16_t *) calloc(1, sizeof(int16_t));

    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t Y;
    uint8_t Cb;
    uint8_t Cr;

    uint32_t nb_mcus_allocated = nb_mcus_line_allocated * nb_mcu_line;
    printf("nb mcus allocated: %u\n", nb_mcus_allocated);

    for (uint16_t nb_alloc = 0; nb_alloc < nb_mcu_column / nb_mcus_line_allocated; ++nb_alloc) {
        // printf("Current number alloc: %u\n", nb_alloc);
        for (uint32_t mcu_line = 0; mcu_line < nb_mcus_line_allocated && nb_alloc * nb_mcus_line_allocated + mcu_line < nb_mcu_column; mcu_line++) {
            // printf("Current mcu line: %u\n", mcu_line);
            uint32_t global_mcu_line = mcu_line + nb_alloc * nb_mcus_line_allocated;
            uint32_t mcu_line_offset = mcu_line * 8 * mcus_line_array_width * 3;
            // Troncature en bas possible que sur la dernière ligne de MCU
            if (tronc_down && global_mcu_line == nb_mcu_column - 1) {
                // On parcourt les lignes de la MCU jusqu'à la dernière présente dans l'image
                for (uint8_t line = 0; line < height_remainder; ++line) {
                    uint32_t column;
                    for (column = 0; column < width; ++column) {
                        uint32_t mcu_pixel = mcu_line_offset + (column / 8) * 64 * 3 + line * 8 + column % 8;
                        rgb_to_ycbcr(fgetc(image), fgetc(image), fgetc(image), &Y, &Cb, &Cr);
                        mcus_array[mcu_pixel] = Y; // Y
                        mcus_array[mcu_pixel + 64] = Cb; // Cb
                        mcus_array[mcu_pixel + 128] = Cr; // Cr
                    }
                    // Troncature à droite possible que sur la dernière colonne de MCU
                    if (tronc_right) {
                        uint8_t column_index = column % width_mcu;
                        for (uint8_t column_offset = column_index; column_offset < width_mcu; ++column_offset) {
                            // TODO => Modifier accès en fonction de la taille des MCUs et rajouter Cb et Cr
                            // On copie la valeur précédente pour remplir le reste de la ligne
                            uint32_t row_in_last_mcu = mcu_line_offset + (nb_mcu_line - 1) * 64 * 3 + line * height_mcu;
                            uint32_t mcu_pixel = row_in_last_mcu + column_offset;
                            uint32_t mcu_last_pixel = row_in_last_mcu + column_index - 1;
                            mcus_array[mcu_pixel] = mcus_array[mcu_last_pixel - 1];
                            mcus_array[mcu_pixel + 64] = mcus_array[mcu_last_pixel - 1 + 64];
                            mcus_array[mcu_pixel + 128] = mcus_array[mcu_last_pixel - 1 + 128];
                        }
                    }

                }
                // Puis on copie la dernière ligne de pixels présente dans l'image dans les lignes manquantes
                for (uint8_t line_offset = height_remainder; line_offset < 8; ++line_offset) {
                    uint32_t column;
                    for (column = 0; column < width; ++column) {
                        uint32_t same_column_in_same_mcu = mcu_line_offset + (column / 8) * 64 * 3 + column % 8;
                        uint32_t mcu_pixel = same_column_in_same_mcu + line_offset * 8;
                        uint32_t mcu_last_pixel = same_column_in_same_mcu + (height_remainder - 1) * 8;
                        mcus_array[mcu_pixel] = mcus_array[mcu_last_pixel];
                        mcus_array[mcu_pixel + 64] = mcus_array[mcu_last_pixel + 64];
                        mcus_array[mcu_pixel + 128] = mcus_array[mcu_last_pixel + 128];
                    }
                    // Troncature à droite possible que sur la dernière colonne de MCU
                    if (tronc_right) {
                        uint8_t column_index = column % width_mcu;
                        for (uint8_t column_offset = column_index; column_offset < width_mcu; ++column_offset) {
                            uint32_t last_mcu_in_line = mcu_line_offset + (nb_mcu_line - 1) * 64 * 3;
                            uint32_t mcu_pixel = last_mcu_in_line + line_offset * 8 + column_offset;
                            uint32_t mcu_last_pixel = last_mcu_in_line + (height_remainder - 1) * 8 + column_index - 1;
                            // On copie la valeur précédente pour remplir le reste de la ligne
                            mcus_array[mcu_pixel] = mcus_array[mcu_last_pixel];
                            mcus_array[mcu_pixel + 64] = mcus_array[mcu_last_pixel + 64];
                            mcus_array[mcu_pixel + 128] = mcus_array[mcu_last_pixel + 128];
                        }
                    }
                }
            } else { // Pas de troncature vers le bas ou on ne se trouve pas sur la dernière ligne de MCU
                for (uint8_t line = 0; line < height_mcu; ++line) {
                    uint32_t column;
                    for (column = 0; column < width; ++column) {
                        uint32_t mcu_pixel = mcu_line_offset + (column / 8) * 64 * 3 + line * 8 + column % 8;
                        red = fgetc(image);
                        green = fgetc(image);
                        blue = fgetc(image);
                        rgb_to_ycbcr(red, green, blue, &Y, &Cb, &Cr);
                        mcus_array[mcu_pixel] = Y; // Y
                        mcus_array[mcu_pixel + 64] = Cb; // Cb
                        mcus_array[mcu_pixel + 128] = Cr; // Cr
                    }
                    // Troncature à droite possible que sur la dernière colonne de MCU
                    if (tronc_right) {
                        uint8_t column_index = column % width_mcu;
                        for (uint8_t column_offset = column_index; column_offset < width_mcu; ++column_offset) {
                            // TODO => Modifier accès en fonction de la taille des MCUs et rajouter Cb et Cr
                            // On copie la valeur précédente pour remplir le reste de la ligne
                            uint32_t row_in_last_mcu = mcu_line_offset + (nb_mcu_line - 1) * 64 * 3 + line * height_mcu;
                            uint32_t mcu_pixel = row_in_last_mcu + column_offset;
                            uint32_t mcu_last_pixel = row_in_last_mcu + column_index - 1;
                            mcus_array[mcu_pixel] = mcus_array[mcu_last_pixel - 1];
                            mcus_array[mcu_pixel + 64] = mcus_array[mcu_last_pixel - 1 + 64];
                            mcus_array[mcu_pixel + 128] = mcus_array[mcu_last_pixel - 1 + 128];
                        }
                    }
                }
            }
        }
        // TODO
        // Give size to allocate on GPU

        // Call GPU
        encoding(mcus_array, d_mcus_array, nb_mcus_allocated*3, mcus_array_size, true);
        // Take result from GPU
        // Call coding from results of GPU
        coding_mcus_Y_Cb_Cr(mcus_array, nb_mcus_allocated, ht_dc_Y, ht_ac_Y, ht_dc_C, ht_ac_C, stream, predicator_Y, predicator_Cb, predicator_Cr, index);
    }
    /* On libère tous les espaces mémoire alloués. */
    gpuErrchk(cudaFree(d_mcus_array));
    free(mcus_array);
    free(predicator_Y);
    free(predicator_Cb);
    free(predicator_Cr);
    free(index);
}
