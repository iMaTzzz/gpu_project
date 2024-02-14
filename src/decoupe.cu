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
    // Allocate memory on the device
    int16_t *d_mcus_array;
    gpuErrchk(cudaMalloc(&d_mcus_array, mcus_array_size));

    uint16_t *index = (uint16_t *) malloc(sizeof(uint16_t));
    int16_t *predicator = (int16_t *) calloc(1, sizeof(int16_t));

    for (uint16_t nb_alloc = 0; nb_alloc < nb_mcu_column / nb_mcus_line_allocated; ++nb_alloc) {
        for (uint32_t mcu_line = 0; mcu_line < nb_mcus_line_allocated && nb_alloc * nb_mcus_line_allocated + mcu_line < nb_mcu_column; mcu_line++) {
            uint32_t global_mcu_line = mcu_line + nb_alloc * nb_mcus_line_allocated;
            uint32_t mcu_offset = mcu_line * 8 * mcus_line_array_width;
            // Troncature en bas possible que sur la dernière ligne de MCU
            if (tronc_down && global_mcu_line == nb_mcu_column - 1) {
                // On parcourt les lignes de la MCU jusqu'à la dernière présente dans l'image
                for (uint8_t line = 0; line < height_remainder; ++line) {
                    uint32_t column;
                    for (column = 0; column < width; ++column) {
                        mcus_array[mcu_offset + (column / 8) * 64 + line * 8 + column % 8] = fgetc(image);
                    }
                    // Troncature à droite possible que sur la dernière colonne de MCU
                    if (tronc_right) {
                        uint8_t column_index = column % 8;
                        for (uint8_t column_offset = column_index; column_offset < 8; ++column_offset) {
                            // On copie la valeur précédente pour remplir le reste de la ligne
                            uint32_t row_in_last_mcu = mcu_offset + (nb_mcu_line - 1) * 64 + line * 8;
                            mcus_array[row_in_last_mcu + column_offset] = mcus_array[row_in_last_mcu + column_offset - 1];
                        }
                    }

                }
                // Puis on copie la dernière ligne de pixels présente dans l'image dans les lignes manquantes
                for (uint8_t line_offset = height_remainder; line_offset < 8; ++line_offset) {
                    uint32_t column;
                    for (column = 0; column < width; ++column) {
                        uint32_t same_column_in_same_mcu = mcu_offset + (column / 8) * 64 + column % 8;
                        mcus_array[same_column_in_same_mcu + line_offset * 8] = mcus_array[same_column_in_same_mcu + (height_remainder - 1) * 8];
                    }
                    // Troncature à droite possible que sur la dernière colonne de MCU
                    if (tronc_right) {
                        uint8_t column_index = column % 8;
                        for (uint8_t column_offset = column_index; column_offset < 8; ++column_offset) {
                            uint32_t last_mcu_in_line = mcu_offset + (nb_mcu_line - 1) * 64;
                            // On copie la valeur précédente pour remplir le reste de la ligne
                            mcus_array[last_mcu_in_line + line_offset * 8 + column_offset] = mcus_array[last_mcu_in_line + (height_remainder - 1) * 8 + column_index - 1];
                        }
                    }
                }
            } else { // Pas de troncature vers le bas ou on ne se trouve pas sur la dernière ligne de MCU
                for (uint8_t line = 0; line < 8; ++line) {
                    uint32_t column;
                    for (column = 0; column < width; ++column) {
                        mcus_array[mcu_offset + (column / 8) * 64 + line * 8 + column % 8] = fgetc(image);
                    }
                    // Troncature à droite possible que sur la dernière colonne de MCU
                    if (tronc_right) {
                        uint8_t column_index = column % 8;
                        for (uint8_t column_offset = column_index; column_offset < 8; ++column_offset) {
                            // On copie la valeur précédente pour remplir le reste de la ligne
                            uint32_t row_in_last_mcu = mcu_offset + (nb_mcu_line - 1) * 64 + line * 8;
                            mcus_array[row_in_last_mcu + column_offset] = mcus_array[row_in_last_mcu + column_index - 1];
                        }
                    }
                }
            }
            // TODO
            // Call GPU
            encoding(mcus_array, d_mcus_array, nb_mcus_line_allocated * nb_mcu_line, mcus_array_size, true);
            // Take result from GPU
            // Call coding from results of GPU
            coding_mcus(mcus_array, nb_mcus_line_allocated * nb_mcu_line, ht_dc, ht_ac, stream, predicator, index);
        }
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
    size_t mcus_line_array_size = height_mcu * mcus_line_array_width * 3 * sizeof(int16_t);

    // Tableau des MCUs qui contient toutes les composantes dans l'ordre séquentiel de l'encodage
    int16_t *mcus_line_array = (int16_t *) malloc(mcus_line_array_size);
    if (mcus_line_array == NULL) {
        printf("malloc for mcus line array failed\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory on the device
    int16_t *d_mcus_line_array;
    gpuErrchk(cudaMalloc(&d_mcus_line_array, mcus_line_array_size));

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


    for (uint32_t mcu_line = 0; mcu_line < nb_mcu_column; ++mcu_line) {
        // Troncature en bas possible que sur la dernière ligne de MCU
        if (tronc_down && mcu_line == nb_mcu_column - 1) {
            // On parcourt les lignes de la MCU jusqu'à la dernière présente dans l'image
            for (uint8_t line = 0; line < height_remainder; ++line) {
                uint32_t column;
                for (column = 0; column < width; ++column) {
                    uint32_t mcu_pixel = (column / 8) * 64 * 3 + line * 8 + column % 8;
                    rgb_to_ycbcr(fgetc(image), fgetc(image), fgetc(image), &Y, &Cb, &Cr);
                    mcus_line_array[mcu_pixel] = Y; // Y
                    mcus_line_array[mcu_pixel + 64] = Cb; // Cb
                    mcus_line_array[mcu_pixel + 128] = Cr; // Cr
                }
                // Troncature à droite possible que sur la dernière colonne de MCU
                if (tronc_right) {
                    uint8_t column_index = column % width_mcu;
                    for (uint8_t column_offset = column_index; column_offset < width_mcu; ++column_offset) {
                        // TODO => Modifier accès en fonction de la taille des MCUs et rajouter Cb et Cr
                        // On copie la valeur précédente pour remplir le reste de la ligne
                        uint32_t row_in_last_mcu = (nb_mcu_line - 1) * 64 * 3 + line * height_mcu;
                        uint32_t mcu_pixel = row_in_last_mcu + column_offset;
                        uint32_t mcu_last_pixel = row_in_last_mcu + column_index - 1;
                        mcus_line_array[mcu_pixel] = mcus_line_array[mcu_last_pixel - 1];
                        mcus_line_array[mcu_pixel + 64] = mcus_line_array[mcu_last_pixel - 1 + 64];
                        mcus_line_array[mcu_pixel + 128] = mcus_line_array[mcu_last_pixel - 1 + 128];
                    }
                }

            }
            // Puis on copie la dernière ligne de pixels présente dans l'image dans les lignes manquantes
            for (uint8_t line_offset = height_remainder; line_offset < 8; ++line_offset) {
                uint32_t column;
                for (column = 0; column < width; ++column) {
                    uint32_t same_column_in_same_mcu = (column / 8) * 64 * 3 + column % 8;
                    uint32_t mcu_pixel = same_column_in_same_mcu + line_offset * 8;
                    uint32_t mcu_last_pixel = same_column_in_same_mcu + (height_remainder - 1) * 8;
                    mcus_line_array[mcu_pixel] = mcus_line_array[mcu_last_pixel];
                    mcus_line_array[mcu_pixel + 64] = mcus_line_array[mcu_last_pixel + 64];
                    mcus_line_array[mcu_pixel + 128] = mcus_line_array[mcu_last_pixel + 128];
                }
                // Troncature à droite possible que sur la dernière colonne de MCU
                if (tronc_right) {
                    uint8_t column_index = column % width_mcu;
                    for (uint8_t column_offset = column_index; column_offset < width_mcu; ++column_offset) {
                        uint32_t mcu_pixel = (nb_mcu_line - 1) * 64 * 3 + line_offset * 8 + column_offset;
                        uint32_t mcu_last_pixel = (nb_mcu_line - 1) * 64 + (height_remainder - 1) * 8 + column_index - 1;
                        // On copie la valeur précédente pour remplir le reste de la ligne
                        mcus_line_array[mcu_pixel] = mcus_line_array[mcu_last_pixel];
                        mcus_line_array[mcu_pixel + 64] = mcus_line_array[mcu_last_pixel + 64];
                        mcus_line_array[mcu_pixel + 128] = mcus_line_array[mcu_last_pixel + 128];
                    }
                }
            }
        } else { // Pas de troncature vers le bas ou on ne se trouve pas sur la dernière ligne de MCU
            for (uint8_t line = 0; line < height_mcu; ++line) {
                uint32_t column;
                for (column = 0; column < width; ++column) {
                    uint32_t mcu_pixel = (column / 8) * 64 * 3 + line * 8 + column % 8;
                    red = fgetc(image);
                    green = fgetc(image);
                    blue = fgetc(image);
                    rgb_to_ycbcr(red, green, blue, &Y, &Cb, &Cr);
                    mcus_line_array[mcu_pixel] = Y; // Y
                    mcus_line_array[mcu_pixel + 64] = Cb; // Cb
                    mcus_line_array[mcu_pixel + 128] = Cr; // Cr
                }
                // Troncature à droite possible que sur la dernière colonne de MCU
                if (tronc_right) {
                    uint8_t column_index = column % width_mcu;
                    for (uint8_t column_offset = column_index; column_offset < width_mcu; ++column_offset) {
                        // TODO => Modifier accès en fonction de la taille des MCUs et rajouter Cb et Cr
                        // On copie la valeur précédente pour remplir le reste de la ligne
                        // mcus_line_array[line * mcus_line_array_width + column_offset] = mcus_line_array[line * mcus_line_array_width + column_offset - 1];
                        uint32_t row_in_last_mcu = (nb_mcu_line - 1) * 64 * 3 + line * height_mcu;
                        uint32_t mcu_pixel = row_in_last_mcu + column_offset;
                        uint32_t mcu_last_pixel = row_in_last_mcu + column_index - 1;
                        mcus_line_array[mcu_pixel] = mcus_line_array[mcu_last_pixel - 1];
                        mcus_line_array[mcu_pixel + 64] = mcus_line_array[mcu_last_pixel - 1 + 64];
                        mcus_line_array[mcu_pixel + 128] = mcus_line_array[mcu_last_pixel - 1 + 128];
                    }
                }
            }
        }
        // TODO
        // Give size to allocate on GPU

        // Call GPU
        encoding(mcus_line_array, d_mcus_line_array, nb_mcu_line*3, mcus_line_array_size, true);
        // Take result from GPU
        // Call coding from results of GPU
        coding_mcus_Y_Cb_Cr(mcus_line_array, nb_mcu_line, ht_dc_Y, ht_ac_Y, ht_dc_C, ht_ac_C, stream, predicator_Y, predicator_Cb, predicator_Cr, index);
    }

    // OLD
    // /* On parcourt successivement les différentes MCUs (ligne par ligne et de gauche à droite). */
    // for (uint32_t i = 0; i < nb_mcu_column; i++) {
        // //printf("i = %i\n", i);
        // for (uint32_t j = 0; j < nb_mcu_line; j++) {
            // // printf("j = %i\n", j);
            // if (tronc_line && i == nb_mcu_column - 1) {
                // tronc_line_mcu = 1;
            // } else {
                // tronc_line_mcu = 0;
            // }
            // if (tronc_column && j == nb_mcu_line - 1) {
                // tronc_column_mcu = 1;
            // } else {
                // tronc_column_mcu = 0;
            // }
            // /* On parcourt successivement les différents pixels d'une MCU (ligne par ligne et de gauche à droite). */
            // for (uint8_t k = 0; k < height_mcu; k++) {
                // //printf("k = %i\n", k);
                // for (uint8_t l = 0; l < width_mcu; l++) {
                // /* Cas troncature à droite qui concerne le pixel traité. */
                    // if (tronc_column_mcu && l >= width_remainder) {
                        // /* Cas troncature à droite et troncature en bas qui concernent le pixel traité. */
                        // if (tronc_line_mcu && k >= height_remainder) {
                            // mcu_Y[k][l] = mcu_Y[height_remainder - 1][l]; 
                            // mcu_Cb[k][l] = mcu_Cb[height_remainder - 1][l]; 
                            // mcu_Cr[k][l] = mcu_Cr[height_remainder - 1][l]; 
                        // } else {
                            // mcu_Y[k][l] = mcu_Y[k][width_remainder - 1];
                            // mcu_Cb[k][l] = mcu_Cb[k][width_remainder - 1];
                            // mcu_Cr[k][l] = mcu_Cr[k][width_remainder - 1];
                        // }
                    // /* Cas troncature en bas qui concerne le pixel traité. */
                    // } else if (tronc_line_mcu && k >= height_remainder) {
                        // mcu_Y[k][l] = mcu_Y[height_remainder - 1][l];
                        // mcu_Cb[k][l] = mcu_Cb[height_remainder - 1][l];
                        // mcu_Cr[k][l] = mcu_Cr[height_remainder- 1][l];
                        // fseek(image, 3, SEEK_CUR); // fseek permet de déplacer le curseur où on le souhaite dans le fichier. 
                    // /* Pas de troncature impactant le pixel traité. */
                    // } else {
                        // //printf("l = %i\n", l);
                        // red = fgetc(image);
                        // green = fgetc(image);
                        // blue = fgetc(image);
                        // mcu_Y[k][l] = rgb_to_ycbcr(red, green, blue, Y);
                        // mcu_Cb[k][l] = rgb_to_ycbcr(red, green, blue, Cb);
                        // mcu_Cr[k][l] = rgb_to_ycbcr(red, green, blue, Cr);
                        // // printf("%ld\n", ftell(image));
                    // }
                // }
                // /* Si on n'a pas atteint la dernière ligne de la MCU, on passe à la ligne suivante. */
                // if (k < height_mcu - 1) {
                    // if (!tronc_line_mcu && !tronc_column_mcu) {
                        // fseek(image, 3 * (width - width_mcu), SEEK_CUR);
                    // } else if (!tronc_line_mcu && tronc_column_mcu) {
                        // fseek(image, 3 * (width - width_remainder), SEEK_CUR);
                    // } else if (tronc_line_mcu && !tronc_column_mcu) {
                        // if (k < height_remainder - 1) {
                            // fseek(image, 3 * (width - width_mcu), SEEK_CUR);
                        // } else {
                            // fseek(image, -3 * width_mcu, SEEK_CUR);
                        // }
                    // } else {
                        // if (k < height_remainder - 1) {
                            // fseek(image, 3 * (width - width_remainder), SEEK_CUR);
                        // } else {
                            // fseek(image, - 3 * width_remainder, SEEK_CUR);
                        // }
                    // }
                // }
            // }
            // // print_matrix_mcu(mcu_Y, width_mcu, heigth_mcu);
            // /* On encode d'abord la composante Y */
            // for (uint8_t k = 0; k < v1; k++) {
                // for (uint8_t l = 0; l < h1; l++) {
                    // /* On traite la MCU bloc par bloc. */ 
                    // for (uint8_t m = 0; m < 8; m++) {
                        // for (uint8_t n = 0; n < 8; n++) {                            
                            // bloc[m][n] = mcu_Y[8*k+m][8*l+n];
                        // }
                    // }
                    // dct_loeffler(bloc, bloc_array); // On transforme le bloc en tableau, en appliquant la DCT.
                    // //dct_faster_loeffler(bloc, bloc_array);
                    // //dct_arai(bloc, bloc_array);
                    // //dct_arai_bis(bloc, bloc_array);
                    // // print_array_16(mcu_array);
                    // quantify(bloc_array, true); // On applique la quantification au bloc.
                    // // print_array_16(mcu_array);
                    // coding(bloc_array, ht_dc_Y, ht_ac_Y, stream, predicator_Y, index); // On encode le bloc.
                // }
            // }
            
            // /* Puis on encode la composante Cb */
            // if (h2 != h1 || v2 != v1) {
                // downsampling(mcu_Cb, mcu_Cb_sampling, h2, v2, h1, v1); // On sous-échantillonne si nécessaire.
            // }
            // for (uint8_t k = 0; k < v2; k++) {
                // for (uint8_t l = 0; l < h2; l++) {
                    // /* On traite la MCU bloc par bloc. */ 
                    // for (uint8_t m = 0; m < 8; m++) {
                        // for (uint8_t n = 0; n < 8; n++) {
                            // if (h2 != h1 || v2 != v1) {
                                // bloc[m][n] = mcu_Cb_sampling[8*k+m][8*l+n];
                            // } else {
                                // bloc[m][n] = mcu_Cb[8*k+m][8*l+n];
                            // }
                        // }
                    // }
                    // //print_matrix_8(mcu_Cb);
                    // dct_loeffler(bloc, bloc_array); // On transforme le bloc en tableau, en appliquant la DCT.
                    // //dct_faster_loeffler(bloc, bloc_array);
                    // //dct_arai(bloc, bloc_array);
                    // //dct_arai_bis(bloc, bloc_array);
                    // // print_array_16(mcu_array);
                    // quantify(bloc_array, false); // On applique la quantification au bloc.
                    // // print_array_16(mcu_array);
                    // coding(bloc_array, ht_dc_C, ht_ac_C, stream, predicator_Cb, index); // On encode le bloc.
                // }
            // }


            // /* Enfin on encode la composante Cr */
            // if (h3 != h1 || v3 != v1) {
                // downsampling(mcu_Cr, mcu_Cr_sampling, h3, v3, h1, v1); // On sous-échantillonne si nécessaire.
            // }
            // for (uint8_t k = 0; k < v3; k++) {
                // for (uint8_t l = 0; l < h3; l++) {
                    // /* On traite la MCU bloc par bloc. */ 
                    // for (uint8_t m = 0; m < 8; m++) {
                        // for (uint8_t n = 0; n < 8; n++) {
                            // if (h3 != h1 || v3 != v1) {
                                // bloc[m][n] = mcu_Cr_sampling[8*k+m][8*l+n];
                            // } else {
                                // bloc[m][n] = mcu_Cr[8*k+m][8*l+n];
                            // }
                        // }
                    // }
                    // //print_matrix_8(mcu_Cb);
                    // dct_loeffler(bloc, bloc_array); // On transforme le bloc en tableau, en appliquant la DCT.
                    // //dct_faster_loeffler(bloc, bloc_array);
                    // //dct_arai(bloc, bloc_array);
                    // //dct_arai_bis(bloc, bloc_array);
                    // // print_array_16(mcu_array);
                    // quantify(bloc_array, false); // On applique la quantification au bloc.
                    // // print_array_16(mcu_array);
                    // coding(bloc_array, ht_dc_C, ht_ac_C, stream, predicator_Cr, index); // On encode le bloc.
                // }
            // }
            
            // /* Si on n'est pas sur une MCU de la dernière colonne */
            // if (j != column_mcu - 1) {
                // if (!tronc_line_mcu) {
                    // long offset = (1 - height_mcu) * (long)width;
                    // fseek(image, 3 * offset, SEEK_CUR); // -2240 = -7 * width
                // } else { // Si la MCU est tronquée en bas, on ne doit pas remonter le curseur trop haut. 
                    // long offset = (1 - (long) height_remainder) * (long)width;
                    // fseek(image, 3 * offset, SEEK_CUR); 
                // }
            // }
        // }
    // }
    /* On libère tous les espaces mémoire alloués. */
    gpuErrchk(cudaFree(d_mcus_line_array));
    free(mcus_line_array);
    free(predicator_Y);
    free(predicator_Cb);
    free(predicator_Cr);
    free(index);
}
