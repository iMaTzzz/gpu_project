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

/*
    Dans cette fonction qui s'occupe des images en noir et blanc,
    on traite chaque MCU intégralement, en effectuant les transformations successives,
    avant de passer à la suivante.
*/
void treat_image_grey(FILE *image, uint32_t width, uint32_t height, struct huff_table *ht_dc, struct huff_table *ht_ac, struct bitstream *stream)
{
    printf("----- NEW treat image grey -----\n");

    /* On alloue tous les espaces mémoire nécessaires. */
    // TODO
    uint16_t *index = malloc(sizeof(uint16_t));
    // Idée faire une passe pour récupérer les predicateurs
    int16_t *predicator = calloc(1, sizeof(int16_t));

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

    printf("tronc_down: %d\n", (uint8_t) tronc_down);
    printf("tronc_right: %d\n", (uint8_t) tronc_right);

    uint16_t mcus_line_array_width = width_remainder == 0 ? width : width + 8 - width_remainder;

    printf("img width: %d\n", width);
    printf("mcus line width: %d\n", mcus_line_array_width);

    int16_t *mcus_line_array = malloc(8 * mcus_line_array_width * sizeof(int16_t));

    uint32_t mcu_index = 0; // absolute in whole image

    for (uint32_t mcu_line = 0; mcu_line < nb_mcu_column; ++mcu_line) {
        // Troncature en bas possible que sur la dernière ligne de MCU
        if (tronc_down && mcu_line == nb_mcu_column - 1) {
            // On parcourt les lignes de la MCU jusqu'à la dernière présente dans l'image
            for (uint8_t line = 0; line < height_remainder; ++line) {
                uint32_t column;
                for (column = 0; column < width; ++column) {
                    mcus_line_array[8 * (line + column)] = fgetc(image);
                }
                // Troncature à droite possible que sur la dernière colonne de MCU
                if (tronc_right) {
                    uint8_t column_index = column % 8;
                    for (uint8_t column_offset = column_index; column_offset < 8; ++column_offset) {
                        // On copie la valeur précédente pour remplir le reste de la ligne
                        mcus_line_array[8 * (line + column_offset)] = mcus_line_array[8 * (line + column_offset) - 1];
                    }
                }

            }
            // Puis on copie la dernière ligne de pixels présente dans l'image dans les lignes manquantes
            for (uint8_t line_offset = height_remainder; line_offset < 8; ++line_offset) {
                uint32_t column;
                for (column = 0; column < width; ++column) {
                    mcus_line_array[8 * (line_offset + column)] = mcus_line_array[8 * (line_offset + column - 1)];
                }
                // Troncature à droite possible que sur la dernière colonne de MCU
                if (tronc_right) {
                    uint8_t column_index = column % 8;
                    for (uint8_t column_offset = column_index; column_offset < 8; ++column_offset) {
                        // On copie la valeur précédente pour remplir le reste de la ligne
                        mcus_line_array[8 * (line_offset + column_offset)] = mcus_line_array[height * width - 1];
                    }
                }
            }
        } else { // Pas de troncature vers le bas ou on ne se trouve pas sur la dernière ligne de MCU
            for (uint8_t line = 0; line < 8; ++line) {
                uint32_t column;
                for (column = 0; column < width; ++column) {
                    // mcus_line_array[line * mcus_line_array_width + column] = fgetc(image);
                    // mcus_line_array[8 * (line + column)] = fgetc(image);
                    mcus_line_array[(column / 8) * 64 + line * 8 + column % 8] = fgetc(image);
                }
                // Troncature à droite possible que sur la dernière colonne de MCU
                if (tronc_right) {
                    uint8_t column_index = column % 8;
                    for (uint8_t column_offset = column_index; column_offset < 8; ++column_offset) {
                        // On copie la valeur précédente pour remplir le reste de la ligne
                        // mcus_line_array[line * mcus_line_array_width + column_offset] = mcus_line_array[line * mcus_line_array_width + column_offset - 1];
                        // mcus_line_array[8 * (line + column_offset)] = mcus_line_array[8 * (line + column_offset) - 1];
                        uint32_t row_in_last_mcu = (nb_mcu_line - 1) * 64 + line * 8;
                        mcus_line_array[row_in_last_mcu + column_offset] = mcus_line_array[row_in_last_mcu + column_index - 1];
                    }
                }
            }
        }

        // print MCUs line
        for (uint32_t i_mcu = 0; i_mcu < nb_mcu_line; ++i_mcu) {
            // printf("mcu number %d (mcu line number %d)\n", mcu_index, mcu_line);
            printf("mcu number %d\n", mcu_index);
            for (uint8_t i = 0; i < 8; i++) {
                for (uint8_t j = 0; j < 8; j++) {
                    printf("%d ", mcus_line_array[i_mcu * 64 + i * 8 + j]);
                }
                printf("\n");
            }
            printf("\n\n");
            mcu_index++;
        }

        // TODO
        // Call GPU
        // encoding(mcus_line_array, nb_mcu_line, true);
        // Take result from GPU
        // Call coding from results of GPU
        // for (uint8_t mcu_index = 0; mcu_index < nb_mcu_line; ++mcu_index) {
        //     coding(mcus_line_array + mcu_index * 64, ht_dc, ht_ac, stream, predicator, index);
        // }
    }

    // /* On parcourt successivement les différentes MCUs (ligne par ligne et de gauche à droite). */
    // for (uint32_t i = 0; i < nb_mcu_column; i++) {
        // for (uint32_t j = 0; j < nb_mcu_line; j++) {
            // if (tronc_line && i == line_mcu - 1) {
                // tronc_line_mcu = 1;
            // } else {
                // tronc_line_mcu = 0;
            // }
            // if (tronc_column && j == column_mcu - 1) {
                // tronc_column_mcu = 1;
            // } else {
                // tronc_column_mcu = 0;
            // }
            // /* On parcourt successivement les différents pixels d'une MCU (ligne par ligne et de gauche à droite). */
            // for (uint8_t k = 0; k < 8; k++) {
                // // printf("k = %i\n", k);
                // for (uint8_t l = 0; l < 8; l++) {
                    // /* Cas troncature à droite qui concerne le pixel traité. */
                    // if (tronc_column_mcu && l >= width_remainder) {
                        // /* Cas troncature à droite et troncature en bas qui concernent le pixel traité. */
                        // if (tronc_line_mcu && k >= height_remainder) {
                            // mcu[k][l] = mcu[height_remainder - 1][l]; 
                        // } else {
                            // mcu[k][l] = mcu[k][width_remainder - 1];
                        // }
                    // /* Cas troncature en bas qui concerne le pixel traité. */
                    // } else if (tronc_line_mcu && k >= height_remainder) {
                        // mcu[k][l] = mcu[height_remainder - 1][l];
                        // fseek(image, 1, SEEK_CUR); // fseek permet de déplacer le curseur où on le souhaite dans le fichier.
                    // /* Pas de troncature impactant le pixel traité. */ 
                    // } else {
                        // // printf("l = %i\n", l);
                        // // printf("%ld\n", ftell(image));
                        // mcu[k][l] = fgetc(image);
                        // // printf("%ld\n", ftell(image));
                    // }
                // }
                // /* Si on n'a pas atteint la dernière ligne de la MCU, on passe à la ligne suivante. */
                // if (k < 7) {
                    // if (!tronc_line_mcu && !tronc_column_mcu) {
                        // fseek(image, width - 8, SEEK_CUR);
                    // } else if (!tronc_line_mcu && tronc_column_mcu) {
                        // fseek(image, width - width_remainder, SEEK_CUR);
                    // } else if (tronc_line_mcu && !tronc_column_mcu) {
                        // if (k < height_remainder - 1) {
                            // fseek(image, width - 8, SEEK_CUR);
                        // } else {
                            // fseek(image, -8, SEEK_CUR);
                        // }
                    // } else {
                        // if (k < height_remainder - 1) {
                            // fseek(image, width - width_remainder, SEEK_CUR);
                        // } else {
                            // fseek(image, - width_remainder, SEEK_CUR);
                        // }
                    // }
                // }
            // }
            // dct_loeffler(mcu, mcu_array); // On transforme la MCU en tableau, en appliquant la DCT.
            // // dct_faster_loeffler(mcu, mcu_array);
            // // print_array_16(mcu_array);
            // quantify(mcu_array, true); // On applique la quantification.
            // // print_array_16(mcu_array);
            // coding(mcu_array, ht_dc, ht_ac, stream, predicator, index); // On encode la MCU.

            // /* Si on n'est pas sur une MCU de la dernière colonne */
            // if (j != column_mcu - 1) {
                // if (!tronc_line_mcu) {
                    // long offset = -7 * (long)width;
                    // fseek(image, offset, SEEK_CUR);
                // } else { // Si la MCU est tronquée en bas, on ne doit pas remonter le curseur trop haut. 
                    // long offset = (1 - (long) height_remainder) * (long)width;
                    // fseek(image, offset, SEEK_CUR);
                // }
            // }
        // }
    // }
    /* On libère tous les espaces mémoire alloués. */
    free(mcus_line_array);
    free(predicator);
    free(index);
}






/* Cette fonction alloue une matrice de la taille de MCU/bloc voulue */
static uint8_t **alloc_mcu(uint8_t width, uint8_t heigth)
{
    uint8_t **mcu = malloc(heigth * sizeof(uint8_t *));
    for (int row = 0; row < heigth; row++) {
        mcu[row] = malloc(width * sizeof(uint8_t));
    }
    return mcu;
}

/* Cette fonction libère l'espace alloué par un bloc ou une MCU. */
static void free_mcu(uint8_t **mcu, uint8_t heigth) {
    for (int row = 0; row < heigth; row++) {
        free(mcu[row]);
    }
    free(mcu);
}

/*
    Dans cette fonction qui s'occupe des images en noir et blanc,
    on traite chaque MCU intégralement, en effectuant les transformations successives,
    avant de passer à la suivante.
*/
void treat_image_grey_o(FILE *image, uint32_t width, uint32_t height, struct huff_table *ht_dc, struct huff_table *ht_ac, struct bitstream *stream)
{
    printf("----- OLD treat image grey -----\n");

    /* On alloue tous les espaces mémoire nécessaires. */
    uint16_t *index = malloc(sizeof(uint16_t));
    uint8_t **mcu = alloc_mcu(8, 8);
    int16_t *predicator = calloc(1, sizeof(int16_t));
    int16_t *mcu_array = malloc(64 * sizeof(int16_t));
    uint32_t line_mcu = height / 8; // Nombre de lignes de MCUs.
    uint32_t column_mcu = width / 8; // Nombre de colonnes de MCUs.
    bool tronc_line = 0; // Indique si il y a une troncature en bas.
    bool tronc_column = 0; // Indique si il y a une troncature à droite.
    uint8_t height_remainder = height % 8; // Nombre de lignes dépassant (en cas de troncature).
    uint8_t width_remainder = width % 8; // Nombre de colonnes dépassant (en cas de troncature).
    if (height_remainder != 0) {
        line_mcu++; // On rajoute une ligne de MCUs.
        tronc_line = 1; // Il y a troncature en bas.
    }
    if (width_remainder != 0) {
        column_mcu++; // On rajoute une colonne de MCUs.
        tronc_column = 1; // Il y a troncature à droite.
    }
    printf("Tronc_line = %u, Tronc_column = %u\n", tronc_line, tronc_column);
    printf("Nombre de mcu par ligne : %i, Nombre de mcu par colonne : %i\n", column_mcu, line_mcu);
    bool tronc_line_mcu; // Indique si il y a une troncature en bas dans la MCU courante.
    bool tronc_column_mcu; // Indique si il y a une troncature à droite dans la MCU courante.
    /* On parcourt successivement les différentes MCUs (ligne par ligne et de gauche à droite). */

    uint32_t mcu_index = 0; // absolute in whole image

    for (uint32_t i = 0; i < line_mcu; i++) {
        // printf("i = %i\n", i);
        for (uint32_t j = 0; j < column_mcu; j++) {
            // printf("j = %i\n", j);
            if (tronc_line && i == line_mcu - 1) {
                tronc_line_mcu = 1;
            } else {
                tronc_line_mcu = 0;
            }
            if (tronc_column && j == column_mcu - 1) {
                tronc_column_mcu = 1;
            } else {
                tronc_column_mcu = 0;
            }
            /* On parcourt successivement les différents pixels d'une MCU (ligne par ligne et de gauche à droite). */
            for (uint8_t k = 0; k < 8; k++) {
                // printf("k = %i\n", k);
                for (uint8_t l = 0; l < 8; l++) {
                    /* Cas troncature à droite qui concerne le pixel traité. */
                    if (tronc_column_mcu && l >= width_remainder) {
                        /* Cas troncature à droite et troncature en bas qui concernent le pixel traité. */
                        if (tronc_line_mcu && k >= height_remainder) {
                            mcu[k][l] = mcu[height_remainder - 1][l]; 
                        } else {
                            mcu[k][l] = mcu[k][width_remainder - 1];
                        }
                    /* Cas troncature en bas qui concerne le pixel traité. */
                    } else if (tronc_line_mcu && k >= height_remainder) {
                        mcu[k][l] = mcu[height_remainder - 1][l];
                        fseek(image, 1, SEEK_CUR); // fseek permet de déplacer le curseur où on le souhaite dans le fichier.
                    /* Pas de troncature impactant le pixel traité. */ 
                    } else {
                        // printf("l = %i\n", l);
                        // printf("%ld\n", ftell(image));
                        mcu[k][l] = fgetc(image);
                        // printf("%ld\n", ftell(image));
                    }
                }
                /* Si on n'a pas atteint la dernière ligne de la MCU, on passe à la ligne suivante. */
                if (k < 7) {
                    if (!tronc_line_mcu && !tronc_column_mcu) {
                        fseek(image, width - 8, SEEK_CUR);
                    } else if (!tronc_line_mcu && tronc_column_mcu) {
                        fseek(image, width - width_remainder, SEEK_CUR);
                    } else if (tronc_line_mcu && !tronc_column_mcu) {
                        if (k < height_remainder - 1) {
                            fseek(image, width - 8, SEEK_CUR);
                        } else {
                            fseek(image, -8, SEEK_CUR);
                        }
                    } else {
                        if (k < height_remainder - 1) {
                            fseek(image, width - width_remainder, SEEK_CUR);
                        } else {
                            fseek(image, - width_remainder, SEEK_CUR);
                        }
                    }
                }
            }
            // dct_loeffler(mcu, mcu_array); // On transforme la MCU en tableau, en appliquant la DCT.
            // dct_faster_loeffler(mcu, mcu_array);
            // print_array_16(mcu_array);
            // quantify(mcu_array, true); // On applique la quantification.
            // print_array_16(mcu_array);
            // coding(mcu_array, ht_dc, ht_ac, stream, predicator, index); // On encode la MCU.

            // print MCUs
            printf("mcu number %d\n", mcu_index);
            for (uint8_t i = 0; i < 8; i++) {
                for (uint8_t j = 0; j < 8; j++) {
                    printf("%d ", mcu[i][j]);
                }
                printf("\n");
            }
            printf("\n\n");
            mcu_index++;

            /* Si on n'est pas sur une MCU de la dernière colonne */
            if (j != column_mcu - 1) {
                if (!tronc_line_mcu) {
                    long offset = -7 * (long)width;
                    fseek(image, offset, SEEK_CUR);
                } else { // Si la MCU est tronquée en bas, on ne doit pas remonter le curseur trop haut. 
                    long offset = (1 - (long) height_remainder) * (long)width;
                    fseek(image, offset, SEEK_CUR);
                }
            }
        }
    }
    /* On libère tous les espaces mémoire alloués. */
    free_mcu(mcu, 8);
    free(mcu_array);
    free(predicator);
    free(index);
}