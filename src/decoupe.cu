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

/* Cette fonction alloue une matrice de la taille de MCU/bloc voulue */
static uint8_t ***alloc_mcus_line_matrix(uint32_t nb_mcu_line, uint8_t width, uint8_t heigth)
{
    uint8_t ***mcus_line = malloc(nb_mcu_line * sizeof(uint8_t **));
    for (uint32_t mcu = 0; mcu < nb_mcu_line; ++mcu) {
        mcus_line[mcu] = malloc(heigth * sizeof(uint8_t *));
        for (uint8_t row = 0; row < heigth; ++row) {
            mcus_line[mcu][row] = malloc(width * sizeof(uint8_t));
        }
    }
    return mcus_line;
}

/* Cette fonction libère l'espace alloué par un bloc ou une MCU. */
static void free_mcus_line_matrix(uint8_t ***mcus_line_matrix, uint32_t nb_mcu_line, uint8_t heigth)
{
    for (uint32_t mcu = 0; mcu < nb_mcu_line; ++mcu) {
        for (uint8_t row = 0; row < heigth; ++row) {
            free(mcus_line_matrix[mcu][row]);
        }
        free(mcus_line_matrix[mcu]);
    }
    free(mcus_line_matrix);
}

static int16_t **alloc_mcus_line_array(uint32_t nb_mcu_line, uint8_t width, uint8_t heigth)
{
    int16_t **mcus_line_array = malloc(nb_mcu_line * sizeof(int16_t *));
    for (uint32_t mcu = 0; mcu < nb_mcu_line; ++mcu) {
        mcus_line_array[mcu] = malloc(width*heigth*sizeof(int16_t));
    }
    return mcus_line_array;
}

static void free_mcus_line_array(int16_t **mcus_line_array, uint32_t nb_mcu_line)
{
    for (uint32_t mcu = 0; mcu < nb_mcu_line; ++mcu) {
        free(mcus_line_array[mcu]);
    }
    free(mcus_line_array);
}

/*
    Dans cette fonction qui s'occupe des images en noir et blanc,
    on traite chaque MCU intégralement, en effectuant les transformations successives,
    avant de passer à la suivante. 
*/
extern "C"
void treat_image_grey(FILE *image, uint32_t width, uint32_t height, struct huff_table *ht_dc, struct huff_table *ht_ac, struct bitstream *stream)
{
    /* On alloue tous les espaces mémoire nécessaires. */
    // TODO
    uint16_t *index = malloc(sizeof(uint16_t));
    // Idée faire une passe pour récupérer les predicateurs
    int16_t *predicator = calloc(1, sizeof(int16_t));

    uint32_t nb_mcu_column = height / 8; // Nombre de MCUs par colonne
    uint32_t nb_mcu_line = width / 8; // Nombre de MCUs par ligne
    uint8_t height_remainder = height % 8; // Nombre de lignes dépassant (en cas de troncature).
    uint8_t width_remainder = width % 8; // Nombre de colonnes dépassant (en cas de troncature).
    bool tronc_line = 0; // Indique si il y a une troncature en bas. 
    bool tronc_column = 0; // Indique si il y a une troncature à droite.
    if (height_remainder != 0) { 
        nb_mcu_column++; // On rajoute une ligne de MCUs.
        tronc_column = 1; // Il y a troncature en bas.
    }
    if (width_remainder != 0) {
        nb_mcu_line++; // On rajoute une colonne de MCUs.
        tronc_line = 1; // Il y a troncature à droite.
    }
    uint8_t ***mcus_line_matrix = alloc_mcus_line_matrix(nb_mcu_line, 8, 8);
    int16_t **mcus_line_array = alloc_mcus_line_array(nb_mcu_line, 8, 8);

    bool tronc_line_mcu; // Indique si il y a une troncature en bas dans la MCU courante.
    bool tronc_column_mcu; // Indique si il y a une troncature à droite dans la MCU courante.

    // TODO Gerer les troncatures
    for (uint32_t mcu_line = 0; mcu_line < nb_mcu_column; ++mcu_line) {
        for (uint8_t line = 0; line < 8; ++line) {
            for (uint32_t column = 0; column < width; ++column) {
                mcus_line_matrix[column/8][line][column%8] = fgetc(image);
            }
        }
        // TODO
        // Call GPU
        // Take result from GPU
        // Call coding from results of GPU
    }

    /* On parcourt successivement les différentes MCUs (ligne par ligne et de gauche à droite). */
    for (uint32_t i = 0; i < nb_mcu_column; i++) {
        for (uint32_t j = 0; j < nb_mcu_line; j++) {
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
            dct_loeffler(mcu, mcu_array); // On transforme la MCU en tableau, en appliquant la DCT.
            // dct_faster_loeffler(mcu, mcu_array);
            // print_array_16(mcu_array);
            quantify(mcu_array, true); // On applique la quantification.
            // print_array_16(mcu_array);
            coding(mcu_array, ht_dc, ht_ac, stream, predicator, index); // On encode la MCU.

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
    free_mcus_line_matrix(mcus_line_matrix, nb_mcu_line, 8);
    free_mcus_line_array(mcus_line_array, nb_mcu_line);
    free(predicator);
    free(index);
}