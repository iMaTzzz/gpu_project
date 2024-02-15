#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "quantification.h"
#include "coding.h"
#include "dct_cpu.h"
#include "jpeg_writer.h"
#include "rgb_to_ycbcr.h"
#include "downsampling.h"
#include "bitstream.h"

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
void treat_image_grey_cpu(FILE *image, uint32_t width, uint32_t height, struct huff_table *ht_dc, struct huff_table *ht_ac, struct bitstream *stream)
{
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
    // printf("Tronc_line = %u, Tronc_column = %u\n", tronc_line, tronc_column);
    // printf("Nombre de mcu par ligne : %i, Nombre de mcu par colonne : %i\n", column_mcu, line_mcu);
    bool tronc_line_mcu; // Indique si il y a une troncature en bas dans la MCU courante.
    bool tronc_column_mcu; // Indique si il y a une troncature à droite dans la MCU courante.
    /* On parcourt successivement les différentes MCUs (ligne par ligne et de gauche à droite). */
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
            cpu_dct_loeffler(mcu, mcu_array); // On transforme la MCU en tableau, en appliquant la DCT.
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
    free_mcu(mcu, 8);
    free(mcu_array);
    free(predicator);
    free(index);
}

/*
    Dans cette fonction qui s'occupe des images en couleur,
    on traite chaque MCU intégralement, en effectuant les transformations successives (avec un éventuel sous-échantillonnage),
    avant de passer à la suivante. 
*/
void treat_image_color_cpu(FILE *image, uint32_t width, uint32_t height, struct huff_table *ht_dc_Y, 
                        struct huff_table *ht_ac_Y, struct huff_table *ht_dc_C, struct huff_table *ht_ac_C, 
                        struct bitstream *stream, uint8_t h1, uint8_t v1, uint8_t h2, uint8_t v2, uint8_t h3, uint8_t v3)
{
    uint8_t height_mcu = 8*v1; // Nombre de pixels sur une colonne d'une MCU.
    uint8_t width_mcu = 8*h1;  // Nombre de pixels sur une ligne d'une MCU.
    /* On alloue tous les espaces mémoire nécessaires. */
    uint16_t *index = malloc(sizeof(uint16_t));
    uint8_t **mcu_Y = alloc_mcu(width_mcu, height_mcu); // On utilise une MCU différente pour chaque composante.  
    uint8_t **mcu_Cb = alloc_mcu(width_mcu, height_mcu);
    uint8_t **mcu_Cr = alloc_mcu(width_mcu, height_mcu);
    uint8_t **mcu_Cb_sampling = alloc_mcu(h2*8, v2*8); 
    uint8_t **mcu_Cr_sampling = alloc_mcu(h3*8, v3*8);
    uint8_t **bloc = alloc_mcu(8, 8); 
    int16_t *predicator_Y = calloc(1, sizeof(int16_t)); // On utilise un prédicateur par composante.
    int16_t *predicator_Cb = calloc(1, sizeof(int16_t));
    int16_t *predicator_Cr = calloc(1, sizeof(int16_t));
    int16_t *bloc_array = malloc(64 * sizeof(int16_t));
    uint32_t line_mcu = height / height_mcu; // Nombre de lignes de MCUs.
    uint32_t column_mcu = width / width_mcu; // Nombre de colonnes de MCUs.
    uint8_t height_remainder = height % (height_mcu); // Nombre de lignes dépassant (en cas de troncature).
    uint8_t width_remainder = width % (width_mcu); // Nombre de colonnes dépassant (en cas de troncature).
    bool tronc_line = 0; // Indique si il y a une troncature en bas.
    bool tronc_column = 0; // Indique si il y a une troncature à droite.
    if (height_remainder != 0) {
        line_mcu++; // On rajoute une ligne de MCUs.
        tronc_line = 1; // Il y a troncature en bas.
    }
    if (width_remainder != 0) {
        column_mcu++; // On rajoute une colonne de MCUs.
        tronc_column = 1; // Il y a troncature à droite.
    }
    // printf("Tronc_line = %u, Tronc_column = %u\n", tronc_line, tronc_column);
    // printf("Nombre de mcu par ligne : %i, Nombre de mcu par colonne : %i\n", column_mcu, line_mcu);
    bool tronc_line_mcu; // Indique si il y a une troncature en bas dans la MCU courante.
    bool tronc_column_mcu; // Indique si il y a une troncature à droite dans la MCU courante.
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t Y;
    uint8_t Cb;
    uint8_t Cr;
    /* On parcourt successivement les différentes MCUs (ligne par ligne et de gauche à droite). */
    for (uint32_t i = 0; i < line_mcu; i++) {
        //printf("i = %i\n", i);
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
            for (uint8_t k = 0; k < height_mcu; k++) {
                //printf("k = %i\n", k);
                for (uint8_t l = 0; l < width_mcu; l++) {
                /* Cas troncature à droite qui concerne le pixel traité. */
                    if (tronc_column_mcu && l >= width_remainder) {
                        /* Cas troncature à droite et troncature en bas qui concernent le pixel traité. */
                        if (tronc_line_mcu && k >= height_remainder) {
                            mcu_Y[k][l] = mcu_Y[height_remainder - 1][l]; 
                            mcu_Cb[k][l] = mcu_Cb[height_remainder - 1][l]; 
                            mcu_Cr[k][l] = mcu_Cr[height_remainder - 1][l]; 
                        } else {
                            mcu_Y[k][l] = mcu_Y[k][width_remainder - 1];
                            mcu_Cb[k][l] = mcu_Cb[k][width_remainder - 1];
                            mcu_Cr[k][l] = mcu_Cr[k][width_remainder - 1];
                        }
                    /* Cas troncature en bas qui concerne le pixel traité. */
                    } else if (tronc_line_mcu && k >= height_remainder) {
                        mcu_Y[k][l] = mcu_Y[height_remainder - 1][l];
                        mcu_Cb[k][l] = mcu_Cb[height_remainder - 1][l];
                        mcu_Cr[k][l] = mcu_Cr[height_remainder- 1][l];
                        fseek(image, 3, SEEK_CUR); // fseek permet de déplacer le curseur où on le souhaite dans le fichier. 
                    /* Pas de troncature impactant le pixel traité. */
                    } else {
                        //printf("l = %i\n", l);
                        red = fgetc(image);
                        green = fgetc(image);
                        blue = fgetc(image);
                        rgb_to_ycbcr(red, green, blue, &Y, &Cb, &Cr);
                        mcu_Y[k][l] = Y;
                        mcu_Cb[k][l] = Cb;
                        mcu_Cr[k][l] = Cr;
                        // printf("%ld\n", ftell(image));
                    }
                }
                /* Si on n'a pas atteint la dernière ligne de la MCU, on passe à la ligne suivante. */
                if (k < height_mcu - 1) {
                    if (!tronc_line_mcu && !tronc_column_mcu) {
                        fseek(image, 3 * (width - width_mcu), SEEK_CUR);
                    } else if (!tronc_line_mcu && tronc_column_mcu) {
                        fseek(image, 3 * (width - width_remainder), SEEK_CUR);
                    } else if (tronc_line_mcu && !tronc_column_mcu) {
                        if (k < height_remainder - 1) {
                            fseek(image, 3 * (width - width_mcu), SEEK_CUR);
                        } else {
                            fseek(image, -3 * width_mcu, SEEK_CUR);
                        }
                    } else {
                        if (k < height_remainder - 1) {
                            fseek(image, 3 * (width - width_remainder), SEEK_CUR);
                        } else {
                            fseek(image, - 3 * width_remainder, SEEK_CUR);
                        }
                    }
                }
            }
            // print_matrix_mcu(mcu_Y, width_mcu, heigth_mcu);
            /* On encode d'abord la composante Y */
            for (uint8_t k = 0; k < v1; k++) {
                for (uint8_t l = 0; l < h1; l++) {
                    /* On traite la MCU bloc par bloc. */ 
                    for (uint8_t m = 0; m < 8; m++) {
                        for (uint8_t n = 0; n < 8; n++) {                            
                            bloc[m][n] = mcu_Y[8*k+m][8*l+n];
                        }
                    }
                    cpu_dct_loeffler(bloc, bloc_array); // On transforme le bloc en tableau, en appliquant la DCT.
                    // print_array_16(mcu_array);
                    quantify(bloc_array, true); // On applique la quantification au bloc.
                    // print_array_16(mcu_array);
                    coding(bloc_array, ht_dc_Y, ht_ac_Y, stream, predicator_Y, index); // On encode le bloc.
                }
            }
            
            /* Puis on encode la composante Cb */
            if (h2 != h1 || v2 != v1) {
                downsampling(mcu_Cb, mcu_Cb_sampling, h2, v2, h1, v1); // On sous-échantillonne si nécessaire.
            }
            for (uint8_t k = 0; k < v2; k++) {
                for (uint8_t l = 0; l < h2; l++) {
                    /* On traite la MCU bloc par bloc. */ 
                    for (uint8_t m = 0; m < 8; m++) {
                        for (uint8_t n = 0; n < 8; n++) {
                            if (h2 != h1 || v2 != v1) {
                                bloc[m][n] = mcu_Cb_sampling[8*k+m][8*l+n];
                            } else {
                                bloc[m][n] = mcu_Cb[8*k+m][8*l+n];
                            }
                        }
                    }
                    //print_matrix_8(mcu_Cb);
                    cpu_dct_loeffler(bloc, bloc_array); // On transforme le bloc en tableau, en appliquant la DCT.
                    // print_array_16(mcu_array);
                    quantify(bloc_array, false); // On applique la quantification au bloc.
                    // print_array_16(mcu_array);
                    coding(bloc_array, ht_dc_C, ht_ac_C, stream, predicator_Cb, index); // On encode le bloc.
                }
            }


            /* Enfin on encode la composante Cr */
            if (h3 != h1 || v3 != v1) {
                downsampling(mcu_Cr, mcu_Cr_sampling, h3, v3, h1, v1); // On sous-échantillonne si nécessaire.
            }
            for (uint8_t k = 0; k < v3; k++) {
                for (uint8_t l = 0; l < h3; l++) {
                    /* On traite la MCU bloc par bloc. */ 
                    for (uint8_t m = 0; m < 8; m++) {
                        for (uint8_t n = 0; n < 8; n++) {
                            if (h3 != h1 || v3 != v1) {
                                bloc[m][n] = mcu_Cr_sampling[8*k+m][8*l+n];
                            } else {
                                bloc[m][n] = mcu_Cr[8*k+m][8*l+n];
                            }
                        }
                    }
                    //print_matrix_8(mcu_Cb);
                    cpu_dct_loeffler(bloc, bloc_array); // On transforme le bloc en tableau, en appliquant la DCT.
                    // print_array_16(mcu_array);
                    quantify(bloc_array, false); // On applique la quantification au bloc.
                    // print_array_16(mcu_array);
                    coding(bloc_array, ht_dc_C, ht_ac_C, stream, predicator_Cr, index); // On encode le bloc.
                }
            }
            
            /* Si on n'est pas sur une MCU de la dernière colonne */
            if (j != column_mcu - 1) {
                if (!tronc_line_mcu) {
                    long offset = (1 - height_mcu) * (long)width;
                    fseek(image, 3 * offset, SEEK_CUR); // -2240 = -7 * width
                } else { // Si la MCU est tronquée en bas, on ne doit pas remonter le curseur trop haut. 
                    long offset = (1 - (long) height_remainder) * (long)width;
                    fseek(image, 3 * offset, SEEK_CUR); 
                }
            }
        }
    }
    /* On libère tous les espaces mémoire alloués. */
    free_mcu(mcu_Y, height_mcu);
    free_mcu(mcu_Cb, height_mcu);
    free_mcu(mcu_Cr, height_mcu);
    free(bloc_array);
    free(predicator_Y);
    free(predicator_Cb);
    free(predicator_Cr);
    free(index);
    free_mcu(bloc, 8);
    free_mcu(mcu_Cb_sampling, v2*8);
    free_mcu(mcu_Cr_sampling, v3*8);
}

