#include <stdint.h>
#include <stdio.h>

// Pour tester
/*
void print_matrix(uint8_t **matrix, uint8_t h, uint8_t v)
{
    for (int i = 0; i < v; i++) {
        for (int j = 0; j < h; j++) {
            printf("%04hhx\t", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
*/


/*
    Fonction sous-échantillonnage généralisé

    Entrées : une matrice à faire downsampling -> mcu_in
              une matrice déja downsampling -> mcu_out
              valeur horizontal h (soit h2 ou h3) pour CbCr
              valeur vertical v (soit v2 ou v3)   pour CbCr
              valeur horizontal h1                pour Y
              valeur vertical v1                  pour Y

    Sortie : Rien
*/
void downsampling(uint8_t **mcu_in, uint8_t **mcu_out, uint8_t h, uint8_t v, uint8_t h1, uint8_t v1)
{
    // On compte le quotient le horizontale et vertical
    uint8_t horizontal_quotient = h1/h;
    uint8_t vertical_quotient = v1/v;
    // nb_pixel = combien pixel à sommer et faire le moyenne
    uint8_t nb_pixel = horizontal_quotient * vertical_quotient;
    // On parcours la dimension de l'image (h1 et v1)
    for (uint8_t line = 0; line < 8*v; line++) {
        for (uint8_t column = 0; column < 8*h; column++) {
            uint16_t sum = 0;
            // On parcours le nb_pixel pixel et cherche la moyenne
            for (uint8_t i = vertical_quotient * line; i < vertical_quotient * (line + 1); i++) {
                for (uint8_t j = horizontal_quotient * column; j < horizontal_quotient * (column + 1); j++) {
                    sum += mcu_in[i][j];
                }
            }
            mcu_out[line][column] = sum/nb_pixel;
        }
    }
}

