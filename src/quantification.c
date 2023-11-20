#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "qtables.h"
#include <stdio.h>

/*
    Quantifier le bloc 8x8 (qui a été transformé en array zigzag)
    Entrées : array int16_t de taille 64 et un bool qui vérifier si on est dans le cas Y ou CbCr
    Sortie : Rien 
*/
void quantify(int16_t *array, bool luminance)
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
