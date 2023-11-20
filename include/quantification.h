#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

/*
    Quantifier le bloc 8x8 (qui a été transformé en array zigzag)
    Entrées : array int16_t de taille 64 et un bool qui vérifier si on est dans le cas Y ou CbCr
    Sortie : Rien 
*/
extern void quantify(int16_t *bloc, bool luminance);