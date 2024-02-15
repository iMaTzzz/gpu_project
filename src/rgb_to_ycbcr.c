#include <stdint.h>
#include <assert.h>
#include "jpeg_writer.h"
#include <stdlib.h>
#include <stdio.h>
#include "rgb_to_ycbcr.h"

/* 
 * Convertir pixel RGB à YCbCr

 * Parameter entrées : Pixels red, green et blue et des pointeurs vers les pixels Y, Cb, Cr 
 
 * Return : Void (Modifie en place Y, Cb et Cr)
*/
void rgb_to_ycbcr(uint8_t red, uint8_t green, uint8_t blue, uint8_t *Y, uint8_t *Cb, uint8_t *Cr)
{
    *Y = (uint8_t)(0.299 * red + 0.587 * green + 0.114 * blue);
    *Cb = (uint8_t)(-0.1687 * red - 0.3313 * green + 0.5 * blue + 128);
    *Cr = (uint8_t)(0.5 * red - 0.4187 * green - 0.0813 * blue + 128);
}
