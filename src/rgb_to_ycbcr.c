#include <stdint.h>
#include <assert.h>
#include "jpeg_writer.h"
#include <stdlib.h>
#include <stdio.h>
#include "rgb_to_ycbcr.h"

/* 
 * Convertir pixel RGB à YCbCr

 * Parameter entrées : pixel red, green et blue et un enum color_component Y, Cb, Cr 
 
 * Return : Y ou Cb ou Cr selon le enum
*/
uint8_t rgb_to_ycbcr(uint8_t red, uint8_t green, uint8_t blue, enum color_component cc)
{
    switch (cc) {
        case Y:
            return (uint8_t)(0.299 * red + 0.587 * green + 0.114 * blue);
        case Cb:
            return (uint8_t)(-0.1687 * red - 0.3313 * green + 0.5 * blue + 128);
        case Cr:
            return (uint8_t)(0.5 * red - 0.4187 * green - 0.0813 * blue + 128);
        default:
            break;
    }
    exit(EXIT_FAILURE);
}
