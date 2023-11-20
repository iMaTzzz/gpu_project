#include <stdint.h>

/* 
 * Convertir pixel RGB à YCbCr

 * Parameter entrées : pixel red, green et blue et un enum color_component Y, Cb, Cr 
 
 * Return : Y ou Cb ou Cr selon le enum
*/
extern uint8_t rgb_to_ycbcr(uint8_t red, uint8_t green, uint8_t blue, enum color_component cc);