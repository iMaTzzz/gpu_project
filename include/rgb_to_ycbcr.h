#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* 
 * Convertir pixel RGB à YCbCr

 * Parameter entrées : pixel red, green et blue et un enum color_component Y, Cb, Cr 
 
 * Return : Y ou Cb ou Cr selon le enum
*/
extern void rgb_to_ycbcr(uint8_t red, uint8_t green, uint8_t blue, uint8_t *Y, uint8_t *Cb, uint8_t *Cr);

#ifdef __cplusplus
}
#endif
