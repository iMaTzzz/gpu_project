#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "zigzag.h"

/* Constantes utilisées dans les deux versions des algorithmes de Loeffler */
#define VALUE_0_765366865 0.765366865
#define VALUE_1_847759065 1.847759065
#define VALUE_1_175875602 1.175875602
#define VALUE_0_390180644 0.390180644
#define VALUE_1_961570560 1.961570560
#define VALUE_0_899976223 0.899976223
#define VALUE_1_501321110 1.501321110
#define VALUE_0_298631336 0.298631336
#define VALUE_2_562915447 2.562915447
#define VALUE_3_072711026 3.072711026
#define VALUE_2_053119869 2.053119869
#define VALUE_0_707106781 0.707106781
#define VALUE_0_382683433 0.382683433
#define VALUE_0_541196100 0.541196100
#define VALUE_1_306562965 1.306562965
#define VALUE_0_275899379 0.275899379
#define VALUE_1_387039845 1.387039845
#define VALUE_1_414213562 1.414213562
#define VALUE_0_555570233 0.555570233
#define VALUE_0_831469612 0.831469612
#define VALUE_0_195090322 0.195090322
#define VALUE_0_980785280 0.980785280
#define VALUE_1_662939225 1.662939225

/* 
Algorithme de Loeffler : C'est un algorithme permettant de calculer la 1D-DCT sur 8 pixels.
Ainsi, pour calculer la 2D-DCT pour un bloc 8x8, il suffit d'appliquer l'algorithme de Loeffler
à chaque ligne puis à chaque colonne et on divise chaque coefficient par 8. On obtient donc les
coefficients voulus.
*/


/* Fonction qui applique la transformée 2D grâce à l'algorithme de Loeffler non optimisé mais précis */
void dct_loeffler(uint8_t **bloc_spatiale, int16_t *mcu_array)
{
  int32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
  int32_t tab_tmp[8][8];
  for (uint8_t row = 0; row < 8; row++) {
    tmp0 = (int32_t) (bloc_spatiale[row][0] + bloc_spatiale[row][7] - 256);
    tmp1 = (int32_t) (bloc_spatiale[row][1] + bloc_spatiale[row][6] - 256);
    tmp2 = (int32_t) (bloc_spatiale[row][2] + bloc_spatiale[row][5] - 256);
    tmp3 = (int32_t) (bloc_spatiale[row][3] + bloc_spatiale[row][4] - 256);

    tmp4 = tmp0 + tmp3;
    tmp5 = tmp1 + tmp2;
    tmp6 = tmp0 - tmp3;
    tmp7 = tmp1 - tmp2;

    tab_tmp[row][0] = tmp4 + tmp5;
    tab_tmp[row][4] = tmp4 - tmp5;

    tmp8 = (tmp6 + tmp7) * VALUE_0_541196100;

    tab_tmp[row][2] = tmp8 + tmp6 * VALUE_0_765366865;
    tab_tmp[row][6] = tmp8 - tmp7 * VALUE_1_847759065;

    tmp0 = (int32_t) (bloc_spatiale[row][0] - bloc_spatiale[row][7]);
    tmp1 = (int32_t) (bloc_spatiale[row][1] - bloc_spatiale[row][6]);
    tmp2 = (int32_t) (bloc_spatiale[row][2] - bloc_spatiale[row][5]);
    tmp3 = (int32_t) (bloc_spatiale[row][3] - bloc_spatiale[row][4]);

    tmp6 = tmp0 + tmp2;
    tmp7 = tmp1 + tmp3;

    tmp8 = (tmp6 + tmp7) * VALUE_1_175875602;
    tmp6 = tmp6 * (- VALUE_0_390180644) + tmp8;
    tmp7 = tmp7 * (- VALUE_1_961570560) + tmp8;
    
    tmp8 = (tmp0 + tmp3) * (- VALUE_0_899976223);
    tmp0 = tmp0 * VALUE_1_501321110 + tmp8 + tmp6;
    tmp3 = tmp3 * VALUE_0_298631336 + tmp8 + tmp7;

    tmp8 = (tmp1 + tmp2) * (- VALUE_2_562915447);
    tmp1 = tmp1 * VALUE_3_072711026 + tmp8 + tmp7;
    tmp2 = tmp2 * VALUE_2_053119869 + tmp8 + tmp6;

    tab_tmp[row][1] = tmp0;
    tab_tmp[row][3] = tmp1;
    tab_tmp[row][5] = tmp2;
    tab_tmp[row][7] = tmp3;
  }
  for (uint8_t column = 0; column < 8; column++) {
    tmp0 = tab_tmp[0][column] + tab_tmp[7][column];
    tmp1 = tab_tmp[1][column] + tab_tmp[6][column];
    tmp2 = tab_tmp[2][column] + tab_tmp[5][column];
    tmp3 = tab_tmp[3][column] + tab_tmp[4][column];

    tmp4 = tmp0 + tmp3;
    tmp5 = tmp1 + tmp2;
    tmp6 = tmp0 - tmp3;
    tmp7 = tmp1 - tmp2;

    tmp0 = tab_tmp[0][column] - tab_tmp[7][column];
    tmp1 = tab_tmp[1][column] - tab_tmp[6][column];
    tmp2 = tab_tmp[2][column] - tab_tmp[5][column];
    tmp3 = tab_tmp[3][column] - tab_tmp[4][column];

    mcu_array[matrix_zig_zag[0][column]] = ((int16_t) (tmp4 + tmp5) >> 3); 
    mcu_array[matrix_zig_zag[4][column]] = ((int16_t) (tmp4 - tmp5) >> 3); 

    tmp8 = (tmp6 + tmp7) * VALUE_0_541196100;
    
    mcu_array[matrix_zig_zag[2][column]] = ((int16_t) (tmp8 + tmp6 * VALUE_0_765366865) >> 3);
    mcu_array[matrix_zig_zag[6][column]] = ((int16_t) (tmp8 - tmp7 * VALUE_1_847759065) >> 3);

    tmp6 = tmp0 + tmp2;
    tmp7 = tmp1 + tmp3;

    tmp8 = (tmp6 + tmp7) * VALUE_1_175875602;
    tmp6 = tmp6 * (- VALUE_0_390180644) + tmp8;
    tmp7 = tmp7 * (- VALUE_1_961570560) + tmp8;

    tmp8 = (tmp0 + tmp3) * (- VALUE_0_899976223);
    tmp0 = tmp0 * VALUE_1_501321110 + tmp8 + tmp6;
    tmp3 = tmp3 * VALUE_0_298631336 + tmp8 + tmp7;
    
    tmp8 = (tmp1 + tmp2) * (- VALUE_2_562915447);
    tmp1 = tmp1 * VALUE_3_072711026 + tmp8 + tmp7;
    tmp2 = tmp2 * VALUE_2_053119869 + tmp8 + tmp6;

    mcu_array[matrix_zig_zag[1][column]] = (int16_t) (tmp0 >> 3);
    mcu_array[matrix_zig_zag[3][column]] = (int16_t) (tmp1 >> 3);
    mcu_array[matrix_zig_zag[5][column]] = (int16_t) (tmp2 >> 3);
    mcu_array[matrix_zig_zag[7][column]] = (int16_t) (tmp3 >> 3);
  }
}
