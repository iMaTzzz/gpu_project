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

/* Fonction qui applique la transformée 2D grâce à l'algorithme de Loeffler optimisé mais trop imprécis */
void dct_faster_loeffler(uint8_t **bloc_spatiale, int16_t *mcu_array)
{
    float a0, a1, a2, a3, a4, a5, a6, a7;
    float b0, b1, b2, b3, b4, b5, b6, b7;
    float c4, c5, c6, c7;
    float tmp0, tmp1, tmp2;
    float tab_tmp[8][8];
    for (uint8_t row = 0; row < 8; row++) {
        // Partie paire
        a0 = (float) (bloc_spatiale[row][0] + bloc_spatiale[row][7] - 256);
        a1 = (float) (bloc_spatiale[row][1] + bloc_spatiale[row][6] - 256);
        a2 = (float) (bloc_spatiale[row][2] + bloc_spatiale[row][5] - 256);
        a3 = (float) (bloc_spatiale[row][3] + bloc_spatiale[row][4] - 256);

        b0 = a0 + a3;
        b1 = a1 + a2;
        b2 = a1 - a2;
        b3 = a0 - a3;

        tab_tmp[row][0] = b0 + b1; 
        tab_tmp[row][4] = b0 - b1;

        tmp0 = b2 * VALUE_1_662939225;

        tab_tmp[row][2] = (b3 - b2) * VALUE_0_275899379 + tmp0;
        tab_tmp[row][6] = (b3 + b2) * (- VALUE_1_387039845) + tmp0;

        // Partie impaire

        a4 = (float) (bloc_spatiale[row][3] - bloc_spatiale[row][4]);
        a5 = (float) (bloc_spatiale[row][2] - bloc_spatiale[row][5]);
        a6 = (float) (bloc_spatiale[row][1] - bloc_spatiale[row][6]);
        a7 = (float) (bloc_spatiale[row][0] - bloc_spatiale[row][7]);

        tmp1 = a4 * VALUE_1_387039845;

        b4 = (a7 - a4) * VALUE_0_555570233 + tmp1;
        b7 = (a7 + a4) * (- VALUE_0_831469612) + tmp1;
        
        tmp2 = a5 * VALUE_1_175875602;

        b5 = (a6 - a5) * VALUE_0_195090322 + tmp2;
        b6 = (a6 + a5) * (- VALUE_0_980785280) + tmp2;

        c4 = (b4 + b6);
        c5 = (b7 - b5);
        c6 = (b4 - b6);
        c7 = (b7 + b5);

        tab_tmp[row][1] = c7 + c4;
        tab_tmp[row][3] = c5 * VALUE_1_414213562;
        tab_tmp[row][5] = c6 * VALUE_1_414213562;
        tab_tmp[row][7] = c7 - c4;
    }
    for (uint8_t column = 0; column < 8; column++) {
        // Partie paire
        a0 = tab_tmp[0][column] + tab_tmp[7][column];
        a1 = tab_tmp[1][column] + tab_tmp[6][column];
        a2 = tab_tmp[2][column] + tab_tmp[5][column];
        a3 = tab_tmp[3][column] + tab_tmp[4][column];

        b0 = a0 + a3;
        b1 = a1 + a2;
        b2 = a1 - a2;
        b3 = a0 - a3;


        mcu_array[matrix_zig_zag[0][column]] = ((int16_t) (b0 + b1) >> 3);
        mcu_array[matrix_zig_zag[4][column]] = ((int16_t) (b0 - b1) >> 3);

        tmp0 = b2 * VALUE_1_662939225;

        mcu_array[matrix_zig_zag[2][column]] = ((int16_t) ((b3 - b2) * VALUE_0_275899379 + tmp0) >> 3);
        mcu_array[matrix_zig_zag[6][column]] = ((int16_t) ((b3 + b2) * (- VALUE_1_387039845) + tmp0) >> 3);

        // Partie impaire

        a4 = tab_tmp[3][column] - tab_tmp[4][column];
        a5 = tab_tmp[2][column] - tab_tmp[5][column];
        a6 = tab_tmp[1][column] - tab_tmp[6][column];
        a7 = tab_tmp[0][column] - tab_tmp[7][column];

        tmp1 = a4 * VALUE_1_387039845;

        b4 = (a7 - a4) * VALUE_0_555570233 + tmp1;
        b7 = (a7 + a4) * (- VALUE_0_831469612) + tmp1;
        
        tmp2 = a5 * VALUE_1_175875602;

        b5 = (a6 - a5) * VALUE_0_195090322 + tmp2;
        b6 = (a6 + a5) * (- VALUE_0_980785280) + tmp2;

        c4 = (b4 + b6);
        c5 = (b7 - b5);
        c6 = (b4 - b6);
        c7 = (b7 + b5);

        mcu_array[matrix_zig_zag[1][column]] = ((int16_t) (c7 + c4) >> 3);
        mcu_array[matrix_zig_zag[3][column]] = ((int16_t) (c5 * VALUE_1_414213562) >> 3);
        mcu_array[matrix_zig_zag[5][column]] = ((int16_t) (c6 * VALUE_1_414213562) >> 3);
        mcu_array[matrix_zig_zag[7][column]] = ((int16_t) (c7 - c4) >> 3);
    }
}

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

// void dct_arai(uint8_t **bloc_spatiale, int16_t **bloc_freq)
// {
//     int32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17, tmp18;
//     int32_t tab_tmp[8][8];
//     for (uint8_t row = 0; row < 8; row++) {
//         tmp0 = (int32_t) (bloc_spatiale[row][0] + bloc_spatiale[row][7] + 256);
//         tmp1 = (int32_t) (bloc_spatiale[row][1] + bloc_spatiale[row][6] + 256);
//         tmp2 = (int32_t) (bloc_spatiale[row][2] + bloc_spatiale[row][5] + 256);
//         tmp3 = (int32_t) (bloc_spatiale[row][3] + bloc_spatiale[row][4] + 256);
//         tmp4 = (int32_t) (bloc_spatiale[row][3] - bloc_spatiale[row][4]);
//         tmp5 = (int32_t) (bloc_spatiale[row][2] - bloc_spatiale[row][5]);
//         tmp6 = (int32_t) (bloc_spatiale[row][1] - bloc_spatiale[row][6]);
//         tmp7 = (int32_t) (bloc_spatiale[row][0] - bloc_spatiale[row][7]);

//         tmp8 = tmp0 + tmp3;
//         tmp9 = tmp1 + tmp2;
//         tmp10 = tmp1 - tmp2;
//         tmp11 = tmp0 - tmp3;

//         tab_tmp[row][0] = tmp8 + tmp9;
//         tab_tmp[row][4] = tmp8 - tmp9;

//         tmp12 = (tmp10 + tmp11) * VALUE_0_707106781;
//         tab_tmp[row][2] = tmp11 + tmp12;
//         tab_tmp[row][6] = tmp11 - tmp12;

//         tmp8 = tmp4 + tmp5;
//         tmp9 = tmp5 + tmp6;
//         tmp10 = tmp6 + tmp7;

//         tmp13 = (tmp8 - tmp10) * VALUE_0_382683433;
//         tmp14 = tmp8 * VALUE_0_541196100 + tmp13;
//         tmp15 = tmp10 * VALUE_1_306562965 + tmp13;
//         tmp16 = tmp9 * VALUE_0_707106781;

//         tmp17 = tmp7 + tmp16;
//         tmp18 = tmp7 - tmp16;
        
//         tab_tmp[row][1] = tmp17 + tmp15;
//         tab_tmp[row][3] = tmp18 - tmp14;
//         tab_tmp[row][5] = tmp18 + tmp14;
//         tab_tmp[row][7] = tmp17 - tmp15;
//     }
//     for (uint8_t column = 0; column < 8; column++) {
//         tmp0 = tab_tmp[0][column] + tab_tmp[7][column];
//         tmp1 = tab_tmp[1][column] + tab_tmp[6][column];
//         tmp2 = tab_tmp[2][column] + tab_tmp[5][column];
//         tmp3 = tab_tmp[3][column] + tab_tmp[4][column];
//         tmp4 = tab_tmp[3][column] - tab_tmp[4][column];
//         tmp5 = tab_tmp[2][column] - tab_tmp[5][column];
//         tmp6 = tab_tmp[1][column] - tab_tmp[6][column];
//         tmp7 = tab_tmp[0][column] - tab_tmp[7][column];

//         tmp8 = tmp0 + tmp3;
//         tmp9 = tmp1 + tmp2;
//         tmp10 = tmp1 - tmp2;
//         tmp11 = tmp0 - tmp3;

//         bloc_freq[0][column] = ((int16_t) (tmp8 + tmp9) >> 3);
//         bloc_freq[4][column] = ((int16_t) (tmp8 - tmp9) >> 3);

//         tmp12 = (tmp10 + tmp11) * VALUE_0_707106781;
//         bloc_freq[2][column] = tmp11 + tmp12;
//         bloc_freq[6][column] = tmp11 - tmp12;

//         tmp8 = tmp4 + tmp5;
//         tmp9 = tmp5 + tmp6;
//         tmp10 = tmp6 + tmp7;

//         tmp13 = (tmp8 - tmp10) * VALUE_0_382683433;
//         tmp14 = tmp8 * VALUE_0_541196100 + tmp13;
//         tmp15 = tmp10 * VALUE_1_306562965 + tmp13;
//         tmp16 = tmp9 * VALUE_0_707106781;

//         tmp17 = tmp7 + tmp16;
//         tmp18 = tmp7 - tmp16;
        
//         bloc_freq[1][column] = ((int16_t) (tmp17 + tmp15) >> 3);
//         bloc_freq[3][column] = ((int16_t) (tmp18 - tmp14) >> 3);
//         bloc_freq[5][column] = ((int16_t) (tmp18 + tmp14) >> 3);
//         bloc_freq[7][column] = ((int16_t) (tmp17 - tmp15) >> 3);
//     }
// }

// void dct_arai_bis(uint8_t **bloc_spatiale, int16_t **bloc_freq)
// {
//     double tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17, tmp18;
//     double tab_tmp[8][8];
//     for (uint8_t row = 0; row < 8; row++) {
//         tmp0 = (double) (bloc_spatiale[row][0] + bloc_spatiale[row][7] + 256);
//         tmp1 = (double) (bloc_spatiale[row][1] + bloc_spatiale[row][6] + 256);
//         tmp2 = (double) (bloc_spatiale[row][2] + bloc_spatiale[row][5] + 256);
//         tmp3 = (double) (bloc_spatiale[row][3] + bloc_spatiale[row][4] + 256);
//         tmp4 = (double) (bloc_spatiale[row][3] - bloc_spatiale[row][4]);
//         tmp5 = (double) (bloc_spatiale[row][2] - bloc_spatiale[row][5]);
//         tmp6 = (double) (bloc_spatiale[row][1] - bloc_spatiale[row][6]);
//         tmp7 = (double) (bloc_spatiale[row][0] - bloc_spatiale[row][7]);

//         tmp8 = tmp0 + tmp3;
//         tmp9 = tmp1 + tmp2;
//         tmp10 = tmp1 - tmp2;
//         tmp11 = tmp0 - tmp3;

//         tab_tmp[row][0] = tmp8 + tmp9;
//         tab_tmp[row][4] = tmp8 - tmp9;

//         tmp12 = (tmp10 + tmp11) * ((double) VALUE_0_707106781);
//         tab_tmp[row][2] = tmp11 + tmp12;
//         tab_tmp[row][6] = tmp11 - tmp12;

//         tmp8 = tmp4 + tmp5;
//         tmp9 = tmp5 + tmp6;
//         tmp10 = tmp6 + tmp7;

//         tmp13 = (tmp8 - tmp10) * ((double) VALUE_0_382683433);
//         tmp14 = tmp8 * ((double) VALUE_0_541196100) + tmp13;
//         tmp15 = tmp10 * ((double) VALUE_1_306562965) + tmp13;
//         tmp16 = tmp9 * ((double) VALUE_0_707106781);

//         tmp17 = tmp7 + tmp16;
//         tmp18 = tmp7 - tmp16;
        
//         tab_tmp[row][1] = tmp17 + tmp15;
//         tab_tmp[row][3] = tmp18 - tmp14;
//         tab_tmp[row][5] = tmp18 + tmp14;
//         tab_tmp[row][7] = tmp17 - tmp15;
//     }
//     for (uint8_t column = 0; column < 8; column++) {
//         tmp0 = tab_tmp[0][column] + tab_tmp[7][column];
//         tmp1 = tab_tmp[1][column] + tab_tmp[6][column];
//         tmp2 = tab_tmp[2][column] + tab_tmp[5][column];
//         tmp3 = tab_tmp[3][column] + tab_tmp[4][column];
//         tmp4 = tab_tmp[3][column] - tab_tmp[4][column];
//         tmp5 = tab_tmp[2][column] - tab_tmp[5][column];
//         tmp6 = tab_tmp[1][column] - tab_tmp[6][column];
//         tmp7 = tab_tmp[0][column] - tab_tmp[7][column];

//         tmp8 = tmp0 + tmp3;
//         tmp9 = tmp1 + tmp2;
//         tmp10 = tmp1 - tmp2;
//         tmp11 = tmp0 - tmp3;

//         bloc_freq[0][column] = ((int16_t) (tmp8 + tmp9) >> 3);
//         bloc_freq[4][column] = ((int16_t) (tmp8 - tmp9) >> 3);

//         tmp12 = (tmp10 + tmp11) * ((double) VALUE_0_707106781);
//         bloc_freq[2][column] = tmp11 + tmp12;
//         bloc_freq[6][column] = tmp11 - tmp12;

//         tmp8 = tmp4 + tmp5;
//         tmp9 = tmp5 + tmp6;
//         tmp10 = tmp6 + tmp7;

//         tmp13 = (tmp8 - tmp10) * ((double) VALUE_0_382683433);
//         tmp14 = tmp8 * ((double) VALUE_0_541196100) + tmp13;
//         tmp15 = tmp10 * ((double) VALUE_1_306562965) + tmp13;
//         tmp16 = tmp9 * ((double) VALUE_0_707106781);

//         tmp17 = tmp7 + tmp16;
//         tmp18 = tmp7 - tmp16;
        
//         bloc_freq[1][column] = ((int16_t) (tmp17 + tmp15) >> 3);
//         bloc_freq[3][column] = ((int16_t) (tmp18 - tmp14) >> 3);
//         bloc_freq[5][column] = ((int16_t) (tmp18 + tmp14) >> 3);
//         bloc_freq[7][column] = ((int16_t) (tmp17 - tmp15) >> 3);
//     }
// }


// void dct_loeffler_float_point(uint8_t **bloc_spatiale, int16_t **bloc_freq)
// {
//     for (int i = 0 ; i < 8; i++) {
//         for (int j = 0 ; j < 8; j++) {
//             bloc_freq[i][j] = ((int16_t) bloc_spatiale[i][j] - 128);
//             // printf("%hi\t", bloc_freq[i][j]);
//         }
//         // printf("\n");
//     }
//     int32_t tmp0, tmp1, tmp2, tmp3, tmp10, tmp11, tmp12, tmp13, z1;
//     for (uint8_t row = 0; row < 8; row++) {        
//         tmp0 = (int32_t) bloc_freq[row][0] + (int32_t) bloc_freq[row][7];
//         tmp1 = (int32_t) (bloc_freq[row][1] + bloc_freq[row][6]);
//         tmp2 = (int32_t) (bloc_freq[row][2] + bloc_freq[row][5]);
//         tmp3 = (int32_t) (bloc_freq[row][3] + bloc_freq[row][4]);

//         tmp10 = tmp0 + tmp3;
//         tmp12 = tmp0 - tmp3;
//         tmp11 = tmp1 + tmp2;
//         tmp13 = tmp1 - tmp2;

//         tmp0 = (int32_t) (bloc_freq[row][0] - bloc_freq[row][7]);
//         tmp1 = (int32_t) (bloc_freq[row][1] - bloc_freq[row][6]);
//         tmp2 = (int32_t) (bloc_freq[row][2] - bloc_freq[row][5]);
//         tmp3 = (int32_t) (bloc_freq[row][3] - bloc_freq[row][4]);

//         bloc_freq[row][0] = (int16_t) ((tmp10 + tmp11 - 1024) << 2);
//         bloc_freq[row][4] = (int16_t) ((tmp10 - tmp11) << 2);


//         z1 = (((int16_t) (tmp12 + tmp13)) * ((int16_t) (FIX_0_541196100)));
//         z1 += 1024; //1 << 10

//         bloc_freq[row][2] = (int16_t) ((z1 + (((int16_t) tmp12) * ((int16_t) FIX_0_765366865))) >> 11);
//         bloc_freq[row][6] = (int16_t) ((z1 - (((int16_t) tmp12) * ((int16_t) FIX_1_847759065))) >> 11);

//         tmp12 = tmp0 + tmp2;
//         tmp13 = tmp1 + tmp3;

//         z1 = (((int16_t) (tmp12 + tmp13)) * ((int16_t) (FIX_1_175875602)));
//         z1 += 1024;

//         tmp12 = (((int16_t) tmp12) * ((int16_t) - FIX_0_390180644));
//         tmp13 = (((int16_t) tmp13) * ((int16_t) - FIX_1_961570560));
//         tmp12 += z1;
//         tmp13 += z1;

//         z1 = (((int16_t) (tmp0 + tmp3)) * ((int16_t) - FIX_0_899976223));
//         tmp0 = (((int16_t) tmp0) * ((int16_t) FIX_1_501321110));
//         tmp3 = (((int16_t) tmp0) * ((int16_t) FIX_0_298631336));
//         tmp0 += z1 + tmp12;
//         tmp3 += z1 + tmp13;

//         z1 = (((int16_t) (tmp0 + tmp2)) * ((int16_t) - FIX_2_562915447));
//         tmp1 = (((int16_t) tmp1) * ((int16_t) FIX_3_072711026));
//         tmp2 = (((int16_t) tmp2) * ((int16_t) FIX_2_053119869));
//         tmp1 += z1 + tmp13;
//         tmp2 += z1 + tmp12;

//         bloc_freq[row][1] = (int16_t) (tmp0 >> 11);
//         bloc_freq[row][3] = (int16_t) (tmp1 >> 11);
//         bloc_freq[row][5] = (int16_t) (tmp2 >> 11);
//         bloc_freq[row][7] = (int16_t) (tmp3 >> 11);
//     }
//     for (uint8_t column = 0; column < 8; column++) {
//         tmp0 = (int32_t) (bloc_freq[0][column] + bloc_freq[7][column]);
//         tmp1 = (int32_t) (bloc_freq[1][column] + bloc_freq[6][column]);
//         tmp2 = (int32_t) (bloc_freq[2][column] + bloc_freq[5][column]);
//         tmp3 = (int32_t) (bloc_freq[3][column] + bloc_freq[4][column]);

//         tmp10 = tmp0 + tmp3 + 2;
//         tmp12 = tmp0 - tmp3;
//         tmp11 = tmp1 + tmp2;
//         tmp13 = tmp1 - tmp2;

//         tmp0 = (int32_t) (bloc_freq[0][column] - bloc_freq[7][column]);
//         tmp1 = (int32_t) (bloc_freq[1][column] - bloc_freq[6][column]);
//         tmp2 = (int32_t) (bloc_freq[2][column] - bloc_freq[5][column]);
//         tmp3 = (int32_t) (bloc_freq[3][column] - bloc_freq[4][column]);

//         bloc_freq[0][column] = (int16_t) ((tmp10 + tmp11) >> 2);
//         bloc_freq[4][column] = (int16_t) ((tmp10 - tmp11) >> 2);

//         z1 = (((int16_t) (tmp12 + tmp13)) * ((int16_t) FIX_0_541196100));
//         z1 += 16384;

//         bloc_freq[2][column] = (int16_t) ((z1 + (((int16_t) tmp12) * ((int16_t) FIX_0_765366865))) >> 15);
//         bloc_freq[6][column] = (int16_t) ((z1 - (((int16_t) tmp13) * ((int16_t) FIX_1_847759065))) >> 15);

//         tmp12 = tmp0 + tmp2;
//         tmp13 = tmp1 + tmp3;

//         z1 = (((int16_t) (tmp12 + tmp13)) * ((int16_t) FIX_1_175875602));
//         z1 += 16384;

//         tmp12 = (((int16_t) tmp12) * ((int16_t) - FIX_0_390180644));
//         tmp13 = (((int16_t) tmp13) * ((int16_t) - FIX_1_961570560));
//         tmp12 += z1;
//         tmp13 += z1;

//         z1 = (((int16_t) (tmp0 + tmp3)) * ((int16_t) - FIX_0_899976223));
//         tmp0 = (((int16_t) tmp0) * ((int16_t) FIX_1_501321110));
//         tmp3 = (((int16_t) tmp3) * ((int16_t) FIX_0_298631336));
//         tmp0 += z1 + tmp12;
//         tmp3 += z1 + tmp13;

//         z1 = (((int16_t) (tmp0 + tmp3)) * ((int16_t) - FIX_2_562915447));
//         tmp1 = (((int16_t) tmp1) * ((int16_t) FIX_3_072711026));
//         tmp2 = (((int16_t) tmp2) * ((int16_t) FIX_2_053119869));
//         tmp1 += z1 + tmp13;
//         tmp2 += z1 + tmp12;

//         bloc_freq[1][column] = (int16_t) (tmp0 >> 15);
//         bloc_freq[3][column] = (int16_t) (tmp1 >> 15);
//         bloc_freq[5][column] = (int16_t) (tmp2 >> 15);
//         bloc_freq[7][column] = (int16_t) (tmp3 >> 15);
//     }
// }


// void dct_loeffler_1D_stage1(int16_t x0, int16_t x1, int16_t x2, int16_t x3, int16_t x4, int16_t x5, int16_t x6, int16_t x7, uint8_t ID, int16_t **bloc_freq, bool verif)
// {
//     if (verif) {
//         bloc_freq[ID][0] = x0 + x7;
//         bloc_freq[ID][1] = x1 + x6;
//         bloc_freq[ID][2] = x2 + x5;
//         bloc_freq[ID][3] = x3 + x4;
//         bloc_freq[ID][4] = x3 - x4;
//         bloc_freq[ID][5] = x2 - x5;
//         bloc_freq[ID][6] = x1 - x6;
//         bloc_freq[ID][7] = x0 - x7;
//     } else {
//         bloc_freq[0][ID] = x0 + x7;
//         bloc_freq[1][ID] = x1 + x6;
//         bloc_freq[2][ID] = x2 + x5;
//         bloc_freq[3][ID] = x3 + x4;
//         bloc_freq[4][ID] = x3 - x4;
//         bloc_freq[5][ID] = x2 - x5;
//         bloc_freq[6][ID] = x1 - x6;
//         bloc_freq[7][ID] = x0 - x7;
//     }
// }

// void dct_loeffler_1D_stage2(int16_t x0, int16_t x1, int16_t x2, int16_t x3, int16_t x4, int16_t x5, int16_t x6, int16_t x7, uint8_t ID, int16_t **bloc_freq, bool verif)
// {
//     if (verif) {
//         bloc_freq[ID][0] = x0 + x3;
//         bloc_freq[ID][1] = x1 + x2;
//         bloc_freq[ID][2] = x1 - x2;
//         bloc_freq[ID][3] = x0 - x3;
//         bloc_freq[ID][4] = (int16_t) (x4*c3 + x7*s3);
//         bloc_freq[ID][5] = (int16_t) (x5*c1 + x6*s1);
//         bloc_freq[ID][6] = (int16_t) (x6*c1 - x5*s1);
//         bloc_freq[ID][7] = (int16_t) (x7*c3 - x4*s3);
//     } else {
//         bloc_freq[0][ID] = x0 + x3;
//         bloc_freq[1][ID] = x1 + x2;
//         bloc_freq[2][ID] = x1 - x2;
//         bloc_freq[3][ID] = x0 - x3;
//         bloc_freq[4][ID] = (int16_t) (x4*c3 + x7*s3);
//         bloc_freq[5][ID] = (int16_t) (x5*c1 + x6*s1);
//         bloc_freq[6][ID] = (int16_t) (x6*c1 - x5*s1);
//         bloc_freq[7][ID] = (int16_t) (x7*c3 - x4*s3);
//     }
// }

// void dct_loeffler_1D_stage3(int16_t x0, int16_t x1, int16_t x2, int16_t x3, int16_t x4, int16_t x5, int16_t x6, int16_t x7, uint8_t ID, int16_t **bloc_freq, bool verif)
// {
//     if (verif) {
//         bloc_freq[ID][0] = x0 + x1;
//         bloc_freq[ID][1] = x0 - x1;
//         bloc_freq[ID][2] = ((int16_t) sqrt2 * (x2*c6 + x3*s6));
//         bloc_freq[ID][3] = ((int16_t) sqrt2 * (x3*c6 - x2*s6));
//         bloc_freq[ID][4] = x4 + x6;
//         bloc_freq[ID][5] = x7 - x5;
//         bloc_freq[ID][6] = x4 - x6;
//         bloc_freq[ID][7] = x7 + x5;
//     } else {
//         bloc_freq[0][ID] = x0 + x1;
//         bloc_freq[1][ID] = x0 - x1;
//         bloc_freq[2][ID] = ((int16_t) sqrt2 * (x2*c6 + x3*s6));
//         bloc_freq[3][ID] = ((int16_t) sqrt2 * (x3*c6 - x2*s6));
//         bloc_freq[4][ID] = x4 + x6;
//         bloc_freq[5][ID] = x7 - x5;
//         bloc_freq[6][ID] = x4 - x6;
//         bloc_freq[7][ID] = x7 + x5;
//     }
// }

// void dct_loeffler_1D_stage4(int16_t x0, int16_t x1, int16_t x2, int16_t x3, int16_t x4, int16_t x5, int16_t x6, int16_t x7, uint8_t ID, int16_t **bloc_freq, bool verif)
// {
//     // if (verif) {
//     //     bloc_freq[ID][0] = (int16_t) (x0 * sqrt2) >> 2;
//     //     bloc_freq[ID][1] = (int16_t) ((x7 + x4) * sqrt2) >> 2;
//     //     bloc_freq[ID][2] = (int16_t) (x2 * sqrt2) >> 2;
//     //     bloc_freq[ID][3] = x6 >> 1;  //(sqrt2 * sqrt2 * x6) >> 2
//     //     bloc_freq[ID][4] = (int16_t) (x1 * sqrt2) >> 2;
//     //     bloc_freq[ID][5] = x5 >> 1; //(sqrt2 * sqrt2 * x5) >> 2
//     //     bloc_freq[ID][6] = (int16_t) (x3 * sqrt2) >> 2; 
//     //     bloc_freq[ID][7] = (int16_t) ((x7 - x4) * sqrt2) >> 2;
//     // } else {
//     //     bloc_freq[0][ID] = (int16_t) (x0 * sqrt2) >> 2;
//     //     bloc_freq[1][ID] = (int16_t) ((x7 + x4) * sqrt2) >> 2;
//     //     bloc_freq[2][ID] = (int16_t) (x2 * sqrt2) >> 2;
//     //     bloc_freq[3][ID] = x6 >> 1;  //(sqrt2 * sqrt2 * x6) >> 2
//     //     bloc_freq[4][ID] = (int16_t) (x1 * sqrt2) >> 2;
//     //     bloc_freq[5][ID] = x5 >> 1; //(sqrt2 * sqrt2 * x5) >> 2
//     //     bloc_freq[6][ID] = (int16_t) (x3 * sqrt2) >> 2; 
//     //     bloc_freq[7][ID] = (int16_t) ((x7 - x4) * sqrt2) >> 2;
//     // }
//     if (verif) {
//         bloc_freq[0][ID] = x0;
//         bloc_freq[1][ID] = x7 + x4;
//         bloc_freq[2][ID] = x2;
//         bloc_freq[3][ID] = x6*sqrt2;
//         bloc_freq[4][ID] = x1;
//         bloc_freq[5][ID] = x5*sqrt2;
//         bloc_freq[6][ID] = x3; 
//         bloc_freq[7][ID] = x7 - x4;
//     } else {
//         bloc_freq[ID][0] = x0 >> 3;
//         bloc_freq[ID][1] = (x7 + x4) >> 3;
//         bloc_freq[ID][2] = x2 >> 3;
//         bloc_freq[ID][3] = (int16_t) (x6 *sqrt2) >> 3;
//         bloc_freq[ID][4] = x1 >> 3;
//         bloc_freq[ID][5] = (int16_t) (x5 *sqrt2) >> 3;
//         bloc_freq[ID][6] = x3 >> 3;
//         bloc_freq[ID][7] = (x7 - x4) >> 3;
//     }
// }

// static void transform_to_dct_1D_row(int16_t **bloc_freq)
// {
//     bool verif = 1;
//     omp_set_num_threads(8);
//     #pragma omp parallel
//     {
//         int ID = omp_get_thread_num();
//         int16_t x0_tmp = bloc_freq[ID][0];
//         int16_t x1_tmp = bloc_freq[ID][1];
//         int16_t x2_tmp = bloc_freq[ID][2];
//         int16_t x3_tmp = bloc_freq[ID][3];
//         int16_t x4_tmp = bloc_freq[ID][4];
//         int16_t x5_tmp = bloc_freq[ID][5];
//         int16_t x6_tmp = bloc_freq[ID][6];
//         int16_t x7_tmp = bloc_freq[ID][7];
//         dct_loeffler_1D_stage1(x0_tmp, x1_tmp, x2_tmp, x3_tmp, x4_tmp, x5_tmp, x6_tmp, x7_tmp, ID, bloc_freq, verif);
//     }

//     // printf("Printing matrix after row step 1\n");
//     // for (int i = 0 ; i < 8; i++) {
//     //     for (int j = 0 ; j < 8; j++) {
//     //         printf("%hi\t", bloc_freq[i][j]);
//     //     }
//     //     printf("\n");
//     // }

//     #pragma omp parallel
//     {
//         uint8_t ID = omp_get_thread_num();
//         int16_t x0_tmp = bloc_freq[ID][0];
//         int16_t x1_tmp = bloc_freq[ID][1];
//         int16_t x2_tmp = bloc_freq[ID][2];
//         int16_t x3_tmp = bloc_freq[ID][3];
//         int16_t x4_tmp = bloc_freq[ID][4];
//         int16_t x5_tmp = bloc_freq[ID][5];
//         int16_t x6_tmp = bloc_freq[ID][6];
//         int16_t x7_tmp = bloc_freq[ID][7];
//         dct_loeffler_1D_stage2(x0_tmp, x1_tmp, x2_tmp, x3_tmp, x4_tmp, x5_tmp, x6_tmp, x7_tmp, ID, bloc_freq, verif);
//     }

//     // printf("Printing matrix after row step 2\n");
//     // for (int i = 0 ; i < 8; i++) {
//     //     for (int j = 0 ; j < 8; j++) {
//     //         printf("%hi\t", bloc_freq[i][j]);
//     //     }
//     //     printf("\n");
//     // }

//     #pragma omp parallel 
//     {
//         uint8_t ID = omp_get_thread_num();
//         int16_t x0_tmp = bloc_freq[ID][0];
//         int16_t x1_tmp = bloc_freq[ID][1];
//         int16_t x2_tmp = bloc_freq[ID][2];
//         int16_t x3_tmp = bloc_freq[ID][3];
//         int16_t x4_tmp = bloc_freq[ID][4];
//         int16_t x5_tmp = bloc_freq[ID][5];
//         int16_t x6_tmp = bloc_freq[ID][6];
//         int16_t x7_tmp = bloc_freq[ID][7];
//         dct_loeffler_1D_stage3(x0_tmp, x1_tmp, x2_tmp, x3_tmp, x4_tmp, x5_tmp, x6_tmp, x7_tmp, ID, bloc_freq, verif);
//     }

//     // printf("Printing matrix after row step 3\n");
//     // for (int i = 0 ; i < 8; i++) {
//     //     for (int j = 0 ; j < 8; j++) {
//     //         printf("%hi\t", bloc_freq[i][j]);
//     //     }
//     //     printf("\n");
//     // }

//     #pragma omp parallel
//     {
//         uint8_t ID = omp_get_thread_num();
//         int16_t x0_tmp = bloc_freq[ID][0];
//         int16_t x1_tmp = bloc_freq[ID][1];
//         int16_t x2_tmp = bloc_freq[ID][2];
//         int16_t x3_tmp = bloc_freq[ID][3];
//         int16_t x4_tmp = bloc_freq[ID][4];
//         int16_t x5_tmp = bloc_freq[ID][5];
//         int16_t x6_tmp = bloc_freq[ID][6];
//         int16_t x7_tmp = bloc_freq[ID][7];
//         dct_loeffler_1D_stage4(x0_tmp, x1_tmp, x2_tmp, x3_tmp, x4_tmp, x5_tmp, x6_tmp, x7_tmp, ID, bloc_freq, verif);
//     }
// }

// static void transform_to_dct_1D_column(int16_t **bloc_freq)
// {
//     bool verif = 0;
//     omp_set_num_threads(8);
//     #pragma omp parallel
//     {
//         uint8_t ID = omp_get_thread_num();
//         int16_t x0_tmp = bloc_freq[0][ID];
//         int16_t x1_tmp = bloc_freq[1][ID];
//         int16_t x2_tmp = bloc_freq[2][ID];
//         int16_t x3_tmp = bloc_freq[3][ID];
//         int16_t x4_tmp = bloc_freq[4][ID];
//         int16_t x5_tmp = bloc_freq[5][ID];
//         int16_t x6_tmp = bloc_freq[6][ID];
//         int16_t x7_tmp = bloc_freq[7][ID];
//         // printf("x0_tmp = %hi, row = %hhu\n", x0_tmp, ID);
//         dct_loeffler_1D_stage1(x0_tmp, x1_tmp, x2_tmp, x3_tmp, x4_tmp, x5_tmp, x6_tmp, x7_tmp, ID, bloc_freq, verif);
//     }

//     // printf("Printing matrix after column step 1\n");
//     // for (int i = 0 ; i < 8; i++) {
//     //     for (int j = 0 ; j < 8; j++) {
//     //         printf("%hi\t", bloc_freq[i][j]);
//     //     }
//     //     printf("\n");
//     // }

//     #pragma omp parallel 
//     {
//         uint8_t ID = omp_get_thread_num();
//         int16_t x0_tmp = bloc_freq[0][ID];
//         int16_t x1_tmp = bloc_freq[1][ID];
//         int16_t x2_tmp = bloc_freq[2][ID];
//         int16_t x3_tmp = bloc_freq[3][ID];
//         int16_t x4_tmp = bloc_freq[4][ID];
//         int16_t x5_tmp = bloc_freq[5][ID];
//         int16_t x6_tmp = bloc_freq[6][ID];
//         int16_t x7_tmp = bloc_freq[7][ID];
//         dct_loeffler_1D_stage2(x0_tmp, x1_tmp, x2_tmp, x3_tmp, x4_tmp, x5_tmp, x6_tmp, x7_tmp, ID, bloc_freq, verif);
//     }

//     // printf("Printing matrix after row step 2\n");
//     // for (int i = 0 ; i < 8; i++) {
//     //     for (int j = 0 ; j < 8; j++) {
//     //         printf("%hi\t", bloc_freq[i][j]);
//     //     }
//     //     printf("\n");
//     // }

//     #pragma omp parallel 
//     {
//         uint8_t ID = omp_get_thread_num();
//         int16_t x0_tmp = bloc_freq[0][ID];
//         int16_t x1_tmp = bloc_freq[1][ID];
//         int16_t x2_tmp = bloc_freq[2][ID];
//         int16_t x3_tmp = bloc_freq[3][ID];
//         int16_t x4_tmp = bloc_freq[4][ID];
//         int16_t x5_tmp = bloc_freq[5][ID];
//         int16_t x6_tmp = bloc_freq[6][ID];
//         int16_t x7_tmp = bloc_freq[7][ID];
//         dct_loeffler_1D_stage3(x0_tmp, x1_tmp, x2_tmp, x3_tmp, x4_tmp, x5_tmp, x6_tmp, x7_tmp, ID, bloc_freq, verif);
//     }

//     // printf("Printing matrix after column step 3\n");
//     // for (int i = 0 ; i < 8; i++) {
//     //     for (int j = 0 ; j < 8; j++) {
//     //         printf("%hi\t", bloc_freq[i][j]);
//     //     }
//     //     printf("\n");
//     // }

//     #pragma omp parallel 
//     {
//         uint8_t ID = omp_get_thread_num();
//         int16_t x0_tmp = bloc_freq[0][ID];
//         int16_t x1_tmp = bloc_freq[1][ID];
//         int16_t x2_tmp = bloc_freq[2][ID];
//         int16_t x3_tmp = bloc_freq[3][ID];
//         int16_t x4_tmp = bloc_freq[4][ID];
//         int16_t x5_tmp = bloc_freq[5][ID];
//         int16_t x6_tmp = bloc_freq[6][ID];
//         int16_t x7_tmp = bloc_freq[7][ID];
//         // printf("x0_tmp = %hi, row = %hhu\n", x0_tmp, ID); s
//         dct_loeffler_1D_stage4(x0_tmp, x1_tmp, x2_tmp, x3_tmp, x4_tmp, x5_tmp, x6_tmp, x7_tmp, ID, bloc_freq, verif);
//     }
// }

// void transform_to_dct_loeffler(uint8_t **bloc_spatiale, int16_t **bloc_freq)
// {
//     // for (int i = 0; i < 8; i++) {
//     //     for (int j = 0; j < 8; j++) {
//     //         // hx = half word (2octets)
//     //         printf("%hhu\t", bloc_spatiale[i][j]);
//     //     }
//     //     printf("\n");
//     // }
//     // printf("Issou\n");
//     for (int i = 0 ; i < 8; i++) {
//         for (int j = 0 ; j < 8; j++) {
//             bloc_freq[i][j] = ((int16_t) bloc_spatiale[i][j]) - 128;
//             // printf("%hi\t", bloc_freq[i][j]);
//         }
//         // printf("\n");
//     }

//     transform_to_dct_1D();

    
//     // transform_to_dct_1D_row(bloc_freq);
//     // printf("Printing matrix after row step 4\n");
//     // for (int i = 0; i < 8; i++) {
//     //     for (int j = 0; j < 8; j++) {
//     //         // hx = half word (2octets)
//     //         // printf("%04hx\t", bloc_freq[i][j]);
//     //         printf("%hi\t", bloc_freq[i][j]);
//     //     }
//     //     printf("\n");
//     // }
//     // transform_to_dct_1D_column(bloc_freq);

//     // printf("Printing matrix after column step 4\n");
//     // for (int i = 0; i < 8; i++) {
//     //     for (int j = 0; j < 8; j++) {
//     //         // hx = half word (2octets)
//     //         printf("%04hx\t", bloc_freq[i][j]);
//     //         // printf("%hi\t", bloc_freq[i][j]);
//     //     }
//     //     printf("\n");
//     // }
// }


//     // print to test
//     // printf("Printing matrix after dct : \n");
//     // for (int i = 0; i < 8; i++) {
//     //     for (int j = 0; j < 8; j++) {
//     //         // hx = half word (2octets)
//     //         printf("%04hx\t", bloc_freq[i][j] );
//     //     }
//     //     printf("\n");
//     // }
//     //printf("size : %li = %li\n",sizeof((int8_t)bloc_spatiale[0][2]), sizeof(int8_t));
//     //printf("size : %li = %li\n",sizeof(bloc_freq[0][2]), sizeof(int16_t));

//     // return bloc_freq;
