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

void dct_arai(uint8_t **bloc_spatiale, int16_t *mcu_array)
{
    int32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17, tmp18;
    int32_t tab_tmp[8][8];
    for (uint8_t row = 0; row < 8; row++) {
        tmp0 = (int32_t) (bloc_spatiale[row][0] + bloc_spatiale[row][7] - 256);
        tmp1 = (int32_t) (bloc_spatiale[row][1] + bloc_spatiale[row][6] - 256);
        tmp2 = (int32_t) (bloc_spatiale[row][2] + bloc_spatiale[row][5] - 256);
        tmp3 = (int32_t) (bloc_spatiale[row][3] + bloc_spatiale[row][4] - 256);
        tmp4 = (int32_t) (bloc_spatiale[row][3] - bloc_spatiale[row][4]);
        tmp5 = (int32_t) (bloc_spatiale[row][2] - bloc_spatiale[row][5]);
        tmp6 = (int32_t) (bloc_spatiale[row][1] - bloc_spatiale[row][6]);
        tmp7 = (int32_t) (bloc_spatiale[row][0] - bloc_spatiale[row][7]);

        tmp8 = tmp0 + tmp3;
        tmp9 = tmp1 + tmp2;
        tmp10 = tmp1 - tmp2;
        tmp11 = tmp0 - tmp3;

        tab_tmp[row][0] = tmp8 + tmp9;
        tab_tmp[row][4] = tmp8 - tmp9;

        tmp12 = (tmp10 + tmp11) * VALUE_0_707106781;
        tab_tmp[row][2] = tmp11 + tmp12;
        tab_tmp[row][6] = tmp11 - tmp12;

        tmp8 = tmp4 + tmp5;
        tmp9 = tmp5 + tmp6;
        tmp10 = tmp6 + tmp7;

        tmp13 = (tmp8 - tmp10) * VALUE_0_382683433;
        tmp14 = tmp8 * VALUE_0_541196100 + tmp13;
        tmp15 = tmp10 * VALUE_1_306562965 + tmp13;
        tmp16 = tmp9 * VALUE_0_707106781;

        tmp17 = tmp7 + tmp16;
        tmp18 = tmp7 - tmp16;
        
        tab_tmp[row][1] = tmp17 + tmp15;
        tab_tmp[row][3] = tmp18 - tmp14;
        tab_tmp[row][5] = tmp18 + tmp14;
        tab_tmp[row][7] = tmp17 - tmp15;
    }
    for (uint8_t column = 0; column < 8; column++) {
        tmp0 = tab_tmp[0][column] + tab_tmp[7][column];
        tmp1 = tab_tmp[1][column] + tab_tmp[6][column];
        tmp2 = tab_tmp[2][column] + tab_tmp[5][column];
        tmp3 = tab_tmp[3][column] + tab_tmp[4][column];
        tmp4 = tab_tmp[3][column] - tab_tmp[4][column];
        tmp5 = tab_tmp[2][column] - tab_tmp[5][column];
        tmp6 = tab_tmp[1][column] - tab_tmp[6][column];
        tmp7 = tab_tmp[0][column] - tab_tmp[7][column];

        tmp8 = tmp0 + tmp3;
        tmp9 = tmp1 + tmp2;
        tmp10 = tmp1 - tmp2;
        tmp11 = tmp0 - tmp3;

        mcu_array[matrix_zig_zag[0][column]] = ((int16_t) (tmp8 + tmp9) >> 3);
        mcu_array[matrix_zig_zag[4][column]] = ((int16_t) (tmp8 - tmp9) >> 3);

        tmp12 = (tmp10 + tmp11) * VALUE_0_707106781;
        mcu_array[matrix_zig_zag[2][column]] = tmp11 + tmp12;
        mcu_array[matrix_zig_zag[6][column]] = tmp11 - tmp12;

        tmp8 = tmp4 + tmp5;
        tmp9 = tmp5 + tmp6;
        tmp10 = tmp6 + tmp7;

        tmp13 = (tmp8 - tmp10) * VALUE_0_382683433;
        tmp14 = tmp8 * VALUE_0_541196100 + tmp13;
        tmp15 = tmp10 * VALUE_1_306562965 + tmp13;
        tmp16 = tmp9 * VALUE_0_707106781;

        tmp17 = tmp7 + tmp16;
        tmp18 = tmp7 - tmp16;
        
        mcu_array[matrix_zig_zag[1][column]] = ((int16_t) (tmp17 + tmp15) >> 3);
        mcu_array[matrix_zig_zag[3][column]] = ((int16_t) (tmp18 - tmp14) >> 3);
        mcu_array[matrix_zig_zag[5][column]] = ((int16_t) (tmp18 + tmp14) >> 3);
        mcu_array[matrix_zig_zag[7][column]] = ((int16_t) (tmp17 - tmp15) >> 3);
    }
}

void dct_arai_bis(uint8_t **bloc_spatiale, int16_t *mcu_array)
{
    double tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17, tmp18;
    double tab_tmp[8][8];
    for (uint8_t row = 0; row < 8; row++) {
        tmp0 = (double) (bloc_spatiale[row][0] + bloc_spatiale[row][7] - 256);
        tmp1 = (double) (bloc_spatiale[row][1] + bloc_spatiale[row][6] - 256);
        tmp2 = (double) (bloc_spatiale[row][2] + bloc_spatiale[row][5] - 256);
        tmp3 = (double) (bloc_spatiale[row][3] + bloc_spatiale[row][4] - 256);
        tmp4 = (double) (bloc_spatiale[row][3] - bloc_spatiale[row][4]);
        tmp5 = (double) (bloc_spatiale[row][2] - bloc_spatiale[row][5]);
        tmp6 = (double) (bloc_spatiale[row][1] - bloc_spatiale[row][6]);
        tmp7 = (double) (bloc_spatiale[row][0] - bloc_spatiale[row][7]);

        tmp8 = tmp0 + tmp3;
        tmp9 = tmp1 + tmp2;
        tmp10 = tmp1 - tmp2;
        tmp11 = tmp0 - tmp3;

        tab_tmp[row][0] = tmp8 + tmp9;
        tab_tmp[row][4] = tmp8 - tmp9;

        tmp12 = (tmp10 + tmp11) * ((double) VALUE_0_707106781);
        tab_tmp[row][2] = tmp11 + tmp12;
        tab_tmp[row][6] = tmp11 - tmp12;

        tmp8 = tmp4 + tmp5;
        tmp9 = tmp5 + tmp6;
        tmp10 = tmp6 + tmp7;

        tmp13 = (tmp8 - tmp10) * ((double) VALUE_0_382683433);
        tmp14 = tmp8 * ((double) VALUE_0_541196100) + tmp13;
        tmp15 = tmp10 * ((double) VALUE_1_306562965) + tmp13;
        tmp16 = tmp9 * ((double) VALUE_0_707106781);

        tmp17 = tmp7 + tmp16;
        tmp18 = tmp7 - tmp16;
        
        tab_tmp[row][1] = tmp17 + tmp15;
        tab_tmp[row][3] = tmp18 - tmp14;
        tab_tmp[row][5] = tmp18 + tmp14;
        tab_tmp[row][7] = tmp17 - tmp15;
    }
    for (uint8_t column = 0; column < 8; column++) {
        tmp0 = tab_tmp[0][column] + tab_tmp[7][column];
        tmp1 = tab_tmp[1][column] + tab_tmp[6][column];
        tmp2 = tab_tmp[2][column] + tab_tmp[5][column];
        tmp3 = tab_tmp[3][column] + tab_tmp[4][column];
        tmp4 = tab_tmp[3][column] - tab_tmp[4][column];
        tmp5 = tab_tmp[2][column] - tab_tmp[5][column];
        tmp6 = tab_tmp[1][column] - tab_tmp[6][column];
        tmp7 = tab_tmp[0][column] - tab_tmp[7][column];

        tmp8 = tmp0 + tmp3;
        tmp9 = tmp1 + tmp2;
        tmp10 = tmp1 - tmp2;
        tmp11 = tmp0 - tmp3;

        mcu_array[matrix_zig_zag[0][column]] = ((int16_t) (tmp8 + tmp9) >> 3);
        mcu_array[matrix_zig_zag[4][column]] = ((int16_t) (tmp8 - tmp9) >> 3);

        tmp12 = (tmp10 + tmp11) * ((double) VALUE_0_707106781);
        mcu_array[matrix_zig_zag[2][column]] = tmp11 + tmp12;
        mcu_array[matrix_zig_zag[6][column]] = tmp11 - tmp12;

        tmp8 = tmp4 + tmp5;
        tmp9 = tmp5 + tmp6;
        tmp10 = tmp6 + tmp7;

        tmp13 = (tmp8 - tmp10) * ((double) VALUE_0_382683433);
        tmp14 = tmp8 * ((double) VALUE_0_541196100) + tmp13;
        tmp15 = tmp10 * ((double) VALUE_1_306562965) + tmp13;
        tmp16 = tmp9 * ((double) VALUE_0_707106781);

        tmp17 = tmp7 + tmp16;
        tmp18 = tmp7 - tmp16;
        
        mcu_array[matrix_zig_zag[1][column]] = ((int16_t) (tmp17 + tmp15) >> 3);
        mcu_array[matrix_zig_zag[3][column]] = ((int16_t) (tmp18 - tmp14) >> 3);
        mcu_array[matrix_zig_zag[5][column]] = ((int16_t) (tmp18 + tmp14) >> 3);
        mcu_array[matrix_zig_zag[7][column]] = ((int16_t) (tmp17 - tmp15) >> 3);
    }
}


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
