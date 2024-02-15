#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "zigzag.h"

/* Constantes utilisées dans les deux versions des algorithmes de Loeffler */
#define VALUE_0_765366865 0.765366865        // sqrt(2) * (sin(3pi/8) - cos(3pi/8))
#define VALUE_MINUS_1_847759065 -1.847759065 // -sqrt(2) * (cos(3pi/8) + sin(3pi/8))
#define VALUE_MINUS_1_175875602 -1.175875602 // -(cos(pi/16) + sin(pi/16))
#define VALUE_0_541196100 0.541196100        // sqrt(2) * cos(3pi/8)
#define VALUE_MINUS_0_275899379 -0.275899379 // sin(3pi/16) - cos(3pi/16)
#define VALUE_MINUS_1_387039845 -1.387039845 // -(cos(3pi/16) + sin(3pi/16)
#define VALUE_1_414213562 1.414213562        // sqrt(2)
#define VALUE_0_831469612 0.831469612        // cos(3pi/16)
#define VALUE_0_980785280 0.980785280        // cos(pi/16)
#define VALUE_MINUS_0_785694958 -0.785694958 // sin(pi/16) - cos(pi/16)

/* Fonction qui applique la transformée 2D grâce à l'algorithme de Loeffler non optimisé mais précis */
void cpu_dct_loeffler(uint8_t **mcu_2D, int16_t *mcu_array)
{
    int32_t tmp_array[8][8];
    int32_t a0, a1, a2, a3, a4, a5, a6, a7;
    int32_t b0, b1, b2, b3, b4, b5, b6, b7;
    int32_t c4, c5, c6, c7;
    int32_t tmp0, tmp1, tmp2;
    for (uint8_t row = 0; row < 8; row++) {
        // Stage 1 contains 8 adds (+ 4 offsets)
        a0 = (int32_t) (mcu_2D[row][0] + mcu_2D[row][7] - 256);
        a1 = (int32_t) (mcu_2D[row][1] + mcu_2D[row][6] - 256);
        a2 = (int32_t) (mcu_2D[row][2] + mcu_2D[row][5] - 256);
        a3 = (int32_t) (mcu_2D[row][3] + mcu_2D[row][4] - 256);
        a4 = (int32_t) (mcu_2D[row][3] - mcu_2D[row][4]);
        a5 = (int32_t) (mcu_2D[row][2] - mcu_2D[row][5]);
        a6 = (int32_t) (mcu_2D[row][1] - mcu_2D[row][6]);
        a7 = (int32_t) (mcu_2D[row][0] - mcu_2D[row][7]);

        // Stage 2 contains 6 mult + 10 adds
        b0 = a0 + a3;
        b1 = a1 + a2;
        b2 = a1 - a2;
        b3 = a0 - a3;
        tmp0 = VALUE_0_831469612 * (a4 + a7);
        b4 = VALUE_MINUS_0_275899379 * a7 + tmp0;
        b7 = VALUE_MINUS_1_387039845 * a4 + tmp0;
        tmp1 = VALUE_0_980785280 * (a5 + a6) ;
        b5 = VALUE_MINUS_0_785694958 * a6 + tmp1;
        b6 = VALUE_MINUS_1_175875602 * a5 + tmp1;

        // Stage 3 contains 3 mult + 9 adds
        tmp2 = VALUE_0_541196100 * (b2 + b3);
        tmp_array[row][0] = b0 + b1;
        tmp_array[row][2] = VALUE_0_765366865 * b3 + tmp2;
        tmp_array[row][4] = b0 - b1;
        tmp_array[row][6] = VALUE_MINUS_1_847759065 * b2 + tmp2;
        c4 = b4 + b6;
        c5 = b7 - b5;
        c6 = b4 - b6;
        c7 = b5 + b7;

        // Stage 4 contains 2 mults + 2 adds
        tmp_array[row][1] = c4 + c7;
        tmp_array[row][3] = c5 * VALUE_1_414213562;
        tmp_array[row][5] = c6 * VALUE_1_414213562;
        tmp_array[row][7] = c7 - c4;
    }
    for (uint8_t column = 0; column < 8; column++) {
        // Stage 1 contains 8 adds
        a0 = tmp_array[0][column] + tmp_array[7][column];
        a1 = tmp_array[1][column] + tmp_array[6][column];
        a2 = tmp_array[2][column] + tmp_array[5][column];
        a3 = tmp_array[3][column] + tmp_array[4][column];
        a4 = tmp_array[3][column] - tmp_array[4][column];
        a5 = tmp_array[2][column] - tmp_array[5][column];
        a6 = tmp_array[1][column] - tmp_array[6][column];
        a7 = tmp_array[0][column] - tmp_array[7][column];

        // Stage 2 contains 6 mult + 10 adds
        b0 = a0 + a3;
        b1 = a1 + a2;
        b2 = a1 - a2;
        b3 = a0 - a3;
        tmp0 = VALUE_0_831469612 * (a4 + a7);
        b4 = VALUE_MINUS_0_275899379 * a7 + tmp0;
        b7 = VALUE_MINUS_1_387039845 * a4 + tmp0;
        tmp1 = VALUE_0_980785280 * (a5 + a6) ;
        b5 = VALUE_MINUS_0_785694958 * a6 + tmp1;
        b6 = VALUE_MINUS_1_175875602 * a5 + tmp1;

        // Stage 3 contains 3 mult + 9 adds
        tmp2 = VALUE_0_541196100 * (b2 + b3);
        mcu_array[matrix_zig_zag[0][column]] = (int16_t) ((b0 + b1) >> 3);
        mcu_array[matrix_zig_zag[2][column]] = (int16_t) (((int32_t) (VALUE_0_765366865 * b3 + tmp2)) >> 3);
        mcu_array[matrix_zig_zag[4][column]] = (int16_t) ((b0 - b1) >> 3);
        mcu_array[matrix_zig_zag[6][column]] = (int16_t) (((int32_t) (VALUE_MINUS_1_847759065 * b2 + tmp2)) >> 3);
        c4 = b4 + b6;
        c5 = b7 - b5;
        c6 = b4 - b6;
        c7 = b5 + b7;

        // Stage 4 contains 2 mults + 2 adds + 8 normalized shifts (multiply by 8)
        mcu_array[matrix_zig_zag[1][column]] = (int16_t) ((c4 + c7) >> 3);
        mcu_array[matrix_zig_zag[3][column]] = (int16_t) (((int32_t) (c5 * VALUE_1_414213562)) >> 3);
        mcu_array[matrix_zig_zag[5][column]] = (int16_t) (((int32_t) (c6 * VALUE_1_414213562)) >> 3);
        mcu_array[matrix_zig_zag[7][column]] = (int16_t) ((c7 - c4) >> 3);
    }
}
