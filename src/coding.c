#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "huffman.h"
#include "bitstream.h"
#include "htables.h"

/* Fonction qui code le nombre de bits nécessaire pour coder un entier */
static uint8_t number_of_bits(int16_t integer)
{
    uint8_t count = 0;
    while (integer > 1)
    {
        count++;
        integer /= 2;
    }
    return count + 1;
}

/* Fonction qui calcule la puissance de deux entiers */
static uint16_t power(uint8_t base, uint8_t n)
{
    uint16_t p = base;
    for (uint8_t i = 1; i < n; i++)
    {
        p *= base;
    }
    return p;
}

/* Fonction qui calcule la magnitude d'un entier */
static uint8_t magnitude(int16_t value)
{
    if (value == 0)
        return 0;
    return number_of_bits(abs(value));
}

/* Fonction qui calcule l'indice d'une valeur dans une certaine magnitude */
static void index_in_magnitude(int16_t value, uint8_t magnitude, uint16_t *index)
{
    if (magnitude == 0)
    {
        *index = 0;
    }
    if (value > 0)
    {
        *index = value;
    }
    else
    {
        *index = ((uint16_t)((int16_t)power(2, magnitude) - 1) + value);
    }
}

/* Fonction qui calcule le code RLE d'une valeur non nulle d'une mcu */
static uint8_t code_RLE(uint8_t count_zero, uint8_t magnitude_ac)
{
    uint8_t RLE;
    RLE = (count_zero << 4) + magnitude_ac;
    return RLE;
}

/* Fonction qui encode le coefficient DC et les coefficients AC d'un bloc */
void coding(int16_t *bloc_array, struct huff_table *ht_dc, struct huff_table *ht_ac,
            struct bitstream *stream, int16_t *predicator, uint16_t *index)
{
    /* On encode d'abord le coefficient DC */
    uint8_t magnitude_dc = magnitude(bloc_array[0] - *predicator);
    index_in_magnitude(bloc_array[0] - *predicator, magnitude_dc, index);
    uint8_t nb_bits;
    uint32_t path = huffman_table_get_path(ht_dc, magnitude_dc, &nb_bits);
    /* On écrit dans le bitstream le chemin de huffman de la magnitude puis l'indice de la valeur */
    bitstream_write_bits(stream, path, nb_bits, 0);
    bitstream_write_bits(stream, *index, magnitude_dc, 0);
    *predicator = bloc_array[0];
    // printf("value = %i, magnitude = %i, index = %hu, path  = %u\n,
    //         nb_bits = %hhu\n", array[0] - *predicator, magnitude_dc, *index, path, magnitude_dc);

    /* Ensuite, on encode les 63 coefficients AC */
    uint8_t count_zero = 0;
    for (uint8_t i = 1; i < 64; i++)
    {
        if (bloc_array[i] == 0)
        {
            /* Si le coefficient est nul, on incrémente le compteur de zero */
            count_zero++;
        }
        else
        {
            /* Sinon, on écrit autant de fois le chemin de huffman de ZRL qu'il le faut en fonction du nombre de zero */
            if (count_zero >= 16)
            {
                while (count_zero >= 16)
                {
                    path = huffman_table_get_path(ht_ac, 240, &nb_bits); // F0 = 240
                    bitstream_write_bits(stream, path, nb_bits, 0);
                    count_zero -= 16;
                    // printf("value = zrl, huffman_path = %i, nb_bits = %hhu\n", path, nb_bits);
                }
            }
            /* On encode maintenant le code RLE du coefficient puis l'indice du coefficient */
            uint8_t magnitude_ac = magnitude(bloc_array[i]);
            uint8_t RLE = code_RLE(count_zero, magnitude_ac);
            uint32_t path = huffman_table_get_path(ht_ac, RLE, &nb_bits);
            bitstream_write_bits(stream, path, nb_bits, 0);

            index_in_magnitude(bloc_array[i], magnitude_ac, index);
            bitstream_write_bits(stream, *index, magnitude_ac, 0);
            // printf("value = %i, magnitude = %i, index = %hhu\n", array[i], magnitude_ac, *index);
            // printf("RLE code = %i, huffman path = %u, nb_bits = %hhu\n", RLE, path, nb_bits);
            count_zero = 0; // On oublie pas de reset le compteur à zero
        }
    }
    /* Si on arrive à la fin du bloc et que le compteur est non nul, alors on encode End Of Block */
    if (count_zero != 0)
    {
        path = huffman_table_get_path(ht_ac, 0, &nb_bits);
        bitstream_write_bits(stream, path, nb_bits, 0);
        // printf("value = endofblock, huffman_path = %i, nb_bits = %hhu\n", path, nb_bits);
    }
}

void coding_mcus_line(int16_t *mcus_line_array, uint32_t nb_mcus_line, struct huff_table *ht_dc, struct huff_table *ht_ac,
                      struct bitstream *stream, int16_t *predicator, uint16_t *index)
{
    for (uint32_t mcu_index = 0; mcu_index < nb_mcus_line; ++mcu_index) {
        printf("Mcu index = %u\n", mcu_index);
        uint64_t offset = 64 * mcu_index; 
        /* On encode d'abord le coefficient DC */
        uint8_t magnitude_dc = magnitude(mcus_line_array[offset + 0] - *predicator);
        index_in_magnitude(mcus_line_array[offset + 0] - *predicator, magnitude_dc, index);
        uint8_t nb_bits;
        uint32_t path = huffman_table_get_path(ht_dc, magnitude_dc, &nb_bits);
        /* On écrit dans le bitstream le chemin de huffman de la magnitude puis l'indice de la valeur */
        bitstream_write_bits(stream, path, nb_bits, 0);
        bitstream_write_bits(stream, *index, magnitude_dc, 0);
        *predicator = mcus_line_array[offset + 0];
        // printf("value = %i, magnitude = %i, index = %hu, path  = %u\n,
        //         nb_bits = %hhu\n", array[0] - *predicator, magnitude_dc, *index, path, magnitude_dc);

        /* Ensuite, on encode les 63 coefficients AC */
        uint8_t count_zero = 0;
        for (uint8_t i = 1; i < 64; i++)
        {
            if (mcus_line_array[offset + i] == 0)
            {
                /* Si le coefficient est nul, on incrémente le compteur de zero */
                count_zero++;
            }
            else
            {
                /* Sinon, on écrit autant de fois le chemin de huffman de ZRL qu'il le faut en fonction du nombre de zero */
                if (count_zero >= 16)
                {
                    while (count_zero >= 16)
                    {
                        path = huffman_table_get_path(ht_ac, 240, &nb_bits); // F0 = 240
                        bitstream_write_bits(stream, path, nb_bits, 0);
                        count_zero -= 16;
                        // printf("value = zrl, huffman_path = %i, nb_bits = %hhu\n", path, nb_bits);
                    }
                }
                /* On encode maintenant le code RLE du coefficient puis l'indice du coefficient */
                uint8_t magnitude_ac = magnitude(mcus_line_array[offset + i]);
                uint8_t RLE = code_RLE(count_zero, magnitude_ac);
                uint32_t path = huffman_table_get_path(ht_ac, RLE, &nb_bits);
                bitstream_write_bits(stream, path, nb_bits, 0);

                index_in_magnitude(mcus_line_array[offset + i], magnitude_ac, index);
                bitstream_write_bits(stream, *index, magnitude_ac, 0);
                // printf("value = %i, magnitude = %i, index = %hhu\n", array[i], magnitude_ac, *index);
                // printf("RLE code = %i, huffman path = %u, nb_bits = %hhu\n", RLE, path, nb_bits);
                count_zero = 0; // On oublie pas de reset le compteur à zero
            }
        }
        /* Si on arrive à la fin du bloc et que le compteur est non nul, alors on encode End Of Block */
        if (count_zero != 0)
        {
            path = huffman_table_get_path(ht_ac, 0, &nb_bits);
            bitstream_write_bits(stream, path, nb_bits, 0);
            // printf("value = endofblock, huffman_path = %i, nb_bits = %hhu\n", path, nb_bits);
        }
    }
    printf("Coding mcus line done\n");
}