#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "jpeg_writer.h"
#include "bitstream.h"
#include "huffman.h"
#include "coding.h"

/*
    Ecrire header de l'image gris

    Entrées : nom de fichier ppm, nom de fichier jpeg, width et height de l'image ppm, hash table dc et ac luminance (Y)

    Sortie : le structure jpeg
*/
extern struct jpeg *write_jpeg_gris_header(const char *ppm_file, const char* jpeg_file, uint32_t width, uint32_t height, struct huff_table *ht_dc, struct huff_table *ht_ac);

/*
    Ecrire header de l'image color

    Entrées : nom de fichier ppm, nom de fichier jpeg, width et height de l'image ppm, hash table dc et ac luminance (Y), 
              hash table dc et ac chrominance (Cb Cr) et les valeurs des sous-échantillonage

    Sortie : le structure jpeg
*/
extern struct jpeg *write_jpeg_color_header(const char *ppm_file, const char* jpeg_file, uint32_t width, uint32_t height, struct huff_table *ht_dc_Y, struct huff_table *ht_ac_Y, struct huff_table *ht_dc_C, struct huff_table *ht_ac_C, uint8_t h1, uint8_t v1, uint8_t h2, uint8_t v2, uint8_t h3, uint8_t v3);