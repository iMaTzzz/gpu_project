#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "jpeg_writer.h"
#include "htables.h"
#include "qtables.h"
#include "bitstream.h"
#include "huffman.h"
#include "coding.h"

/*
    Ecrire header de l'image gris

    Entrées : nom de fichier ppm, nom de fichier jpeg, width et height de l'image ppm, hash table dc et ac luminance (Y)

    Sortie : le structure jpeg
*/
struct jpeg *write_jpeg_gris_header(const char *ppm_file, const char* jpeg_file, uint32_t width, uint32_t height, struct huff_table *ht_dc, struct huff_table *ht_ac)
{
    struct jpeg *jpg = jpeg_create();
    jpeg_set_ppm_filename(jpg, ppm_file);
    jpeg_set_image_width(jpg, width);
    jpeg_set_image_height(jpg, height);
    jpeg_set_nb_components(jpg, 1); // nb component = 1 pour image gris
    jpeg_set_jpeg_filename(jpg, jpeg_file);
    jpeg_set_sampling_factor(jpg, Y, H, 1); // 1x1 pour l'image gris
    jpeg_set_sampling_factor(jpg, Y, V, 1);
    jpeg_set_huffman_table(jpg, DC, Y, ht_dc); // DC et Y 
    jpeg_set_huffman_table(jpg, AC, Y, ht_ac); // AC et Y 
    jpeg_set_quantization_table(jpg, Y, quantification_table_Y);
    jpeg_write_header(jpg);
    return jpg;
}

/*
    Ecrire header de l'image color

    Entrées : nom de fichier ppm, nom de fichier jpeg, width et height de l'image ppm, hash table dc et ac luminance (Y), 
              hash table dc et ac chrominance (Cb Cr) et les valeurs des sous-échantillonage

    Sortie : le structure jpeg
*/
struct jpeg *write_jpeg_color_header(const char *ppm_file, const char* jpeg_file, uint32_t width, uint32_t height, struct huff_table *ht_dc_Y, struct huff_table *ht_ac_Y
                                    , struct huff_table *ht_dc_C, struct huff_table *ht_ac_C, uint8_t h1, uint8_t v1, uint8_t h2, uint8_t v2, uint8_t h3, uint8_t v3){
    struct jpeg *jpg = jpeg_create();
    jpeg_set_ppm_filename(jpg, ppm_file);
    jpeg_set_image_width(jpg, width);
    jpeg_set_image_height(jpg, height);
    jpeg_set_nb_components(jpg, 3); // nb component = 3 pour image color
    jpeg_set_jpeg_filename(jpg, jpeg_file);
    /*
        sampling factor pour Y, Cb, Cr 
    */
    jpeg_set_sampling_factor(jpg, Y, H, h1);
    jpeg_set_sampling_factor(jpg, Y, V, v1);
    jpeg_set_sampling_factor(jpg, Cb, H, h2);
    jpeg_set_sampling_factor(jpg, Cb, V, v2);
    jpeg_set_sampling_factor(jpg, Cr, H, h3);
    jpeg_set_sampling_factor(jpg, Cr, V, v3);
    jpeg_set_huffman_table(jpg, DC, Y, ht_dc_Y);
    jpeg_set_huffman_table(jpg, AC, Y, ht_ac_Y);
    jpeg_set_huffman_table(jpg, DC, Cb, ht_dc_C);
    jpeg_set_huffman_table(jpg, AC, Cb, ht_ac_C);
    jpeg_set_huffman_table(jpg, DC, Cr, ht_dc_C);
    jpeg_set_huffman_table(jpg, AC, Cr, ht_ac_C);
    jpeg_set_quantization_table(jpg, Y, quantification_table_Y);
    jpeg_set_quantization_table(jpg, Cb, quantification_table_CbCr);
    jpeg_set_quantization_table(jpg, Cr, quantification_table_CbCr);
    jpeg_write_header(jpg);
    return jpg;
}



