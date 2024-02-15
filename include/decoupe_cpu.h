#include "bitstream.h"
#include "huffman.h"
/*
    Dans cette fonction qui s'occupe des images en noir et blanc,
    on traite chaque MCU intégralement, en effectuant les transformations successives,
    avant de passer à la suivante. 
*/
extern void treat_image_grey_cpu(FILE *image, uint32_t width, uint32_t height, struct huff_table *ht_dc, 
                              struct huff_table *ht_ac, struct bitstream *stream);

/*
    Dans cette fonction qui s'occupe des images en couleur,
    on traite chaque MCU intégralement, en effectuant les transformations successives (avec un éventuel sous-échantillonnage),
    avant de passer à la suivante. 
*/
extern void treat_image_color_cpu(FILE *image, uint32_t width, uint32_t height, struct huff_table *ht_dc_Y, 
                        struct huff_table *ht_ac_Y, struct huff_table *ht_dc_C, struct huff_table *ht_ac_C, 
                        struct bitstream *stream, uint8_t h1, uint8_t v1, uint8_t h2, uint8_t v2, uint8_t h3, 
                        uint8_t v3);