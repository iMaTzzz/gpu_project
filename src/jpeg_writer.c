#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "bitstream.h"
#include "huffman.h"

/*
    Type opaque contenant l'intégralité des informations 
    nécessaires à l'écriture de l'en-tête JPEG.
*/
struct jpeg{
    const char *ppm_filename;
    const char *jpeg_filename;
    int width;
    int height;
    uint8_t nb_components;
    uint8_t *qtable_y;
    uint8_t *qtable_cbcr;
    uint8_t h1, h2, h3, v1, v2, v3;
    struct huff_table *ht_dc_Y;
    struct huff_table *ht_ac_Y;
    struct huff_table *ht_dc_C;
    struct huff_table *ht_ac_C;
    struct bitstream* bitstream;
};

/* Type opaque représentant un arbre de Huffman. */
//struct huff_table;

/***********************************************/
/* Ouverture, fermeture et fonctions générales */
/***********************************************/

/* Alloue et retourne une nouvelle structure jpeg. */
struct jpeg *jpeg_create(void)
{
    struct jpeg *jpg = malloc(sizeof(struct jpeg));
    return jpg;
}

/*
    Détruit une structure jpeg. 
    Toute la mémoire qui lui est associée est libérée.
*/
void jpeg_destroy(struct jpeg *jpg){
    bitstream_destroy(jpg->bitstream);
    free(jpg);
}

/*
    Ecrit tout l'en-tête JPEG dans le fichier de sortie à partir des
    informations contenues dans la structure jpeg passée en paramètre. 
    En sortie, le bitstream est positionné juste après l'écriture de 
    l'en-tête SOS, à l'emplacement du premier octet de données brutes à écrire.
*/

void jpeg_write_dht(struct bitstream *stream, uint8_t *ht_length_vector, uint8_t *ht_symbols, uint8_t code)
{
    bitstream_write_bits(stream, 0xFF, 8, 1); // marquer
    bitstream_write_bits(stream, 0xC4, 8, 1);
    // length of section DHT fixed (without last part)
    uint8_t length_fixed = 19;
	for (uint8_t i = 0; i < 16; i++)
		length_fixed += ht_length_vector[i];
    bitstream_write_bits(stream, length_fixed, 16, 0);
    bitstream_write_bits(stream, code, 8, 0);
    for(uint8_t i = 0; i < 16 ; i++){bitstream_write_bits(stream, ht_length_vector[i], 8, 0);};
    for(uint8_t i = 0; length_fixed > 19 ; i++, length_fixed--){bitstream_write_bits(stream, ht_symbols[i], 8, 0);};     
}

void jpeg_write_header(struct jpeg *jpg){
    jpg->bitstream = bitstream_create(jpg->jpeg_filename);
    struct bitstream *stream = jpg->bitstream;
    // SOI symbol
    bitstream_write_bits(stream, 0xFF, 8, 1); // marquer
    bitstream_write_bits(stream, 0xD8, 8, 1);

    // APP0 symbol
    bitstream_write_bits(stream, 0xFF, 8, 1); // marquer
    bitstream_write_bits(stream, 0xE0, 8, 1);
    bitstream_write_bits(stream, 16, 16, 0); // length of APP0
    bitstream_write_bits(stream, 'J', 8, 0);  // JFIF ID
    bitstream_write_bits(stream, 'F', 8, 0);
    bitstream_write_bits(stream, 'I', 8, 0);
    bitstream_write_bits(stream, 'F', 8, 0);
    bitstream_write_bits(stream, 0x00, 8, 0);
    bitstream_write_bits(stream, 0x01, 8, 0); // JFIF Version 1.1
    bitstream_write_bits(stream, 0x01, 8, 0);
    for(int i = 0; i < 7 ; i++){bitstream_write_bits(stream, 0x00, 8, 0);;};

    // COM symbol
    bitstream_write_bits(stream, 0xFF, 8, 1); // marquer
    bitstream_write_bits(stream, 0xFE, 8, 1);
    bitstream_write_bits(stream, 16, 16, 0); // length of COM
    bitstream_write_bits(stream, '<', 8, 0);
    bitstream_write_bits(stream, '3', 8, 0);
    bitstream_write_bits(stream, ' ', 8, 0);
    bitstream_write_bits(stream, 'l', 8, 0);
    bitstream_write_bits(stream, 'e', 8, 0);
    bitstream_write_bits(stream, ' ', 8, 0);
    bitstream_write_bits(stream, 'p', 8, 0);
    bitstream_write_bits(stream, 'r', 8, 0);
    bitstream_write_bits(stream, 'o', 8, 0);
    bitstream_write_bits(stream, 'j', 8, 0);
    bitstream_write_bits(stream, 'e', 8, 0);
    bitstream_write_bits(stream, 't', 8, 0);
    bitstream_write_bits(stream, ' ', 8, 0);
    bitstream_write_bits(stream, 'C', 8, 0);

    // DQT (quantization table) pour luminance (Y)
    bitstream_write_bits(stream, 0xFF, 8, 1); // marquer
    bitstream_write_bits(stream, 0xDB, 8, 1);
    bitstream_write_bits(stream, 67, 16, 0); // length of COM
    bitstream_write_bits(stream, 0x00, 8, 0); // précision et indice pour Y
    for(int i = 0; i < 64 ; i++){bitstream_write_bits(stream, (jpg->qtable_y)[i], 8, 0);}; // passer l'array de quantization Y( sous forme le zigzag )

    // DQT (quantization table) pour chrominance (CbCr)
    if(jpg->nb_components == 3){
        bitstream_write_bits(stream, 0xFF, 8, 1); // marquer
        bitstream_write_bits(stream, 0xDB, 8, 1);
        bitstream_write_bits(stream, 67, 16, 0); // length of COM        
        bitstream_write_bits(stream, 0x01, 8, 0); // précision et indice pour Y
        for(int i = 0; i < 64 ; i++){bitstream_write_bits(stream, (jpg->qtable_cbcr)[i], 8, 0);}; // passer l'array de quantization Y( sous forme le zigzag )
    }

    // SOF
    bitstream_write_bits(stream, 0xFF, 8, 1); // marquer
    bitstream_write_bits(stream, 0xC0, 8, 1);
    if(jpg->nb_components == 1){
        bitstream_write_bits(stream, 11, 16, 0); // length of SOF
    }else{
        bitstream_write_bits(stream, 17, 16, 0); // length of SOF
    } 
    bitstream_write_bits(stream, 0x08, 8, 0); // data precision - 8bit notre cas
    bitstream_write_bits(stream, jpg->height, 16, 0);
    bitstream_write_bits(stream, jpg->width, 16, 0);
    // prend two octet au premier, et 2 octet derriere
    // fputc(((jpg->height)>>8)&0xFF, file); fputc((jpg->height)&0xFF, file); // picture height
	// fputc(((jpg->width)>>8)&0xFF, file); fputc((jpg->width)&0xFF, file); // picture width
    bitstream_write_bits(stream, jpg->nb_components, 8, 0); // nb de composant : 1 pour gris, 3 pour YCbCr
    bitstream_write_bits(stream, 0x01, 8, 0); // 1 octet : id de composant iC
    bitstream_write_bits(stream, jpg->h1, 4, 0); //4 bits sampling factor hori et vertical
    bitstream_write_bits(stream, jpg->v1, 4, 0);
    bitstream_write_bits(stream, 0x00, 8, 0); // 1 octet table de quantification (0 pour gris Y , 1 pour CbCr )
    if(jpg->nb_components == 3){
        bitstream_write_bits(stream, 0x02, 8, 0); // 1 octet : id de composant iC
        bitstream_write_bits(stream, jpg->h2, 4, 0); //4 bits sampling factor hori et vertical
        bitstream_write_bits(stream, jpg->v2, 4, 0);
        bitstream_write_bits(stream, 0x01, 8, 0); // 1 octet table de quantification (0 pour gris Y , 1 pour CbCr )
        bitstream_write_bits(stream, 0x03, 8, 0); // 1 octet : id de composant iC
        bitstream_write_bits(stream, jpg->h3, 4, 0); //4 bits sampling factor hori et vertical
        bitstream_write_bits(stream, jpg->v3, 4, 0);
        bitstream_write_bits(stream, 0x01, 8, 0); // 1 octet table de quantification (0 pour gris Y , 1 pour CbCr )
    }

    // DHT (2 pour l'image gris (dc/ac) pour l'instant )
    // DC, Y
    uint8_t *ht_length_vector = huffman_table_get_length_vector(jpg->ht_dc_Y);
    uint8_t *ht_symbols = huffman_table_get_symbols(jpg->ht_dc_Y);
    jpeg_write_dht(stream, ht_length_vector, ht_symbols, 0x00); // (0=DC, 1=AC) and table id (0=luma(Y), 1=chroma)   
    // AC, Y
    ht_length_vector = huffman_table_get_length_vector(jpg->ht_ac_Y);
    ht_symbols = huffman_table_get_symbols(jpg->ht_ac_Y);
    jpeg_write_dht(stream, ht_length_vector, ht_symbols, 0x10); // (0=DC, 1=AC) and table id (0=luma(Y), 1=chroma)  

    if(jpg->nb_components == 3){
        // DC, CbCr
        ht_length_vector = huffman_table_get_length_vector(jpg->ht_dc_C);
        ht_symbols = huffman_table_get_symbols(jpg->ht_dc_C);
        jpeg_write_dht(stream, ht_length_vector, ht_symbols, 0x01); // (0=DC, 1=AC) and table id (0=luma(Y), 1=chroma) 

        // AC, CbCr
        ht_length_vector = huffman_table_get_length_vector(jpg->ht_ac_C);
        ht_symbols = huffman_table_get_symbols(jpg->ht_ac_C);
        jpeg_write_dht(stream, ht_length_vector, ht_symbols, 0x11); // (0=DC, 1=AC) and table id (0=luma(Y), 1=chroma) 
    }
  
    // SOS
    bitstream_write_bits(stream, 0xFF, 8, 1); // marquer
    bitstream_write_bits(stream, 0xDA, 8, 1);   
    if(jpg->nb_components == 1){ // gris
        bitstream_write_bits(stream, 8, 16, 0); // length (2N+6)
        bitstream_write_bits(stream, 0x01, 8, 0); // nb composant   
        bitstream_write_bits(stream, 0x01, 8, 0); // identifiant de la composante (Y)   
        bitstream_write_bits(stream, 0x00, 8, 0); // indice table huffman
    }else{
        bitstream_write_bits(stream, 12, 16, 0); // length (2N+6)
        bitstream_write_bits(stream, 0x03, 8, 0); // nb composant    
        bitstream_write_bits(stream, 0x01, 8, 0); // identifiant de la composante (Y)   
        bitstream_write_bits(stream, 0x00, 8, 0); // indice table huffman
        bitstream_write_bits(stream, 0x02, 8, 0); // // identifiant de la composante (Cb)  
        bitstream_write_bits(stream, 0x11, 8, 0); // indice table huffman
        bitstream_write_bits(stream, 0x03, 8, 0); // // identifiant de la composante (Cr)  
        bitstream_write_bits(stream, 0x11, 8, 0); // indice table huffman
    }
    bitstream_write_bits(stream, 0x00, 8, 0);
    bitstream_write_bits(stream, 0x3F, 8, 0);
    bitstream_write_bits(stream, 0x00, 8, 0);
}

/* Ecrit le footer JPEG (marqueur EOI) dans le fichier de sortie. */
void jpeg_write_footer(struct jpeg *jpg){
    // EOI symbol
    bitstream_flush(jpeg_get_bitstream(jpg));
    bitstream_write_bits(jpeg_get_bitstream(jpg), 0xFF, 8, 1); // marquer
    bitstream_write_bits(jpeg_get_bitstream(jpg), 0xD9, 8, 1);
}

/*
    Retourne le bitstream associé au fichier de sortie enregistré 
    dans la structure jpeg.
*/
struct bitstream *jpeg_get_bitstream(struct jpeg *jpg)
{
    return jpg->bitstream;
}


/****************************************************/
/* Gestion des paramètres de l'encodeur via le jpeg */
/****************************************************/

/* Ecrit le nom de fichier PPM ppm_filename dans la structure jpeg. */
void jpeg_set_ppm_filename(struct jpeg *jpg,
                                  const char *ppm_filename)
{
    jpg->ppm_filename = ppm_filename;
};

/* Retourne le nom de fichier PPM lu dans la structure jpeg. */
char *jpeg_get_ppm_filename(struct jpeg *jpg){
    return (char *)jpg->ppm_filename;
}

/* Ecrit le nom du fichier de sortie jpeg_filename dans la structure jpeg. */
void jpeg_set_jpeg_filename(struct jpeg *jpg,
                                   const char *jpeg_filename)
{
    jpg->jpeg_filename = jpeg_filename;
}                                

/* Retourne le nom du fichier de sortie lu depuis la structure jpeg. */
char *jpeg_get_jpeg_filename(struct jpeg *jpg){
    return (char *)jpg->jpeg_filename;
}


/*
    Ecrit la hauteur de l'image traitée, en nombre de pixels,
    dans la structure jpeg.
*/
void jpeg_set_image_height(struct jpeg *jpg,
                                  uint32_t image_height)
{
    jpg->height = image_height;
}

/*
    Retourne la hauteur de l'image traitée, en nombre de pixels,
    lue dans la structure jpeg.
*/
uint32_t jpeg_get_image_height(struct jpeg *jpg){
    return jpg->height;
}

/*
    Ecrit la largeur de l'image traitée, en nombre de pixels,
    dans la structure jpeg.
*/
void jpeg_set_image_width(struct jpeg *jpg,
                                 uint32_t image_width)
{
    jpg->width = image_width;
}

/*
    Retourne la largeur de l'image traitée, en nombre de pixels,
    lue dans la structure jpeg.
*/
uint32_t jpeg_get_image_width(struct jpeg *jpg){
    return jpg->width;
}


/*
    Ecrit le nombre de composantes de couleur de l'image traitée
    dans la structure jpeg.
*/
void jpeg_set_nb_components(struct jpeg *jpg,
                                   uint8_t nb_components)
{
    jpg->nb_components = nb_components;
}

/*
    Retourne le nombre de composantes de couleur de l'image traitée 
    lu dans la structure jpeg.
*/
uint8_t jpeg_get_nb_components(struct jpeg *jpg){
    return jpg->nb_components;
}



/*
    Ecrit dans la structure jpeg le facteur d'échantillonnage sampling_factor
    à utiliser pour la composante de couleur cc et la direction dir.
*/
void jpeg_set_sampling_factor(struct jpeg *jpg,
                                     enum color_component cc,
                                     enum direction dir,
                                     uint8_t sampling_factor)
{
    if(cc == Y){
        if(dir == H){
            jpg->h1 = sampling_factor;
        }else{
            jpg->v1 = sampling_factor;
        }
    }else if (cc == Cb){
        if(dir == H){
            jpg->h2 = sampling_factor;
        }else{
            jpg->v2 = sampling_factor;
        }
    }else{
        if(dir == H){
            jpg->h3 = sampling_factor;
        }else{
            jpg->v3 = sampling_factor;
        }
    }
}

/*
    Retourne le facteur d'échantillonnage utilisé pour la composante 
    de couleur cc et la direction dir, lu dans la structure jpeg.
*/
uint8_t jpeg_get_sampling_factor(struct jpeg *jpg,
                                        enum color_component cc,
                                        enum direction dir)
{
    if(cc == Y){
        if(dir == H){
            return jpg->h1;
        }else{
            return jpg->v1;
        }
    }else if (cc == Cb){
        if(dir == H){
            return jpg->h2;
        }else{
            return jpg->v2;
        }
    }else{
        if(dir == H){
            return jpg->h3;
        }else{
            return jpg->v3;
        }
    }
}


/*
    Ecrit dans la structure jpeg la table de Huffman huff_table à utiliser
    pour encoder les données de la composante fréquentielle acdc, pour la
    composante de couleur cc.
*/
void jpeg_set_huffman_table(struct jpeg *jpg,
                                   enum sample_type acdc,
                                   enum color_component cc,
                                   struct huff_table *htable)
{
    if(cc == Y){
        if(acdc == DC){
            jpg->ht_dc_Y = htable;
        }else{
            jpg->ht_ac_Y = htable;
        }
    }else{
        if(acdc == DC){
            jpg->ht_dc_C = htable;
        }else{
            jpg->ht_ac_C = htable;
        }
    }  
}

/*
    Retourne un pointeur vers la table de Huffman utilisée pour encoder
    les données de la composante fréquentielle acdc pour la composante 
    de couleur cc, lue dans la structure jpeg.
*/
struct huff_table *jpeg_get_huffman_table(struct jpeg *jpg,
                                                 enum sample_type acdc,
                                                 enum color_component cc)
{
    if(cc == Y){
        if(acdc == DC){
            return jpg->ht_dc_Y;
        }else{
            return jpg->ht_ac_Y;
        }
    }else{
        if(acdc == DC){
            return jpg->ht_dc_C;
        }else{
            return jpg->ht_ac_C;
        }
    }     
}


/*
    Ecrit dans la structure jpeg la table de quantification à utiliser
    pour compresser les coefficients de la composante de couleur cc.
*/
void jpeg_set_quantization_table(struct jpeg *jpg,
                                        enum color_component cc,
                                        uint8_t *qtable)
{
    if(cc == Y){
        jpg->qtable_y = qtable;
    }else{
        jpg->qtable_cbcr = qtable;
    }
}

/*
    Retourne un pointeur vers la table de quantification associée à la 
    composante de couleur cc, lue dans a structure jpeg.
*/
uint8_t *jpeg_get_quantization_table(struct jpeg *jpg,
                                            enum color_component cc)
{
    if(cc == Y){
        return jpg->qtable_y;
    }else{
        return jpg->qtable_cbcr;
    }
}