// #include <stdint.h>
// #include <stdbool.h>
// #include <stdlib.h>
// #include <stdio.h>
// #include "bitstream.h"

// /*
//     Type représentant le flux d'octets à écrire dans le fichier JPEG de
//     sortie (appelé bitstream dans le sujet).
// */
// struct bitstream {
//     FILE *filename;
//     uint8_t buffer;
//     uint8_t length_buffer;
// };

// /* Retourne un nouveau bitstream prêt à écrire dans le fichier filename. */
// struct bitstream *bitstream_create(const char *filename)
// {
//     struct bitstream *stream = malloc(sizeof(struct bitstream));
//     stream->buffer = 0;
//     stream->length_buffer = 0;
//     FILE *file = fopen(filename, "wb");
        // if (file == NULL) {
        //     perror("Ouverture du fichier de sortie n'a pas marché");
        //     exit(EXIT_FAILURE);
        // }

//     stream->filename = file;
//     return stream;
// }

// /*
//     Ecrit nb_bits bits dans le bitstream. La valeur portée par cet ensemble de
//     bits est value. Le paramètre is_marker permet d'indiquer qu'on est en train
//     d'écrire un marqueur de section dans l'entête JPEG ou non (voir section
//     "Encodage dans le flux JPEG -> Byte stuffing" du sujet).
// */
// void bitstream_write_bits(struct bitstream *stream,
//                                  uint32_t value,
//                                  uint8_t nb_bits,
//                                  bool is_marker)
// {
//     if (is_marker) {
//         bitstream_flush(stream);
//     }
//     // if ((nb_bits + stream->length_buffer) < 8) {
//     //     stream->buffer = (stream->buffer << nb_bits) + value;
//     //     stream->length_buffer += nb_bits;
//     // }
//     // else {
//     //     uint8_t value2 = value >> (nb_bits - 8 + stream->length_buffer);
//     //     stream->buffer = (stream->buffer << (8 - stream->length_buffer)) + value2;
//     //     fputc(stream->buffer, stream->filename);
//     //     if ((!is_marker) && (stream->buffer == 255)) {
//     //         fputc(0, stream->filename);
//     //     }
//     //     value -= value2 << (nb_bits - 8 + stream->length_buffer);
//     //     nb_bits -= (8 - stream->length_buffer);
//     //     stream->buffer = 0;
//     //     stream->length_buffer = 0;
//     //     bitstream_write_bits(stream, value, nb_bits, is_marker);
//     // }
//     // if (is_marker) {
//     //     bitstream_flush(stream);
//     // }
//     // printf("buffer = %hhu\n", stream->buffer);
//     // printf("value = %u\n", value);
//     uint64_t concat = (stream->buffer << nb_bits) + value;
//     // printf("concat = %lu\n", concat);
//     printf("La taille du buffer = %hhu\n", stream->length_buffer);
//     nb_bits += stream -> length_buffer;
//     while (nb_bits >= 8) {
//         uint8_t to_write = (concat >> (nb_bits - 8));
//         printf("On écrit la valeur %hhu\n", to_write);
//         nb_bits -= 8;
//         fputc(to_write, stream -> filename);
//         if (!is_marker && to_write == 255) {
//             fputc(0, stream -> filename);
//         }
//     }
//     stream -> length_buffer = nb_bits;
//     stream -> buffer = concat & ((1<<nb_bits)-1);
//     // printf("Le buffer à la fin est égal à %hhu\n", stream -> buffer);
// }

// /*
//     Force l'exécution des écritures en attente sur le bitstream, s'il en
//     existe.
// */
// void bitstream_flush(struct bitstream *stream)
// {
//     if (stream->length_buffer) {
//         stream->buffer = (stream->buffer) << (8 - stream->length_buffer);
//         fputc(stream->buffer, stream->filename);
//         stream->length_buffer = 0;
//         stream->buffer = 0;
//     }
// }

// /*
//     Détruit le bitstream passé en paramètre, en libérant la mémoire qui lui est
//     associée.
// */
// void bitstream_destroy(struct bitstream *stream)
// {
//     fclose(stream->filename);
//     free(stream);
// }
