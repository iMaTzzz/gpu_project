#define _GNU_SOURCE
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <sys/types.h>
#include <stdbool.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include "coding.h"
#include "htables.h"
#include "huffman.h"
#include "jpeg_header.h"
#include "jpeg_writer.h"
#include "bitstream.h"
#include "decoupe_cpu.h"
#include "decoupe_gpu.cuh"
#include "rgb_to_ycbcr.h"

static void verif_params(char **argv)
{
    fprintf(stderr, "Usage: %s --help --gpu --outfile=ouput.jpg --sample=h1xv1,h2xv2,h3xv3 input.ppm \n", argv[0]);
    fprintf(stderr, "où:\n");
    fprintf(stderr, "\t- --help affiche la liste des options acceptées ;\n");
    fprintf(stderr, "\t- --gpu pour utiliser la version parallèle en utilisant le GPU, par défaut la version CPU est utilisée ;\n");
    fprintf(stderr, "\t- --outfile=ouput.jpg pour rédéfinir le nom du fichier de sortie ;\n");
    fprintf(stderr, "\t- --sample=h1xv1,h2xv2,h3xv3 pour définir les facteurs d'échantillonnage hxv des trois composantes de couleur ;\n");
    fprintf(stderr, "\t- input.ppm est le nom du fichier ppm à convertir ;\n");
    fprintf(stderr, "___________________________________________________________________ \n");
    fprintf(stderr, "Usage: %s --test:relative_directory_path \n", argv[0]);
    fprintf(stderr, "où:\n");
    fprintf(stderr, "\t- --test:relative_directory_path pour tester toutes les images dans le directory sur les deux versions ;\n");
    exit(EXIT_FAILURE);
}

static void error_sampling_values()
{
    fprintf(stderr, "La norme fixe un certain nombre de restrictions sur les valeurs que peuvent prendre les facteurs d'échantillonnage. En particulier :\n");
    fprintf(stderr, "\t- La valeur de chaque facteur h ou v doit être comprise entre 1 et 4 ;\n");
    fprintf(stderr, "\t- La somme des produits hixvi doit être inférieure ou égale à 10;\n");
    fprintf(stderr, "\t- Les facteurs d'échantillonnage des chrominances doivent diviser parfaitement ceux de la luminance.\n");
    exit(EXIT_FAILURE);
}

static void verify_sampling_values(uint8_t h1, uint8_t v1, uint8_t h2, uint8_t v2, uint8_t h3, uint8_t v3)
{
    if (h1 > 4 || v1 > 4 || h2 > 4 || v2 > 4 || h3 > 4 || v3 > 4) error_sampling_values();
    if (h1*v1 + h2*v2 + h3*v3 > 10) error_sampling_values();
    if (h1 % h2 != 0 || h1 % h3 != 0 || v1 % v2 != 0 || v1 % v3 != 0) error_sampling_values();
}

bool read_parameters(FILE *input, uint32_t *width, uint32_t *height) 
{
    /* 
    On va lire les trois premières lignes pour obtenir les informations liées
    au fichier input.
    */

    fgetc(input); // On lit le P
    int verif = fgetc(input) - '5'; // Si on lit 5 (P5), alors verif = 0 sinon verif = 1 pour P6

    *width = 0;
    int c = fgetc(input); // On se trouve au saut de ligne après P5 ou P6
    c = fgetc(input); // On a maintenant la première valeur
    /* La valeur d'un espace en ASCII est 32 */
    while (c != 32) {
        *width *= 10;
        *width += c - '0';
        c = fgetc(input);
    }

    c = fgetc(input); // On se trouve maintenant sur la première valeur
    *height = 0;
    /* La valeur de '\n' en ASCII est 10 */
    while (c != 10) {
        *height *= 10;
        *height += c - '0';
        c = fgetc(input);
    }

    do {
        c = fgetc(input); // Ici on lit 255 jusqu'à atteindre le saut de ligne
    } while (c != 10);  // Le prochain fgetc nous donnera la première valeur du premier pixel

    return verif;
}

static double ppm2jpeg(char* ppm_filename, char* jpg_new_filename, bool cpu, uint8_t h1, uint8_t v1, uint8_t h2, uint8_t v2, uint8_t h3, uint8_t v3)
{
    clock_t start, end;
    start = clock();
    FILE *input = fopen(ppm_filename, "r");
    if (input == NULL) {
        perror("Ouverture du fichier d'entrée n'a pas marché");
        exit(EXIT_FAILURE);
    }
    uint32_t width;
    uint32_t height;
    if (read_parameters(input, &width, &height)) { //Cas RGB
        // printf("Le type de fichier lu est : P6\n");

        /* On crée les tables de huffman pour les composantes Y et Cb/Cr */
        struct huff_table *ht_dc_Y = huffman_table_build(htables_nb_symb_per_lengths[DC][Y], 
                                          htables_symbols[DC][Y], htables_nb_symbols[DC][Y]);
        struct huff_table *ht_ac_Y = huffman_table_build(htables_nb_symb_per_lengths[AC][Y],
                                          htables_symbols[AC][Y], htables_nb_symbols[AC][Y]);
        struct huff_table *ht_dc_C = huffman_table_build(htables_nb_symb_per_lengths[DC][Cb], 
                                        htables_symbols[DC][Cb], htables_nb_symbols[DC][Cb]);
        struct huff_table *ht_ac_C = huffman_table_build(htables_nb_symb_per_lengths[AC][Cb], 
                                        htables_symbols[AC][Cb], htables_nb_symbols[AC][Cb]);

        /* On crée le fichier de sortie en commençant par écrire son header */
        struct jpeg *jpg;
        if (jpg_new_filename == NULL) {
            char jpg_filename[] = "default.jpg";
            jpg = write_jpeg_color_header(ppm_filename, jpg_filename, width, height,
                         ht_dc_Y, ht_ac_Y, ht_dc_C, ht_ac_C, h1, v1, h2, v2, h3, v3);
        } else {
            jpg = write_jpeg_color_header(ppm_filename, jpg_new_filename, width, height,
                         ht_dc_Y, ht_ac_Y, ht_dc_C, ht_ac_C, h1, v1, h2, v2, h3, v3);
        }
        
        /* On crée le bitstream du fichier jpeg et on encode chaque pixel de chaque mcu dans le fichier */
        struct bitstream *stream = jpeg_get_bitstream(jpg);
        if (cpu) {
            treat_image_color_cpu(input, width, height, ht_dc_Y, ht_ac_Y, ht_dc_C, ht_ac_C, 
                            stream, h1, v1, h2, v2, h3, v3);
        } else {
            treat_image_color_gpu(input, width, height, ht_dc_Y, ht_ac_Y, ht_dc_C, ht_ac_C, 
                            stream, h1, v1, h2, v2, h3, v3);
        }

        /* On écrit le footer pour finaliser le fichier jpeg */
        jpeg_write_footer(jpg);

        /* Enfin on détruit tout */
        jpeg_destroy(jpg);
        huffman_table_destroy(ht_dc_Y);
        huffman_table_destroy(ht_ac_Y);
        huffman_table_destroy(ht_dc_C);
        huffman_table_destroy(ht_ac_C);

    } else { //Cas Y
        printf("width: %u, height: %u\n", width, height);
        // printf("Le type de fichier lu est : P5\n");

        /* On crée les tables de huffman pour la composante Y */
        struct huff_table *ht_dc = huffman_table_build(htables_nb_symb_per_lengths[DC][Y], 
                                        htables_symbols[DC][Y], htables_nb_symbols[DC][Y]);
        struct huff_table *ht_ac = huffman_table_build(htables_nb_symb_per_lengths[AC][Y], 
                                        htables_symbols[AC][Y], htables_nb_symbols[AC][Y]);
        struct jpeg *jpg;

        /* On crée le fichier de sortie en commençant par écrire son header */
        if (jpg_new_filename == NULL) {
            char jpg_filename[] = "default.jpg";
            jpg = write_jpeg_gris_header(ppm_filename, jpg_filename, width, height, ht_dc, ht_ac);
        } else {
            jpg = write_jpeg_gris_header(ppm_filename, jpg_new_filename, width, height, ht_dc, ht_ac);
        }

        /* On crée le bitstream du fichier jpeg et on encode chaque pixel de chaque mcu dans le fichier */
        struct bitstream *stream = jpeg_get_bitstream(jpg);
        if (cpu) {
            treat_image_grey_cpu(input, width, height, ht_dc, ht_ac, stream);
        } else {
            treat_image_grey_gpu(input, width, height, ht_dc, ht_ac, stream);
        }
        
        /* On écrit le footer pour finaliser le fichier jpeg */
        jpeg_write_footer(jpg);

        /* Enfin on détruit tout */
        jpeg_destroy(jpg);
        huffman_table_destroy(ht_dc);
        huffman_table_destroy(ht_ac);
    }
    fclose(input);

    // On libère le nouveau nom du fichier de sortie s'il existe.
    if (jpg_new_filename != NULL) {
        free(jpg_new_filename);
    }
    end = clock();
    return ((double) end - start) / CLOCKS_PER_SEC;
}

static void start_test(char* dir_path, uint8_t h1, uint8_t v1, uint8_t h2, uint8_t v2, uint8_t h3, uint8_t v3)
{
    DIR *dir = opendir(dir_path);
    if (dir == NULL) {
        perror("Failed to open directory");
        exit(EXIT_FAILURE);
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        // Process each file
        double mean_time_taken_cpu = 0;
        double mean_time_taken_gpu = 0;
        if (entry->d_type == DT_REG) { // Check if it's a regular file
            // Check if the file name ends with ".ppm" or ".pgm"
            size_t len = strlen(entry->d_name);
            if ((len > 4) && (strcmp(entry->d_name + len - 4, ".ppm") == 0 || strcmp(entry->d_name + len - 4, ".pgm") == 0)) {
                char filename[1024]; // Assuming max file name length is 1024 characters
                snprintf(filename, sizeof(filename), "%s/%s", dir_path, entry->d_name);
                // Get the size of the file
                struct stat st;
                if (stat(filename, &st) == -1) {
                    perror("Failed to get file size");
                    continue; // Skip to the next file
                }
                long file_size = st.st_size;
                uint8_t nb_of_tests = 10;
                for (uint8_t i = 0; i < nb_of_tests; ++i) {
                    // mean_time_taken_cpu += ppm2jpeg(filename, NULL, true, h1, v1, h2, v2, h3, v3); // on CPU
                    // mean_time_taken_gpu += ppm2jpeg(filename, NULL, false, h1, v1, h2, v2, h3, v3);  // on GPU
                    double tmp_cpu = ppm2jpeg(filename, NULL, true, h1, v1, h2, v2, h3, v3); // on CPU
                    double tmp_gpu = ppm2jpeg(filename, NULL, false, h1, v1, h2, v2, h3, v3);  // on GPU
                    printf("time_cpu: %f, time_gpu: %f\n", tmp_cpu, tmp_gpu);
                    mean_time_taken_cpu += tmp_cpu;
                    mean_time_taken_gpu += tmp_gpu;
                }
                mean_time_taken_cpu /= nb_of_tests;
                mean_time_taken_gpu /= nb_of_tests;
                printf("File: %s, Size: %ld bytes, Time taken: CPU=%f, GPU=%f\n", entry->d_name, file_size, mean_time_taken_cpu, mean_time_taken_gpu);
            }
        }
    }

    closedir(dir);
    free(dir_path);
}


int main(int argc, char **argv)
{
    /* On initialise les valeurs de sous-échantillonnage */
    uint8_t h1 = 1;
    uint8_t v1 = 1;
    uint8_t h2 = 1;
    uint8_t v2 = 1;
    uint8_t h3 = 1;
    uint8_t v3 = 1;
    /* Par défaut, on utilise la version cpu */
    bool cpu = true;
    /* On initialise les variables pour pouvoir lancer les tests */
    bool test = false;
    char* dir_path;
    char *jpg_new_filename = NULL;
    if (argc < 2) {
        verif_params(argv);// On affiche la notice s'il n'y a pas au moins 2 paramètres en entrée.
    }
    for (uint8_t i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            verif_params(argv);// On affiche la notice pour l'utilisation de ppm2jpeg
        } else if (strncmp(argv[i], "--gpu", 5) == 0) {
            cpu = false;
        } else if (strncmp(argv[i], "--outfile=", 10) == 0) {
            /* On alloue le nouveau nom du fichier de sortie */
            uint8_t taille = strlen(argv[i]) - 10;
            jpg_new_filename = malloc(taille+1);
            for (uint8_t j = 10; argv[i][j] != '\0'; j++) {
                jpg_new_filename[j-10] = argv[i][j];
            }
            jpg_new_filename[taille] = '\0';
        } else if (strncmp(argv[i], "--sample=", 9) == 0) {
            /* 
            On détermine les nouvelles valeurs du sous-échantillonnage 
            et on vérifie si elles vérifient les conditions.
            */
            h1 = 0;
            v1 = 0;
            h2 = 0;
            v2 = 0;
            h3 = 0;
            v3 = 0;
            uint8_t index = 9;
            while (argv[i][index] != 'x') {
                printf("%c\n", argv[i][index]);
                h1 *= 10;
                h1 += argv[i][index] - '0';
                index++;
            }
            index++; // On incrémente pour skip 'x'
            while (argv[i][index] != ',') {
                v1 *= 10;
                v1 += argv[i][index] - '0';
                index++;
            }
            index++; // On incrémente pour skip ','

            while (argv[i][index] != 'x') {
                h2 *= 10;
                h2 += argv[i][index] - '0';
                index++;
            }
            index++; // On incrémente pour skip 'x'
            while (argv[i][index] != ',') {
                v2 *= 10;
                v2 += argv[i][index] - '0';
                index++;
            }
            index++; // On incrémente pour skip ','

            while (argv[i][index] != 'x') {
                h3 *= 10;
                h3 += argv[i][index] - '0';
                index++;
            }
            index++; // On incrémente pour skip 'x'
            while (argv[i][index] != '\0') {
                v3 *= 10;
                v3 += argv[i][index] - '0';
                index++;
            }
            verify_sampling_values(h1, v1, h2, v2, h3, v3);
        } else if (strncmp(argv[i], "--test=", 7) == 0) {
            test = true;
            uint8_t taille = strlen(argv[i]) - 7;
            dir_path = malloc(taille+1);
            
            for (uint8_t j = 7; argv[i][j] != '\0'; j++) {
                dir_path[j-7] = argv[i][j];
            }
            dir_path[taille] = '\0';
        } 
    }
    if (test) {
        start_test(dir_path, h1, v1, h2, v2, h3, v3);
        return 0;
    }
    char ppm_filename[strlen(argv[argc - 1])];
    strcpy(ppm_filename, argv[argc - 1]);

    double time_taken = ppm2jpeg(ppm_filename, jpg_new_filename, cpu, h1, v2, h2, v2, h3, v3);
    printf("time taken in seconds: %f\n", time_taken);

    return 0;
}
