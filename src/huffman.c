#include <jpeg_writer.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

/* Type représentant un arbre de Huffman. */
struct huff_table
{
    uint8_t *nb_symb_per_lengths;
    uint8_t *symbols;
    uint8_t nb_symbols;
    uint32_t *array_path;
};

/*
    Construit un arbre de Huffman à partir d'une table
    de symboles comme présenté en section 
    "Compression d'un bloc fréquentiel -> Le codage de Huffman" du sujet.
    nb_symb_per_lengths est un tableau contenant le nombre
    de symboles pour chaque longueur de 1 à 16,
    symbols est le tableau  des symboles ordonnés,
    et nb_symbols représente la taille du tableau symbols.
*/
struct huff_table *huffman_table_build(uint8_t *nb_symb_per_lengths,
                                       uint8_t *symbols,
                                       uint8_t nb_symbols)
{
    struct huff_table *ht = malloc(sizeof(struct huff_table));
    ht -> nb_symb_per_lengths = nb_symb_per_lengths;
    ht -> symbols = symbols;
    ht -> nb_symbols = nb_symbols;
    uint32_t *array_path = malloc(nb_symbols * sizeof(uint32_t));
    uint8_t index_symbol = 0;
    uint8_t nb_symbol_for_length = 0; // Nombre de symboles déjà codés sur une longueur
    uint32_t value = 0;
    uint8_t tmp = 0;
    for (uint8_t i = 0; i < nb_symbols; i++) {
        while (nb_symb_per_lengths[index_symbol] == nb_symbol_for_length) {
            index_symbol++;
            tmp++;
            nb_symbol_for_length = 0;
        }
        if (i) {
            value = value << tmp;
        }
        nb_symbol_for_length++;
        array_path[i] = value;
        value++;
        tmp = 0;
    }
    ht -> array_path = array_path;
    return ht;
}

/*
    Retourne le chemin dans l'arbre ht permettant d'atteindre
    la feuille de valeur value. nb_bits est un paramètre de sortie
    permettant de stocker la longueur du chemin retourné.
*/
uint32_t huffman_table_get_path(struct huff_table *ht,
                                       uint8_t value,
                                       uint8_t *nb_bits)
{
    uint8_t index_symb = 0;
    uint8_t length = 0;
    uint8_t sum = 0;
    while (value != ht -> symbols[index_symb]) {
        index_symb++;
    }
    while (sum < index_symb + 1) {
        sum += ht -> nb_symb_per_lengths[length];
        length++;
    }
    *nb_bits = length;
    return ht -> array_path[index_symb];
}

/*
   Retourne le tableau des symboles associé à l'arbre de
   Huffman passé en paramètre.
*/
uint8_t *huffman_table_get_symbols(struct huff_table *ht)
{
    return ht -> symbols;
}

/*
    Retourne le tableau du nombre de symboles de chaque longueur
    associé à l'arbre de Huffman passé en paramètre.
*/
uint8_t *huffman_table_get_length_vector(struct huff_table *ht)
{
    return ht -> nb_symb_per_lengths;
}

/*
    Détruit l'arbre de Huffman passé en paramètre et libère
    toute la mémoire qui lui est associée.
*/
void huffman_table_destroy(struct huff_table *ht)
{
    free(ht->array_path);
    free(ht);
}
