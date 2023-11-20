#include <stdint.h>

/*
    Une fonction généralisée qui fait le sous-échantillonnage en suivant les valeurs sous-échantillonnage en paralètre entrées
*/
extern void downsampling(uint8_t **mcu_in, uint8_t **mcu_out, uint8_t h, uint8_t v, uint8_t h1, uint8_t v1);
