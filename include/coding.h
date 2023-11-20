#include "jpeg_writer.h"
#include "htables.h"

extern void coding(int16_t *array, struct huff_table *ht_dc, struct huff_table *ht_ac, struct bitstream *stream, int16_t *predicator, uint16_t *index);