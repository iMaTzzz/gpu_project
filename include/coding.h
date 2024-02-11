#include "jpeg_writer.h"
#include "htables.h"

extern void coding(int16_t *bloc_array, struct huff_table *ht_dc, struct huff_table *ht_ac, struct bitstream *stream, int16_t *predicator, uint16_t *index);

extern void coding_mcus_line(int16_t *mcus_line_array, uint32_t nb_mcus_line, struct huff_table *ht_dc, struct huff_table *ht_ac, struct bitstream *stream, int16_t *predicator, uint16_t *index);