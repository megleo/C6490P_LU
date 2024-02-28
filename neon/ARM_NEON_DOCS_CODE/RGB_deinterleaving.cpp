#include <arm_neon.h>

void rgb_deinterleave_c(uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *rgb, int len_color) {
    /**
     *  * Take the elements for rgb and store the individual colors "r", "g", "b".
    */
    for (size_t i = 0; i < len_color; i++)
    {
        r[i] = rgb[3 * i];
        g[i] = rgb[3 * i + 1];
        b[i] = rgb[3 * i + 2];
    }
}

void rgb_deinterleave_neon(uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *rgb, int len_color) {
    /**
     * Take the elements of "rgb" and store the individual colors "r", "g", "b".
    */
    int num8x16 = len_color / 16;
    uint8x16x3_t intlv_rgb;

    for (size_t i = 0; i < num8x16; i++)
    {
        intlv_rgb = vld3q_u8(rgb+3*16*i);
        vst1q_u8(r+16*i, intlv_rgb.val[0]);
        vst1q_u8(g+16*i, intlv_rgb.val[1]);
        vst1q_u8(b+16*i, intlv_rgb.val[2]);
    }
}