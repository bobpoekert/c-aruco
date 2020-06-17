#include <stdint.h>

/*
References:
- "OpenCV: Open Computer Vision Library"
  http://sourceforge.net/projects/opencvlibrary/
- "Stack Blur: Fast But Goodlooking"
  http://incubator.quasimondo.com/processing/fast_blur_deluxe.php
*/

uint16_t CV_stack_box_blur_mult[] = {1, 171, 205, 293, 57, 373, 79, 137, 241, 27, 391, 357, 41, 19, 283, 265};
uint16_t CV_stack_box_blur_shift[] = {0, 9, 10, 11, 9, 12, 10, 11, 12, 9, 13, 13, 10, 9, 13, 13};

float CV_gaussian_kernel_tab[] = { 1,
      0.25, 0.5, 0.25,
      0.0625, 0.25, 0.375, 0.25, 0.0625,
      0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125};
uint8_t CV_gaussian_kernel_offsets[] = {0, 1, 4, 9};

typedef struct CV_Image {

    size_t width;
    size_t height;
    uint8_t *data;

} CV_Image;

typedef struct CV_Contours {

    /* the start of the first element is always zero,
     * so element zero of contour_starts is the start of the second element
     * (index 1).
     *
     * the last element of contour_starts is always where a new contour would start.
     *
     * a Contours with only one contour has contour_starts[0] equal to that contour's length
     */

    size_t *contour_starts;
    size_t n_contours;
    size_t *xs;
    size_t *ys;
    size_t max_n_contours;
    size_t array_length;

} CV_Contours;

int8_t CV_neighborhood[][2] = 
  { {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1} };
#define CV_N_NEIGHBORHOODS 8

#define CV_BLUR_STACK_SIZE (16 * 16 + 1)

typedef struct CV_BlurStack {

    uint32_t colors[CV_BLUR_STACK_SIZE];
    uint16_t start;

} CV_BlurStack;

typedef size_t CV_PerspectiveTransform[8];

void CV_get_perspective_transform(CV_Contours *c, size_t contour_idx, size_t warp_size, CV_PerspectiveTransform res);

void CV_BlurStack_init(CV_BlurStack *inp);

void CV_threshold(CV_Image inp, uint8_t thresh);
void CV_adaptive_threshold(CV_Image inp, CV_Image outp, uint8_t kernel_size, uint8_t thresh);
uint8_t CV_otsu(CV_Image src);
void CV_stack_box_blur(CV_Image image_src, CV_Image image_dst, uint8_t kernel_size);

void CV_gaussian_blur(
        CV_Image src_image, CV_Image dst_image, CV_Image mean_image, uint8_t kernel_size);

void CV_find_contours(CV_Image image_src, CV_Image *binary, CV_Contours *res);

void CV_approx_poly_dp(CV_Contours *contour, size_t contour_idx, float epsilon, CV_Contours *res);

void CV_warp(CV_Image image_src, CV_Image *image_dst, CV_Contours *contours, size_t contour_idx, size_t warp_size);

uint8_t CV_is_contour_convex(CV_Contours *contours, size_t contour_idx);

float CV_perimeter(CV_Contours *contours, size_t contour_idx);
size_t CV_count_nonzero(CV_Image image_src,
        size_t x, size_t y,
        size_t w, size_t h);
void CV_binary_border(CV_Image image_src, CV_Image *image_dst);
