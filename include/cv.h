#include <stdint.h>

/*
References:
- "OpenCV: Open Computer Vision Library"
  http://sourceforge.net/projects/opencvlibrary/
- "Stack Blur: Fast But Goodlooking"
  http://incubator.quasimondo.com/processing/fast_blur_deluxe.php
*/


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


typedef size_t CV_PerspectiveTransform[8];

void CV_get_perspective_transform(CV_Contours *c, size_t contour_idx, size_t warp_size, CV_PerspectiveTransform res);

void CV_threshold(CV_Image inp, uint8_t thresh);
void CV_adaptive_threshold(CV_Image inp, CV_Image blurred, uint8_t thresh);
uint8_t CV_otsu(CV_Image src);
void CV_stack_box_blur(CV_Image image_src, uint32_t radius);


void CV_find_contours(CV_Image image_src, CV_Image *binary, CV_Contours *res);

void CV_approx_poly_dp(CV_Contours *contour, size_t contour_idx, float epsilon, CV_Contours *res);

void CV_warp(CV_Image image_src, CV_Image *image_dst, CV_Contours *contours, size_t contour_idx, size_t warp_size);

uint8_t CV_is_contour_convex(CV_Contours *contours, size_t contour_idx);

float CV_perimeter(CV_Contours *contours, size_t contour_idx);
size_t CV_count_nonzero(CV_Image image_src,
        size_t x, size_t y,
        size_t w, size_t h);
void CV_binary_border(CV_Image image_src, CV_Image *image_dst);
