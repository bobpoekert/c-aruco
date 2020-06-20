#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "cv.h"

/*void debug(char *s) {
    printf("%s\n", s);
    fflush(stdout);
}*/

#define debug(v) 

int8_t CV_neighborhood[][2] = 
  { {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1} };
#define CV_N_NEIGHBORHOODS 8

void CV_threshold(CV_Image inp, uint8_t thresh) {

    size_t inp_size = inp.width * inp.height;
    for(; inp_size > 0; inp_size--) {
        inp.data[inp_size] = inp.data[inp_size] >= thresh ? 1 : 0;
    }

}

void CV_adaptive_threshold(
        CV_Image inp, CV_Image blurred,
        uint8_t thresh) {

    assert(inp.width == blurred.width);
    assert(inp.height == blurred.height);

    uint8_t *src = inp.data;
    uint8_t *dst = blurred.data;
    size_t len = inp.width * inp.height;
    uint8_t tab[768];

    for (size_t i=0; i < 768; i++) {
        tab[i] = (i - 255 <= -thresh) ? 255 : 0;
    }

    for (size_t i = 0; i < len; i++) {
        src[i] = tab[src[i] - dst[i] + 255];
    }
}

uint8_t CV_otsu(CV_Image image_src) {

    uint8_t *src = image_src.data;
    size_t len = image_src.width * image_src.height;
    size_t hist[256];
    uint8_t threshold = 0;
    size_t sum = 0;
    size_t sum_b = 0;
    size_t w_b = 0;
    size_t w_f = 0;
    size_t max = 0;
    
    size_t mu, between, i;

    memset(hist, 0, 256);

    for (size_t i=0; i < len; i++) {
        hist[src[i]]++;
    }

    for (size_t i = 0; i < 256; i++) {
        sum += hist[i] * i;
    }

    for (size_t i=0; i < 256; i++) {
        w_b += hist[i];
        if (w_b != 0) {
            w_f = len - w_b;
            if (w_f == 0) {
                break;
            }

            sum_b += hist[i] * i;

            mu = (sum_b / w_b) - ((sum - sum_b) / w_f);

            between = w_b * w_f * mu * mu;

            if (between > max) {
                max = between;
                threshold = i;
            }
        }
    }

    return threshold;

}


#define min(a, b) ((a > b) ? b : a)
#define max(a, b) ((a > b) ? a : b)

#define DV_SIZE 64

/* adapted from http://vitiy.info/Code/stackblur.cpp */

static unsigned short const stackblur_mul[255] =
{
		512,512,456,512,328,456,335,512,405,328,271,456,388,335,292,512,
		454,405,364,328,298,271,496,456,420,388,360,335,312,292,273,512,
		482,454,428,405,383,364,345,328,312,298,284,271,259,496,475,456,
		437,420,404,388,374,360,347,335,323,312,302,292,282,273,265,512,
		497,482,468,454,441,428,417,405,394,383,373,364,354,345,337,328,
		320,312,305,298,291,284,278,271,265,259,507,496,485,475,465,456,
		446,437,428,420,412,404,396,388,381,374,367,360,354,347,341,335,
		329,323,318,312,307,302,297,292,287,282,278,273,269,265,261,512,
		505,497,489,482,475,468,461,454,447,441,435,428,422,417,411,405,
		399,394,389,383,378,373,368,364,359,354,350,345,341,337,332,328,
		324,320,316,312,309,305,301,298,294,291,287,284,281,278,274,271,
		268,265,262,259,257,507,501,496,491,485,480,475,470,465,460,456,
		451,446,442,437,433,428,424,420,416,412,408,404,400,396,392,388,
		385,381,377,374,370,367,363,360,357,354,350,347,344,341,338,335,
		332,329,326,323,320,318,315,312,310,307,304,302,299,297,294,292,
		289,287,285,282,280,278,275,273,271,269,267,265,263,261,259
};

static unsigned char const stackblur_shr[255] =
{
		9, 11, 12, 13, 13, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17,
		17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19,
		19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20,
		20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21,
		21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
		21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
		22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
		22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23,
		23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
		23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
		23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
		23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
		24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
		24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
		24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
		24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24
};

/// Stackblur algorithm body
void CV_stack_blur_job(unsigned char* src,				///< input image data
	   			  unsigned int w,					///< image width
				  unsigned int h,					///< image height
				  unsigned int radius,				///< blur intensity (should be in 2..254 range)
				  unsigned char* stack				///< stack buffer
				  )
{
	unsigned int x, y, xp, yp, i;
	unsigned int sp;
	unsigned int stack_start;
	unsigned char* stack_ptr;

	unsigned char* src_ptr;
	unsigned char* dst_ptr;

	unsigned long sum_r;
	unsigned long sum_in_r;
	unsigned long sum_out_r;

	unsigned int wm = w - 1;
	unsigned int hm = h - 1;
	unsigned int div = (radius * 2) + 1;
	unsigned int mul_sum = stackblur_mul[radius];
	unsigned char shr_sum = stackblur_shr[radius];

    
    for(y = 0; y < h; y++)
    {
        sum_r =
        sum_in_r =
        sum_out_r = 0;

        src_ptr = src + w * y; // start of line (0,y)

        for(i = 0; i <= radius; i++)
        {
            stack_ptr    = &stack[ i ];
            stack_ptr[0] = src_ptr[0];
            sum_r += src_ptr[0] * (i + 1);
            sum_out_r += src_ptr[0];
        }

        for(i = 1; i <= radius; i++)
        {
            if (i <= wm) src_ptr++;
            stack_ptr = &stack[i + radius];
            stack_ptr[0] = src_ptr[0];
            sum_r += src_ptr[0] * (radius + 1 - i);
            sum_in_r += src_ptr[0];
        }

        sp = radius;
        xp = radius;
        if (xp > wm) xp = wm;
        src_ptr = src + (xp + y * w); //   img.pix_ptr(xp, y);
        dst_ptr = src + y * w; // img.pix_ptr(0, y);
        for(x = 0; x < w; x++)
        {
            dst_ptr[0] = (sum_r * mul_sum) >> shr_sum;
            dst_ptr++;

            sum_r -= sum_out_r;

            stack_start = sp + div - radius;
            if (stack_start >= div) stack_start -= div;
            stack_ptr = &stack[stack_start];

            sum_out_r -= stack_ptr[0];

            if(xp < wm)
            {
                src_ptr++;
                ++xp;
            }

            stack_ptr[0] = src_ptr[0];

            sum_in_r += src_ptr[0];
            sum_r    += sum_in_r;

            ++sp;
            if (sp >= div) sp = 0;
            stack_ptr = &stack[sp];

            sum_out_r += stack_ptr[0];
            sum_in_r  -= stack_ptr[0];

        }

    }

    wm = w - 1;
	hm = h - 1;
	div = (radius * 2) + 1;
	mul_sum = stackblur_mul[radius];
	shr_sum = stackblur_shr[radius];
	
    x= y= xp= yp= i =
	sp =
	stack_start =
	stack_ptr = 0;

    memset(stack, 0, 254 * 2 + 1);

    for(x = w; x < w; x++)
    {
        sum_r =
        sum_in_r =
        sum_out_r = 0;

        src_ptr = src + x; // start of line x
        for(i = 0; i <= radius; i++)
        {
            stack_ptr    = &stack[ i ];
            stack_ptr[0] = src_ptr[0];
            sum_r += src_ptr[0] * (i + 1);
            sum_out_r += src_ptr[0];
        }
        for(i = 1; i <= radius; i++)
        {
            if (i <= wm) src_ptr += w;
            stack_ptr = &stack[i + radius];
            stack_ptr[0] = src_ptr[0];
            sum_r += src_ptr[0] * (radius + 1 - i);
            sum_in_r += src_ptr[0];
        }

        sp = radius;
        yp = radius;
        if (yp > hm) yp = hm;
        src_ptr = src + (x + yp * w); // img.pix_ptr(x, yp);
        dst_ptr = src + x; 			  // img.pix_ptr(x, 0);
        for(y = 0; y < h; y++)
        {
            dst_ptr[0] = (sum_r * mul_sum) >> shr_sum;
            dst_ptr += w;

            sum_r -= sum_out_r;

            stack_start = sp + div - radius;
            if(stack_start >= div) stack_start -= div;
            stack_ptr = &stack[stack_start];

            sum_out_r -= stack_ptr[0];

            if(yp < hm)
            {
                src_ptr += w; // stride
                ++yp;
            }

            stack_ptr[0] = src_ptr[0];

            sum_in_r += src_ptr[0];
            sum_r    += sum_in_r;

            ++sp;
            if (sp >= div) sp = 0;
            stack_ptr = &stack[sp];

            sum_out_r += stack_ptr[0];
            sum_in_r  -= stack_ptr[0];
        }
    }

}

void CV_stack_box_blur(CV_Image image_src, uint32_t radius) {

    assert(radius <= 254);
    assert(radius >= 2);

    uint8_t stack[254 * 2 + 1];


    CV_stack_blur_job(
            image_src.data,
            image_src.width, image_src.height,
            radius,
            stack);

}

void CV_neighborhood_deltas(size_t width, int16_t res[]) {
    for (size_t i=0; i < CV_N_NEIGHBORHOODS; ++i) {
        res[i] = CV_neighborhood[i][0] + (CV_neighborhood[i][1] * width);
    }
}

void CV_contour_push(CV_Contours *res, size_t x, size_t y) {
    assert(res->n_contours < res->max_n_contours);
    size_t contour_idx = res->contour_starts[res->n_contours];
    res->xs[contour_idx] = x;
    res->ys[contour_idx] = y;
    res->contour_starts[res->n_contours]++;
}


void CV_contours_get(CV_Contours *contour, size_t idx, size_t point_idx,
        size_t *x, size_t *y) {
    assert(idx < contour->n_contours);
    size_t off = contour->contour_starts[idx] + point_idx;
    *x = contour->xs[off];
    *y = contour->ys[off];
}

size_t CV_contours_length(CV_Contours *contour, size_t contour_idx) {
    assert(contour_idx < contour->n_contours);
    if (contour_idx == 0) {
        return contour->contour_starts[0];
    } else {
        return contour->contour_starts[contour_idx] - contour->contour_starts[contour_idx - 1];
    }
}

void CV_border_following(
        CV_Image src_image,
        size_t pos, size_t nbd,
        size_t x_start, size_t y_start,
        uint8_t hole,
        int16_t deltas[], 
        CV_Contours *res) {

    size_t pos1, pos3, pos4, s, s_end, s_prev;

    uint8_t *src = src_image.data;

    size_t x = x_start;
    size_t y = y_start;


    s = s_end = hole ? 0 : 4;

    do {
        s = (s - 1) & 7;
        pos1 = pos + deltas[s % CV_N_NEIGHBORHOODS];
        if (src[pos1] != 0) {
            break;
        }
    } while (s != s_end);

    if (s == s_end) {
        src[pos] = -nbd;
        CV_contour_push(res, x, y);
    } else {
        pos3 = pos;
        s_prev = s ^ 4;

        while(1) {
            s_end = s;
            do {
                pos4 = pos3 + deltas[(++s) % CV_N_NEIGHBORHOODS];
            } while(src[pos4] == 0);

            s &= 7;

            if (s - 1 < s_end) {
                src[pos3] = -nbd;
            } else if (src[pos3] == 1) {
                src[pos3] = nbd;
            }


            CV_contour_push(res, x, y);

            s_prev = s;

            x = CV_neighborhood[s][0];
            y = CV_neighborhood[s][1];

            if ( (pos4 == pos) && (pos3 == pos1)) break;

            pos3 = pos4;
            s = (s + 4) & 7;

        }

    }

}

void CV_find_contours(CV_Image image_src, CV_Image *binary, CV_Contours *res) {

    assert(image_src.width == binary->width && image_src.height == binary->height);

    size_t width = image_src.width;
    size_t height = image_src.height;

    CV_binary_border(image_src, binary);

    int16_t deltas[CV_N_NEIGHBORHOODS];
    CV_neighborhood_deltas(width + 2, deltas);

    size_t pos = width + 3;
    size_t nbd = 1;

    uint32_t pix;

    uint8_t outer, hole;
    size_t i, j;

    uint8_t *src = image_src.data;

    for (i=0; i < height; ++i, pos += 2) {
        for (j=0; j < width; ++j, ++pos) {
            pix = src[pos];

            if (pix != 0) {
                outer = hole = 0;

                if (pix == 1 && src[pos - 1] == 0) {
                    outer = 1;
                } else if (pix >= 1 && src[pos + 1] == 0) {
                    hole = 0;
                }

                if (outer || hole) {
                    ++nbd;

                    CV_border_following(image_src, pos, nbd, j, i, hole, deltas, res);
                    res->n_contours++;
                    if (res->n_contours >= res->max_n_contours) return;
                }
            }
        }
    }

}


#define CV_POLY_DP_STACK_SIZE 1024
void CV_approx_poly_dp(
        CV_Contours *contour, size_t contour_idx, float epsilon,
        CV_Contours *res) {

    size_t slice_start_idx = 0;
    size_t slice_end_idx = 0;
    size_t right_slice_start_idx = 0;
    size_t right_slice_end_idx = 0;

    struct {
        size_t start_index;
        size_t end_index;
    } stack[CV_POLY_DP_STACK_SIZE];
    size_t stack_end = 0;

    size_t contour_offset = contour->contour_starts[contour_idx];

    size_t pt_x, pt_y, start_x, start_y, end_x, end_y, dist, max_dist;
    float le_eps;
    size_t dx, dy, i, j, k;

    size_t len = CV_contours_length(contour, contour_idx);

    epsilon *= epsilon;
    k = 0;

    for (i=0; i < 3; ++i) {
        max_dist = 0;

        k = (k + right_slice_start_idx) % len;
        CV_contours_get(contour, contour_idx, contour_offset + k, &start_x, &start_y);

        if (++k == len) k = 0;

        for (j=1; j < len; ++j) {
            CV_contours_get(contour, contour_idx, contour_offset + k, &pt_x, &pt_y);
            if (++k == len) k = 0;

            dx = pt_x - start_x;
            dy = pt_y - start_y;
            dist = dx*dx + dy*dy;

            if (dist > max_dist) {
                max_dist = dist;
                right_slice_start_idx = j;
            }
        }
    }

    if (max_dist < epsilon) {
        CV_contour_push(res, start_x, start_y);
    } else {
        slice_start_idx = k;
        slice_end_idx = (right_slice_start_idx += slice_start_idx);

        right_slice_start_idx -= right_slice_start_idx > len ? len : 0;
        right_slice_end_idx = slice_start_idx;

        if (right_slice_end_idx < right_slice_start_idx) {
            right_slice_end_idx += len;
        }

        assert(stack_end == 0);

        stack[stack_end].start_index = right_slice_start_idx;
        stack[stack_end].end_index = right_slice_end_idx;
        stack_end++;

        stack[stack_end].start_index = slice_start_idx;
        stack[stack_end].end_index = slice_end_idx;
        stack_end++;

    }

    while(stack_end > 0) {
        size_t stack_cur = --stack_end;

        CV_contours_get(contour, contour_idx, slice_end_idx % len, &end_x, &end_y);
        CV_contours_get(contour, contour_idx, k = slice_start_idx % len, &start_x, &start_y);
        if (++k == len) k = 0;

        if (slice_end_idx <= slice_start_idx + 1) {
            le_eps = 1;
        } else {
            max_dist = 0;

            dx = end_x - start_x;
            dy = end_y - start_y;

            for(i = slice_start_idx + 1; i < slice_end_idx; ++i) {
                CV_contours_get(contour, contour_idx, k, &pt_x, &pt_y);
                if (++k == len) k = 0;

                dist = abs( (pt_y - start_y) * dx - (pt_x - start_x) * dy);

                if (dist > max_dist) {
                    max_dist = dist;
                    right_slice_start_idx = i;

                }

                le_eps = max_dist * max_dist <= epsilon * (dx * dx + dy * dy);
            }

            if (le_eps) {
                CV_contour_push(res, start_x, start_y);
            } else {
                right_slice_end_idx = slice_end_idx;
                slice_end_idx = right_slice_start_idx;


                assert(stack_end < (CV_POLY_DP_STACK_SIZE - 2));

                stack[stack_end].start_index = right_slice_start_idx;
                stack[stack_end].end_index = right_slice_end_idx;
                stack_end++;
                
                stack[stack_end].start_index = slice_start_idx;
                stack[stack_end].end_index = slice_end_idx;
                stack_end++;

            }
        }

    }

}

void CV_square2quad(CV_Contours *contours, size_t contour_idx, CV_PerspectiveTransform sq) {

    size_t x0, x1, x2, x3,
           y0, y1, y2, y3;

    CV_contours_get(contours, contour_idx, 0, &x0, &y0);
    CV_contours_get(contours, contour_idx, 1, &x1, &y1);
    CV_contours_get(contours, contour_idx, 2, &x2, &y2);
    CV_contours_get(contours, contour_idx, 3, &x3, &y3);

    size_t px, py, dx1, dx2, dy1, dy2, den;

    px = x0 - x1 + x2 - x3;
    py = y0 - y1 + y2 - y3;

    if (px == 0 && py == 0) {
        sq[0] = x1 - x0;
        sq[1] = x2;
        sq[2] = x0;
        sq[3] = y1 - y0;
        sq[4] = y2 - y1;
        sq[5] = y0;
        sq[6] = 0;
        sq[7] = 0;
        sq[8] = 1;
    } else {

        dx1 = x1 - x2;
        dx2 = x3 - x2;
        dy1 = y1 - y2;
        dy2 = y3 - y2;
        den = dx1 * dy2 - dx2 * dy1;

        sq[6] = (px * dy2 - dx2 * py) / den;
        sq[7] = (dx1 * py - px * dy1) / den;
        sq[8] = 1;
        sq[0] = x1 - x0 + sq[6] * x1;
        sq[1] = x3 - x0 + sq[7] * x3;
        sq[2] = x0;
        sq[3] = y1 - y0 + sq[6] * y1;
        sq[4] = y3 - y0 + sq[7] * y3;
        sq[5] = y0;
    }

}

void CV_get_perspective_transform(CV_Contours *contours, size_t contour_idx, size_t size, CV_PerspectiveTransform rq) {

    CV_square2quad(contours, contour_idx, rq);

    rq[0] /= size;
    rq[1] /= size;
    rq[3] /= size;
    rq[4] /= size;
    rq[6] /= size;
    rq[7] /= size;

}

void CV_warp(CV_Image image_src, CV_Image *image_dst, CV_Contours *contours, size_t contour_idx, size_t warp_size) {

    assert(image_src.width == image_dst->width && image_src.height == image_dst->height);

    uint8_t *src = image_src.data;
    uint8_t *dst = image_dst->data;

    size_t width = image_src.width;
    size_t heigt = image_src.height;

    size_t pos = 0;

    ssize_t sx1, sx2, dx1, dx2, sy1, sy2, dy1, dy2, p1, p2, p3, p4;
    size_t r, s, t, u, v, w, x, y, i, j;

    CV_PerspectiveTransform m;
    CV_get_perspective_transform(contours, contour_idx, warp_size - 1, m);

    r = m[8];
    s = m[2];
    t = m[5];

    for (i = 0; i < warp_size; ++i) {
        r += m[7];
        s += m[1];
        t += m[4];

        u = r;
        v = s;
        w = t;

        for (j=0; j < warp_size; ++j) {
            u += m[6];
            v += m[0];
            w += m[3];

            x = v / u;
            y = w / u;

            sx1 = x;
            sx2 = (sx1 == width - 1) ? sx1 : sx1 + 1;
            dx1 = x - sx1;
            dx2 = 1 - dx1;

            sy1 = y;
            sy2 = (sx1 == width - 1) ? sx1 : sx1 + 1;
            dy1 = y - sy1;
            dy2 = 1 - dy1;

            p1 = p2 = sy1 * width;
            p3 = p4 = sy2 * width;

            dst[pos++] = 
                (dy2 * (dx2 * src[p1 + sx1] + dx1 * src[p2 + sx2]) +
                 dy1 * (dx2 * src[p3 + sx1] + dx1 * src[p4 + sx2])) & 0xff;

        }
    }

    image_dst->width = warp_size;
    image_dst->height = warp_size;

}

uint8_t CV_is_contour_convex(CV_Contours *contours, size_t contour_idx) {

    uint8_t orientation = 0;
    uint8_t convex = 1;
    size_t len = CV_contours_length(contours, contour_idx);
    size_t i = 0;
    size_t j = 0;
    size_t cur_pt_x, cur_pt_y, prev_pt_x, prev_pt_y, dxdy0, dydx0, dx0, dy0, dx, dy;

    CV_contours_get(contours, contour_idx, len - 1, &prev_pt_x, &prev_pt_y);
    CV_contours_get(contours, contour_idx, 0, &cur_pt_x, &cur_pt_y);

    for (; i < len; ++i) {
        if (++j == len) j = 0;

        prev_pt_x = cur_pt_x;
        prev_pt_y = cur_pt_y;

        dxdy0 = dx * dy0;
        dydx0 = dy * dx0;

        orientation |= dydx0 > dxdy0 ? 1 : (dydx0 < dxdy0 ? 2 : 3);

        if (orientation == 3) {
            convex = 0;
            break;
        }

        dx0 = dx;
        dy0 = dy;

    }

    return convex;

}

float CV_perimeter(CV_Contours *contours, size_t contour_idx) {

    size_t len = CV_contours_length(contours, contour_idx);
    size_t i = 0;
    size_t j = len - 1;

    float p = 0.0;
    size_t dx, dy;
    size_t ix, iy, jx, jy;

    for (; i < len; j = i++) {
        CV_contours_get(contours, contour_idx, i, &ix, &iy);
        CV_contours_get(contours, contour_idx, j, &jx, &jy);
        dx = ix - jx;
        dy = iy - jy;

        p += sqrt(dx * dx + dy * dy);
    }

    return p;

}

float CV_min_edge_length(CV_Contours *contours, size_t contour_idx) {

    size_t len = CV_contours_length(contours, contour_idx);
    size_t i = 0;
    size_t j = len - 1;

    float min = INFINITY;
    float d, dx, dy;

    size_t ix, iy, jx, jy;

    for (; i < len; j = i++) {

        CV_contours_get(contours, contour_idx, i, &ix, &iy);
        CV_contours_get(contours, contour_idx, j, &jx, &jy);

        dx = ix - jx;
        dy = iy - jy;

        d = dx * dx + dy * dy;

        if (d < min) min = d;

    }

    return sqrt(min);

}

size_t CV_count_nonzero(CV_Image image_src,
        size_t x, size_t y,
        size_t w, size_t h) {

    uint8_t *src = image_src.data;
    size_t pos = x + (y * image_src.width);
    size_t span = image_src.width - w;
    size_t nz = 0;
    size_t i, j;

    for(i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            if (src[pos++] != 0) ++nz;
        }

        pos += span;
    }

    return nz;

}

void CV_binary_border(CV_Image image_src, CV_Image *image_dst) {

    assert(image_src.width == image_dst->width && image_src.height == image_dst->height);

    uint8_t *src = image_src.data;
    uint8_t *dst = image_dst->data;
    size_t height = image_src.height;
    size_t width = image_src.width;

    size_t pos_dst = 0;
    size_t pos_src = 0;
    ssize_t i, j;

    for (j = -2; j < width; ++j) {
        dst[pos_dst++] = 0;
    }

    for (i = 0; i < height; ++i) {
        dst[pos_dst++] = 0;

        for (j = 0; j < width; ++j) {
            dst[pos_dst++] = (0 == src[pos_src++] ? 0 : 1);
        }

        dst[pos_dst++] = 0;
    }

    for (j = -2; j < width; ++j) {
        dst[pos_dst++] = 0;
    }

}
