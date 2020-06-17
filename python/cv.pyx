from libc.stdint cimport *
import numpy as np
cimport numpy as np


cdef extern from "cv.h":

    typedef struct CV_Image:
        size_t width
        size_t height
        uint8_t *data

    typedef struct CV_Contours:
        size_t *contour_starts
        size_t n_contours
        size_t *xs
        size_t *ys
        size_t max_n_contours
        size_t array_length

    void CV_get_perspective_transform(CV_Contours *c, size_t contour_idx, size_t warp_size, CV_PerspectiveTransform res)

    void CV_threshold(CV_Image inp, uint8_t thresh)
    void CV_adaptive_threshold(CV_Image inp, CV_Image outp, uint8_t kernel_size, uint8_t thresh)
    uint8_t CV_otsu(CV_Image src)
    void CV_stack_box_blur(CV_Image image_src, CV_Image image_dst, uint8_t kernel_size)

    void CV_gaussian_blur(
            CV_Image src_image, CV_Image dst_image, CV_Image mean_image, uint8_t kernel_size)

    void CV_find_contours(CV_Image image_src, CV_Image *binary, CV_Contours *res)

    void CV_approx_poly_dp(CV_Contours *contour, size_t contour_idx, float epsilon, CV_Contours *res)

    void CV_warp(CV_Image image_src, CV_Image *image_dst, CV_Contours *contours, size_t contour_idx, size_t warp_size)

    uint8_t CV_is_contour_convex(CV_Contours *contours, size_t contour_idx)

    float CV_perimeter(CV_Contours *contours, size_t contour_idx)
    size_t CV_count_nonzero(CV_Image image_src,
            size_t x, size_t y,
            size_t w, size_t h)
    void CV_binary_border(CV_Image image_src, CV_Image *image_dst)

cdef image_from_array(np.ndarray[np.uint32_t, ndim=2, mode='c'] inp, CV_Image *outp):
    w, h = inp.shape
    outp->width = w
    outp->height = h
    outp->data = &inp[0,0]

def gaussian_blur(np.ndarray[np.uint32_t, ndim=2, mode='c'] inp, kernel_size):
    cdef np.ndarray[np.uint32_t, ndim=2, mode='c'] res = np.zeros_like(inp)
    cdef np.ndarray[np.uint32_t, ndim=2, mode='c'] mean = np.zeros_like(inp)

    cdef CV_Image inp_image
    cdef CV_Image res_image
    cdef CV_Image mean_image

    image_from_array(inp, &inp_image)
    image_from_array(res, &res_image)
    image_from_array(mean, &mean_image)

    CV_guassian_blur(inp_image, res_image, mean_image, kernel_size)

    return (res, mean)

def stack_box_blur(np.ndarray[np.uint32_t, ndim=2, mode='c'] inp, kernel_size):
    cdef np.ndarray[np.uint32_t, ndim=2, mode='c'] res = np.zeros_like(inp)

    cdef CV_Image inp_image
    cdef CV_Image res_image

    image_from_array(inp, &inp_image)
    image_from_array(res, &res_image)

    CV_stack_box_blur(inp_image, res_image)

    return res

def threshold_inplace(np.ndarray[np.uint32_t, ndim=2, mode='c'] inp, thresh):

    cdef CV_Image inp_image
    image_from_array(inp, &inp_image)

    CV_threshold(inp_image, thresh)

def threshold(inp, thresh):
    res = inp.copy()
    threshold_inplace(res, thresh)
    return res

def adaptive_threshold(np.ndarray[np.uint32_t, ndim=2, mode='c'] inp, kernel_size, thresh):
    cdef np.ndarray[np.uint32_t, ndim=2, mode='c'] res = np.zeros_like(inp)

    cdef CV_Image inp_image
    cdef CV_Image res_image

    image_from_array(inp, &inp_image)
    image_from_array(res, &res_image)

    CV_adaptive_threshold(inp_image, res_image, kernel_size, thresh)

    return res

def otsu_inplace(np.ndarray[np.uint32_t, ndim=2, mode='c'] inp):

    cdef CV_Image inp_image
    image_from_array(inp, &inp_image)

    CV_otsu(inp_image)

def otsu(inp):
    res = inp.copy()
    otsu_inplace(res)
    return res


def _find_contours(np.ndarray[np.uint32_t, ndim=2, mode='c'] inp):
    cdef CV_Image inp_image
    image_from_array(inp, &inp_image)

    max_size = inp.shape[0] * inp.shape[1]

    cdef np.ndarray[np.size_t, ndim=1, mode='c'] xs = np.zeros((max_size,), dtype=np.uint64)
    cdef np.ndarray[np.size_t, ndim=1, mode='c'] ys = np.zeros_like(xs)
    cdef np.ndarray[np.size_t, ndim=1, mode='c'] offsets = np.zeros_like(xs)

    CV_Contours res
    res.contour_starts = &offsets[0]
    res.xs = &xs[0]
    res.ys = &ys[0]
    res.max_n_contours = max_size
    res.n_contours = 0
    res.array_length = ys.shape[0]

    CV_find_contours(inp_image, binary_image, &res)

    idx_end = res.contour_starts[res.n_contours]

    return (xs[:idx_end], ys[:idx_end], offsets[:res.n_contours])

class Contours(object):

    def __init__(self, xs, ys, starts, n_contours):
        self.xs = xs
        self.ys = ys
        self.starts = start
        self.n_contours = n_contours

    def __len__(self):
        return self.n_contours

    def __getitem__(self, idx):
        start = self.starts[idx]
        length = self.starts[idx + 1] - start
        return (self.xs[start:(start + length)], self.ys[start:(start + length)])

    def __iter__(self):
        for i in range(self.n_contours):
            yield self[i]

    @classmethod
    def find(self, image):
        xs, ys, starts = _find_contours(image)
        assert xs.shape == ys.shape
        assert ys.shape == starts.shape
        return Contours(xs, ys, starts, starts.shape[0])

cdef init_single_contour(CV_Contours *contours, size_t *dummy_idx, xs, ys):
    contours->n_contours = 1
    contours->max_n_contours = 1
    contours->array_length = xs.shape[0]
    contours->xs = &xs[0]
    contours->ys = &ys[0]
    *dummy_idx = xs.shape[0]
    contours->contour_starts = dummy_idx

def approx_poly_dp(contour, epsilon):
    xs, ys = contour

    cdef size_t dummy_start = 0

    cdef CV_Contours inp
    init_single_contour(&inp, xs, ys)

    res_xs = np.zeros_like(xs)
    res_ys = np.zeros_like(ys)

    cdef CV_Contours res
    cdef size_t res_dummy
    init_single_contour(&res, &res_dummy, xs, ys)

    CV_approx_poly_dp(&inp, 0, epsilon, &res)

    return Contours(res_xs, res_ys, res_starts, res.n_contours)

def warp(np.ndarray[np.uint32_t, ndim=2, mode='c'] inp, contour, warp_size):

    res = np.zeros_like(inp)

    cdef CV_Image inp_image
    cdef CV_Image res_image

    image_from_array(inp, &inp_image)
    image_from_array(res, &res_image)

    contour_xs, contour_ys = contour

    cdef size_t dummy_length = contour_xs.shape[0]

    cdef CV_Contours contours
    cdef size_t c_dummy
    init_single_contour(&contours, &c_dummy, contour_xs, contour_ys)

    CV_warp(inp_image, &res_image, &contours, 0, warp_size)

    return res

def is_contour_convex(contour):

    xs, ys = contour

    cdef size_t dummy_start = xs.shape[0]

    cdef CV_Contours contours
    cdef size_t c_dummy
    init_single_contour(&contours, &c_dummy, xs, ys)

    return CV_is_contour_convex(&contours, 0) != 0

def perimeter(contour):

    xs, ys = contour

    cdef size_t dummy_start = xs.shape[0]

    cdef CV_Contours contours
    cdef size_t c_dummy
    init_single_contour(&contours, &c_dummy, xs, ys)

    return CV_perimeter(&contours, 0)

def count_nonzero(np.ndarray[np.uint32_t, ndims=2, mode='c'] inp,
        x, y, w, h):

    cdef CV_Image inp_image
    image_from_array(inp, &inp_image)

    return CV_count_nonzero(inp_image, x, y, w, h)

def binary_border(np.ndarray[np.uint32_t, ndims=2, mode='c'] inp):
    cdef np.ndarray[np.uint32_t, ndims=2, mode='c'] res = np.zeros_like(inp)

    cdef CV_Image inp_image
    cdef CV_Image res_image

    image_from_array(inp, &inp_image)
    image_from_array(res, &res_image)

    CV_binary_border(inp_image, &res_image)

    return res

def get_perspective_transform(py_contours, warp_size):

    cdef CV_Contours contours
    contours.xs = &py_contours.xs[0]
    contours.ys = &py_contours.ys[0]
    contours.contour_starts = &py_contours.starts[0]
    contours.array_length = py_contours.xs.shape[0]
    py_contours.n_contours = len(py_contours)
    py_contours.max_n_contours = len(py_contours)

    cdef CV_PerspectiveTransform res

    CV_get_perspective_transform(&contours, 0, warp_size, res)

    res_array = np.zeros((8,), dtype=np.size_t)
    for i in range(8):
        res_array[i] = res[i]

    return res_array
