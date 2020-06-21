from libc.stdint cimport *
import numpy as np
cimport numpy as np
from ctypes import c_size_t

cdef extern from "cv.h":

    ctypedef struct CV_Image:
        size_t width
        size_t height
        uint8_t *data

    ctypedef struct CV_Contours:
        size_t *contour_starts
        size_t n_contours
        size_t *xs
        size_t *ys
        size_t max_n_contours
        size_t array_length

    ctypedef size_t CV_PerspectiveTransform[8];

    void CV_get_perspective_transform(CV_Contours *c, size_t contour_idx, size_t warp_size, CV_PerspectiveTransform res)

    void CV_threshold(CV_Image inp, uint8_t thresh)
    void CV_adaptive_threshold(CV_Image inp, CV_Image blurred, uint8_t thresh)
    uint8_t CV_otsu(CV_Image src)
    void CV_stack_box_blur(CV_Image image_src, uint32_t radius)

    void CV_find_contours(CV_Image image_src, int16_t *binary, CV_Contours *res)

    void CV_approx_poly_dp(CV_Contours *contour, size_t contour_idx, float epsilon, CV_Contours *res)

    void CV_warp(CV_Image image_src, CV_Image *image_dst, CV_Contours *contours, size_t contour_idx, size_t warp_size)

    uint8_t CV_is_contour_convex(CV_Contours *contours, size_t contour_idx)

    float CV_perimeter(CV_Contours *contours, size_t contour_idx)
    size_t CV_count_nonzero(CV_Image image_src,
            size_t x, size_t y,
            size_t w, size_t h)
    void CV_binary_border(CV_Image image_src, int16_t *dst)

cdef image_from_array(inp, CV_Image *outp):
    cdef np.ndarray[np.uint8_t, ndim=2, mode='c'] cinp = inp
    w, h = inp.shape
    outp.width = w
    outp.height = h
    outp.data = &cinp[0,0]

def stack_box_blur_inplace(inp_py, kernel_size):
    assert kernel_size >= 2

    cdef np.ndarray[np.uint8_t, ndim=2, mode='c'] inp = inp_py

    cdef CV_Image inp_image
    image_from_array(inp, &inp_image)

    CV_stack_box_blur(inp_image, kernel_size)

def stack_box_blur(inp, kernel_size):
    res = inp.copy()
    stack_box_blur_inplace(res, kernel_size)
    return res

def threshold_inplace(np.ndarray[np.uint8_t, ndim=2, mode='c'] inp, thresh):

    cdef CV_Image inp_image
    image_from_array(inp, &inp_image)

    CV_threshold(inp_image, thresh)

def threshold(inp, thresh):
    res = inp.copy()
    threshold_inplace(res, thresh)
    return res

def adaptive_threshold(py_inp, kernel_size, thresh):
    res = py_inp.copy()
    cdef np.ndarray[np.uint8_t, ndim=2, mode='c'] inp = res
    cdef np.ndarray[np.uint8_t, ndim=2, mode='c'] blurred = res.copy()

    cdef CV_Image inp_image
    cdef CV_Image blurred_image

    image_from_array(inp, &inp_image)
    image_from_array(blurred, &blurred_image)

    CV_stack_box_blur(blurred_image, kernel_size)

    CV_adaptive_threshold(inp_image, blurred_image, thresh)

    return res

def otsu(np.ndarray[np.uint8_t, ndim=2, mode='c'] inp):

    cdef CV_Image inp_image
    image_from_array(inp, &inp_image)

    return CV_otsu(inp_image)

def _find_contours(np.ndarray[np.uint8_t, ndim=2, mode='c'] inp):

    max_size = inp.shape[0] * inp.shape[1]

    cdef np.ndarray[np.uint_t, ndim=1, mode='c'] xs = np.zeros((max_size,), dtype=c_size_t)
    cdef np.ndarray[np.uint_t, ndim=1, mode='c'] ys = np.zeros_like(xs)
    cdef np.ndarray[np.uint_t, ndim=1, mode='c'] offsets = np.zeros_like(xs)

    cdef CV_Image inp_image
    image_from_array(inp, &inp_image)

    cdef np.ndarray[np.int16_t, ndim=1, mode='c'] binary_data = np.zeros(((inp.shape[0] + 2) * (inp.shape[1] + 2),), dtype=np.int16)

    cdef CV_Contours res
    res.contour_starts = <size_t *> &offsets[0]
    res.xs = <size_t *> &xs[0]
    res.ys = <size_t *> &ys[0]
    res.max_n_contours = max_size
    res.n_contours = 0
    res.array_length = max_size

    CV_find_contours(inp_image, &binary_data[0], &res)

    if res.n_contours < 1:
        return None

    idx_end = res.contour_starts[res.n_contours]

    return (
            xs[:idx_end],
            ys[:idx_end],
            offsets[:res.n_contours])

class Contours(object):

    def __init__(self, xs, ys, starts, n_contours):
        self.xs = xs
        self.ys = ys
        self.starts = starts
        self.n_contours = n_contours

    def __len__(self):
        return self.n_contours

    def __getitem__(self, idx):
        if idx == 0:
            start = 0
            length = self.starts[0]
        else:
            start = self.starts[idx - 1]
            length = self.starts[idx] - start
        start = int(start)
        length = int(length)
        return (self.xs[start:(start + length)], self.ys[start:(start + length)])

    def __iter__(self):
        for i in range(self.n_contours):
            yield self[i]

    @classmethod
    def find(cls, image):
        contours = _find_contours(image)
        if contours is None:
            return []
        xs, ys, starts = contours
        return cls(xs, ys, starts, starts.shape[0])

cdef init_single_contour(CV_Contours *contours, size_t *dummy_idx,
        np.ndarray[np.uint_t, ndim=1, mode='c'] xs,
        np.ndarray[np.uint_t, ndim=1, mode='c'] ys):
    contours.n_contours = 1
    contours.max_n_contours = 1
    contours.array_length = xs.shape[0]
    contours.xs = <size_t *> &xs[0]
    contours.ys = <size_t *> &ys[0]
    dummy_idx[0] = xs.shape[0]
    contours.contour_starts = dummy_idx

def approx_poly_dp(contour, epsilon):
    xs, ys = contour

    cdef size_t dummy_start = 0

    cdef CV_Contours inp
    cdef size_t res_dummy
    init_single_contour(&inp, &res_dummy, xs, ys)

    res_xs = np.zeros_like(xs)
    res_ys = np.zeros_like(ys)
    cdef np.ndarray[np.uint_t, ndim=1, mode='c'] res_starts = np.zeros_like(xs)

    cdef CV_Contours res
    init_single_contour(&res, &res_dummy, res_xs, res_ys)
    res.contour_starts = <size_t *> &res_starts[0]

    CV_approx_poly_dp(&inp, 0, epsilon, &res)


    return Contours(res_xs, res_ys, res_starts, res.n_contours)

def warp(np.ndarray[np.uint8_t, ndim=2, mode='c'] inp, contour, warp_size):

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

def count_nonzero(np.ndarray[np.uint8_t, ndim=2, mode='c'] inp,
        x, y, w, h):

    cdef CV_Image inp_image
    image_from_array(inp, &inp_image)

    return CV_count_nonzero(inp_image, x, y, w, h)

def binary_border(np.ndarray[np.uint8_t, ndim=2, mode='c'] inp):
    cdef np.ndarray[np.int16_t, ndim=1, mode='c'] res =\
        np.zeros(((inp.shape[0] + 2) * (inp.shape[1] + 2),), dtype=np.int16)

    cdef CV_Image inp_image

    image_from_array(inp, &inp_image)

    CV_binary_border(inp_image, &res[0])

    return res

def get_perspective_transform(py_contours, warp_size):

    cdef np.ndarray[np.uint_t, ndim=1, mode='c'] xs = py_contours.xs
    cdef np.ndarray[np.uint_t, ndim=1, mode='c'] ys = py_contours.ys
    cdef np.ndarray[np.uint_t, ndim=1, mode='c'] starts = py_contours.starts

    cdef CV_Contours contours
    contours.xs = <size_t *> &xs[0]
    contours.ys = <size_t *> &ys[0]
    contours.contour_starts = <size_t *> &starts[0]
    contours.array_length = py_contours.xs.shape[0]
    py_contours.n_contours = len(py_contours)
    py_contours.max_n_contours = len(py_contours)

    cdef CV_PerspectiveTransform res

    CV_get_perspective_transform(&contours, 0, warp_size, res)

    res_array = np.zeros((8,), dtype=c_size_t)
    for i in range(8):
        res_array[i] = res[i]

    return res_array
