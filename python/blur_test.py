import test_utils as t
import au_cv
import math

from repl_utils import show_animation, show_image

def blur_generator(inp):
    i = 2
    d = 2
    while 1:
        blurred = au_cv.stack_box_blur(inp, i)
        yield au_cv.threshold(blurred, au_cv.otsu(blurred))
        i += d
        if i > 20 or i <= 2:
            d = -d

show_animation(blur_generator(t.test_gray))
#show_image(au_cv.stack_box_blur(t.test_gray, 5))
