import test_utils as t
import au_cv

from repl_utils import show_image

show_image(au_cv.stack_box_blur_inplace(t.test_gray, 5))
