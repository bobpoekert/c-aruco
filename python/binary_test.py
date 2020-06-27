import au_cv
import test_utils as t
import numpy as np

test = au_cv.threshold(t.test_gray, au_cv.otsu(t.test_gray))
binary = au_cv.binary_border(test).reshape((test.shape[0] + 2, test.shape[1] + 2))

from repl_utils import show_image

viz = np.zeros(binary.shape, dtype=np.uint8)
viz[binary != 0] = 255

show_image(viz)
