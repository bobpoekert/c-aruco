import test_utils as t
import au_cv
import numpy as np

img = au_cv.adaptive_threshold(t.test_gray, 3, 7)
contours = au_cv.Contours.find(img, (img.shape[0] / 5))
viz = np.transpose([img, img, img]).swapaxes(0, 1)

#for contour in contours:
#    viz[contour] = 255

viz[contours.xs, contours.ys] = (0, 255, 0)

from repl_utils import show_image
show_image(viz)
