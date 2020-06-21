import test_utils as t
import au_cv
import numpy as np

contours = au_cv.Contours.find(au_cv.adaptive_threshold(t.test_gray, 2, 7))
viz = np.zeros_like(t.test_gray)

#for contour in contours:
#    viz[contour] = 255

print(np.transpose([contours.xs, contours.ys]))

viz[contours.xs, contours.ys] = 255

from repl_utils import show_image
show_image(viz)
