from PIL import Image
import skimage.color as color
import numpy as np

import au_cv

test_image = np.array(Image.open('test.jpg'))
test_gray = color.rgb2gray(test_image)
test_gray = (test_gray * 255).astype(np.uint8)
