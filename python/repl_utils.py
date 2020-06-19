from PIL import Image
import numpy as np
import time

import pygame
import pygame.surfarray as pnp
pygame.init()
pygame.font.init()


def show_image(img):
    arr = np.array(img)
    if len(arr.shape) == 2:
        arr = np.transpose([arr, arr, arr])
    else:
        arr = arr.swapaxes(0, 1)
    window = pygame.display.set_mode((arr.shape[0], arr.shape[1]))
    window.fill((0,0,0))
    window.blit(pnp.make_surface(arr), (0, 0))
    pygame.display.flip()

    running = True
    while running:
        time.sleep(0.1)
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False
                break

    pygame.quit()
