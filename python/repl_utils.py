from PIL import Image
import numpy as np
import time

import pygame
import pygame.surfarray as pnp
pygame.init()
pygame.font.init()

def unpack_image(arr):
    if len(arr.shape) == 2:
        return np.transpose([arr, arr, arr])
    else:
        return arr.swapaxes(0, 1)

def show_image(img):
    arr = np.array(img)
    arr = unpack_image(arr)
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

def show_animation(image_generator):

    image_iter = iter(image_generator)
    next_image = next(image_iter)
    size = next_image.shape

    window = pygame.display.set_mode((size[1], size[0]))

    try:
        while 1:
            window.fill((0,0,0))
            window.blit(pnp.make_surface(unpack_image(next_image)), (0, 0))
            pygame.display.flip()

            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    return

            next_image = next(image_iter)

    except StopIteration:
        return

    finally:
        pygame.quit()
