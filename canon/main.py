import pygame

pygame.init()
WIDTH, HEIGHT = 200, 200
TITLE = "canon shooter"
MAX_FPS = 30
BACKGROUND_COLOR = (255, 255, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption(TITLE)
pygame.key.set_repeat(100, 50)
clock = pygame.time.Clock()


running = True
while running:

    screen.fill(BACKGROUND_COLOR)
    
    pygame.display.flip()
    clock.tick(MAX_FPS)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                break
        elif event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = event.w, event.h
            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                                    
pygame.quit()