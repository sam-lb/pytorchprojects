import pygame
from math import sin, cos, pi

pygame.init()
WIDTH, HEIGHT = 500, 500
TITLE = "canon shooter"
MAX_FPS = 30
BACKGROUND_COLOR = (255, 255, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption(TITLE)
pygame.key.set_repeat(100, 50)
clock = pygame.time.Clock()



class Entity:

    def __init__(self, position, radius, color):
        self.position = position
        self.radius = radius
        self.color = color

    def draw(self):
        pygame.draw.circle(screen, self.color, self.position, self.radius)


class Canon(Entity):

    def __init__(self, position, radius, color, angle, barrel_color):
        super().__init__(position, radius, color)
        self.angle = angle
        self.barrel_color = barrel_color
        self.barrel_radius = self.radius / 4

    def draw(self):
        barrel_position = (
            self.position[0] + self.radius * cos(self.angle),
            self.position[1] - self.radius * sin(self.angle), # y is flipped
        )
        pygame.draw.circle(screen, self.barrel_color, barrel_position, self.barrel_radius)
        super().draw()


class Target(Entity):

    def __init__(self, position, radius, color):
        super().__init__(position, radius, color)


class Simulation:

    def __init__(self, canon, obstacles, target, width, height):
        self.canon = canon
        self.obstacles = obstacles
        self.target = target
        self.running = False
        self.width, self.height = width, height

    def draw(self):
        self.canon.draw()
        for obstacle in self.obstacles:
            obstacle.draw()
        self.target.draw()

    def update(self):
        self.draw()

    def run(self, screen):
        self.running = True
        while self.running:
            screen.fill(BACKGROUND_COLOR)

            self.update()

            pygame.display.flip()
            clock.tick(MAX_FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        break
                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.quit()


canon = Canon((50, HEIGHT - 25), 25, (255, 0, 0), pi / 4, (100, 100, 100))
target = Target((WIDTH - 50, 50), 20, (100, 100, 255))
simulation = Simulation(canon, [], target, WIDTH, HEIGHT)

simulation.run(screen)