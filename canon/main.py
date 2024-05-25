import pygame
from math import sin, cos, pi
from random import randint

pygame.init()
WIDTH, HEIGHT = 1152, 648
TITLE = "canon shooter"
MAX_FPS = 30
BACKGROUND_COLOR = (255, 255, 255)

# screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(TITLE)
pygame.key.set_repeat(100, 50)
clock = pygame.time.Clock()


def sub(x1, x2):
    return (
        x1[0] - x2[0],
        x1[1] - x2[1],
    )

def norm(x):
    return norm_sq(x) ** 0.5

def norm_sq(x):
    return x[0] * x[0] + x[1] * x[1]

def circle_intersect(c1, r1, c2, r2):
    min_separation = r1 + r2
    return norm_sq(sub(c1, c2)) <= min_separation * min_separation

def clamp(x, lower, upper):
    """ clamp a number between lower and upper. lower <= return value <= upper """
    return max(lower, min(upper, x))

def clamp_vector(x, lower, upper):
    """ clamp the norm of a vector between lower and upper. lower <= norm(return value) <= upper """
    norm_x = norm(x)
    new_norm = clamp(norm_x, upper, lower)
    return (
        x[0] / norm_x * new_norm,
        x[1] / norm_x * new_norm,
    )


class Entity:

    def __init__(self, position, radius, color):
        self.position = position
        self.radius = radius
        self.color = color

    def update(self):
        pass

    def draw(self):
        pygame.draw.circle(screen, self.color, self.position, self.radius)


class Projectile(Entity):

    def __init__(self, position, radius, color, velocity):
        super().__init__(position, radius, color)
        self.velocity = velocity

    def update(self, acceleration, dt):
        self.velocity = clamp_vector((
            self.velocity[0] + dt * acceleration[0],
            self.velocity[1] + dt * acceleration[1],
        ), 0, 100)
        self.position = (
            clamp(self.position[0] + dt * self.velocity[0], self.radius, simulation.width - self.radius),
            clamp(self.position[1] + dt * self.velocity[1], self.radius, simulation.height - self.radius),
        )

        if self.position[1] == simulation.height - self.radius:
            # projectile is on the ground
            self.velocity = (
                self.velocity[0],
                0,
            )


class Canon(Entity):

    def __init__(self, position, radius, color, angle, barrel_color):
        super().__init__(position, radius, color)
        self.angle = angle
        self.barrel_color = barrel_color
        self.barrel_radius = self.radius / 4
        self.projectile = None
        self.propulsion = 100
        self.projectile_history = []

    def fire(self):
        self.projectile = Projectile(
            self.position, self.barrel_radius - 1, (0, 0, 0),
            (self.propulsion * cos(self.angle), -self.propulsion * sin(self.angle))
        )
        self.projectile_history = [self.projectile.position]

    def update(self, projectile_acceleration, dt):
        if not (self.projectile is None):
            self.projectile.update(projectile_acceleration, dt)
            self.projectile_history.append(self.projectile.position)

    def draw(self):
        barrel_position = (
            self.position[0] + self.radius * cos(self.angle),
            self.position[1] - self.radius * sin(self.angle), # y is flipped
        )
        pygame.draw.circle(screen, self.barrel_color, barrel_position, self.barrel_radius)
        super().draw()
        if not (self.projectile is None):
            pygame.draw.lines(screen, self.projectile.color, False, self.projectile_history)
            self.projectile.draw()


class Obstacle(Entity):
    
    def __init__(self, position, radius, color):
        super().__init__(position, radius, color)

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
        self.gravity = (0, 15)
        self.dt = 1 / MAX_FPS

    def draw(self):
        self.canon.draw()
        for obstacle in self.obstacles:
            obstacle.draw()
        self.target.draw()

    def update(self):
        self.canon.update(self.gravity, self.dt)
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
                    elif event.key == pygame.K_SPACE:
                        simulation.canon.fire()
                    elif event.key == pygame.K_LEFT:
                        simulation.canon.angle += 0.1
                    elif event.key == pygame.K_RIGHT:
                        simulation.canon.angle -= 0.1
                # elif event.type == pygame.VIDEORESIZE:
                #     self.width, self.height = event.w, event.h
                #     screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.quit()


obstacles = []
OBSTACLE_COUNT = 5
OBSTACLE_RADIUS = 40
OBSTACLE_COLOR = (100, 255, 100)
for i in range(OBSTACLE_COUNT):
    space_found = False
    x, y = 0, 0
    while (not space_found):
        x = randint(OBSTACLE_RADIUS, WIDTH - OBSTACLE_RADIUS)
        y = randint(OBSTACLE_RADIUS, HEIGHT - OBSTACLE_RADIUS)
        for obstacle in obstacles:
            if circle_intersect((x, y), OBSTACLE_RADIUS, obstacle.position, OBSTACLE_RADIUS):
                break
        else:
            space_found = True
    obstacles.append(Obstacle((x, y), OBSTACLE_RADIUS, OBSTACLE_COLOR))

canon = Canon((50, HEIGHT - 25), 25, (255, 0, 0), pi / 4, (100, 100, 100))
target = Target((WIDTH - 50, 50), 20, (100, 100, 255))
simulation = Simulation(canon, obstacles, target, WIDTH, HEIGHT)

simulation.run(screen)