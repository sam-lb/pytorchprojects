import pygame
import time
from math import sin, cos, pi, exp
from random import randint
from model import AgentBrain


pygame.init()
pygame.font.init()
WIDTH, HEIGHT = 1152, 648
TITLE = "canon shooter"
MAX_FPS = 30
BACKGROUND_COLOR = (255, 255, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(TITLE)
pygame.key.set_repeat(100, 50)
clock = pygame.time.Clock()
FONT = pygame.font.SysFont("Arial", 20)


# vector functions

def scale(x, k):
    return (
        x[0] * k,
        x[1] * k,
    )

def add(x1, x2):
    return (
        x1[0] + x2[0],
        x1[1] + x2[1],
    )

def sub(x1, x2):
    return (
        x1[0] - x2[0],
        x1[1] - x2[1],
    )

def norm(x):
    return norm_sq(x) ** 0.5

def norm_sq(x):
    return x[0] * x[0] + x[1] * x[1]

def normalize(x):
    norm_ = norm(x)
    return 0 if norm_ == 0 else x / norm(x)

def dot(x1, x2):
    return x1[0] * x2[0] + x1[1] * x2[1]

def project(x1, x2):
    """ project x1 onto x2 """
    return scale(x2, dot(x1, x2) / norm_sq(x2))

def reflect(x1, x2):
    """ reflect x1 across x2 """
    res = sub(scale(project(x1, x2), 2), x1)
    return res

def distance(x1, x2):
    return norm(sub(x1, x2))

def distance_sq(x1, x2):
    return norm_sq(sub(x1, x2))

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

# misc

def random_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)

def sigmoid(x):
    return 1 / (1 + exp(-x))


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

    def update(self, acceleration, dt, obstacles):
        self.velocity = clamp_vector((
            self.velocity[0] + dt * acceleration[0],
            self.velocity[1] + dt * acceleration[1],
        ), 0, 100)
        self.position = (
            clamp(self.position[0] + dt * self.velocity[0], self.radius, simulation.width - self.radius),
            clamp(self.position[1] + dt * self.velocity[1], self.radius, simulation.height - self.radius),
        )

        for obstacle in obstacles:
            if circle_intersect(self.position, self.radius, obstacle.position, obstacle.radius):
                self.resolve_collision(obstacle, dt)

        if self.position[1] == simulation.height - self.radius:
            # projectile is on the ground
            self.velocity = (
                self.velocity[0],
                0,
            )

    def resolve_collision(self, obstacle, dt):
        c1 = self.position
        r1 = self.radius
        c2 = obstacle.position
        r2 = obstacle.radius
        v = self.velocity

        # calculate the center of the projectile at the time of the collision
        A = norm_sq(v)
        B = 2 * ( v[0] * (c1[0] - c2[0]) + v[1] * (c1[1] - c2[1]) )
        C = c1[0] * c1[0] + c2[0] * c2[0] + c1[1] * c1[1] + c2[1] * c2[1] - 2 * (c1[0] * c2[0] + c1[1] * c2[1]) - (r1 + r2) * (r1 + r2)

        disc = (B * B - 4 * A * C) ** 0.5
        denom = 2 * A
        lambda_ = min((-B + disc) / denom, (-B - disc) / denom)
        res_vector = scale(v, lambda_)
        center_at_collision_time = add(c1, res_vector)

        # calculate the new velocity of the projectile after collision
        reflection_vector = sub(c2, center_at_collision_time)
        self.velocity = scale(reflect(v, reflection_vector), -obstacle.restitution)

        # move the projectile to where it should be at the end of the timestep
        self.position = add(center_at_collision_time, scale(self.velocity, abs(lambda_ * dt)))


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

    def update(self, projectile_acceleration, dt, obstacles):
        if not (self.projectile is None):
            self.projectile.update(projectile_acceleration, dt, obstacles)
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
    
    def __init__(self, position, radius, color, restitution=1):
        super().__init__(position, radius, color)
        self.restitution = restitution


class Target(Entity):

    def __init__(self, position, radius, color):
        super().__init__(position, radius, color)


class Simulation:

    def __init__(self, obstacles, target, width, height, population_size, mutation_rate, generations, load_model=False):
        self.canons = []
        self.agents = []
        self.obstacles = obstacles
        self.target = target
        self.running = False
        self.width, self.height = width, height
        self.gravity = (0, 35)
        self.dt = 1 / MAX_FPS

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.generation_length = 20 # seconds
        self.generation = 1
        self.best_fitness = None

        self.obstacle_positions = []
        for obstacle in self.obstacles:
            self.obstacle_positions.append(obstacle.position[0] / self.width)
            self.obstacle_positions.append(obstacle.position[1] / self.height)

        self.load_model = load_model
        self.initialize_population()

    def initialize_population(self):
        for _ in range(self.population_size):
            agent = Agent(
                (50, HEIGHT - 25), 25, random_color(), pi / 4, OBSTACLE_COUNT, self.mutation_rate
            )
            self.agents.append(agent)
            self.canons.append(agent.canon)

    def draw(self):
        for canon in self.canons:
            canon.draw()
        for obstacle in self.obstacles:
            obstacle.draw()
        self.target.draw()

        screen.blit(FONT.render("Generation: {}".format(self.generation), False, (0, 0, 0)), (10, 10))
        screen.blit(FONT.render("Best fitness: {}".format(self.best_fitness), False, (0, 0, 0)), (10, 50))

    def new_generation(self):
        self.agents.sort(key=lambda agent: agent.evaluate_fitness())
        best_individual = self.agents[-1]
        best_fitness = best_individual.evaluate_fitness()
        print("best individual fitness: {}".format(best_fitness))
        if (self.best_fitness is None or best_fitness > self.best_fitness):
            best_individual.brain.save("best_brain.pth")
            self.best_fitness = best_fitness

        self.generation += 1
        print("now running generation {}".format(self.generation))

        new_agents = []
        survivors = self.agents[self.population_size // 2:]
        for agent in survivors:
            new_agents.append(Agent(
                agent.canon.position, agent.canon.radius, agent.canon.color, pi / 4,
                agent.brain.num_obstacles, agent.brain.mutation_rate, False
            ))
            new_agents[-1].brain = agent.brain
        print(len(self.agents), len(survivors))

        for i in range(0, len(survivors) - 1):
            # new_agents.extend(survivors[i].crossover(survivors[i+1]))
            new_agents.append(survivors[i].crossover(survivors[i+1])[0])
        new_agents.append(survivors[0].crossover(survivors[-1])[0])

        self.agents = new_agents

        print(len(self.canons), "before")
        self.canons = []
        for agent in new_agents:
            self.canons.append(agent.canon)
        print(len(self.canons), "after")


    def update(self):
        for agent in self.agents:
            agent.act(self)
        for canon in self.canons:
            canon.update(self.gravity, self.dt, self.obstacles)
        self.draw()

    def run(self, screen):
        self.running = True
        timer = 0
        last_time = time.time()

        while self.running:
            screen.fill(BACKGROUND_COLOR)

            self.update()

            pygame.display.flip()
            clock.tick(MAX_FPS)
            new_time = time.time()
            timer += new_time - last_time
            last_time = new_time

            if timer >= self.generation_length:
                self.new_generation()
                timer = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        break
        pygame.quit()


class Agent:

    def __init__(self, position, radius, color, angle, num_obstacles, mutation_rate, make_brain=True):
        self.canon = Canon(position, radius, color, angle, (100, 100, 100))
        if make_brain: self.brain = AgentBrain(num_obstacles, mutation_rate)
        self.projectiles_fired = 0
        self.max_distance = distance((0, HEIGHT), (WIDTH, 0))
        self.closest_distance = self.max_distance
        self.hit_target = False

    def evaluate_fitness(self):
        distance_to_target = self.max_distance - self.closest_distance
        target_bonus = 10_000 if self.hit_target else 0
        projectile_penalty = self.projectiles_fired

        fitness = target_bonus + distance_to_target - projectile_penalty
        return fitness
    
    def act(self, simulation):
        inputs = []
        inputs.extend(simulation.obstacle_positions)
        inputs.append(simulation.target.position[0] / simulation.width)
        inputs.append(simulation.target.position[1] / simulation.height)
        inputs.append(self.canon.position[0] / simulation.width)
        inputs.append(self.canon.position[1] / simulation.height)
        inputs.append((self.canon.angle % (2 * pi)) / (2 * pi))
        inputs.append(sigmoid(self.canon.propulsion))

        if not (self.canon.projectile is None):
            self.closest_distance = min(distance(self.canon.projectile.position, simulation.target.position),
                                        self.closest_distance)
            if (circle_intersect(self.canon.projectile.position,
                                 self.canon.projectile.radius,
                                 simulation.target.position,
                                 simulation.target.radius)):
                self.hit_target = True

        actions = self.brain.decide(inputs)

        if actions[0] > 0:
            self.canon.fire()
            self.projectiles_fired += 1
        
        if actions[1] > 0:
            self.canon.angle += float(actions[2])

        # if actions[3] > 0:
        #     self.canon.propulsion += actions[4]

    def crossover(self, other):
        child1, child2 = (
            Agent(self.canon.position, self.canon.radius, self.canon.color,
                  pi / 4, self.brain.num_obstacles, self.brain.mutation_rate, False),
            Agent(self.canon.position, self.canon.radius, self.canon.color,
                  pi / 4, self.brain.num_obstacles, self.brain.mutation_rate, False),
        )
        # child1 = Agent(self.canon.position, self.canon.radius, self.canon.color,
        #           pi / 4, self.brain.num_obstacles, self.brain.mutation_rate, False)

        brain1, brain2 = self.brain.crossover(other.brain)
        child1.brain = brain1
        child2.brain = brain2
        
        # survivor = Agent(self.canon.position, self.canon.radius, self.canon.color, pi / 4,
        #              self.brain.num_obstacles, self.brain.mutation_rate, False)
        # survivor.brain = self.brain

        return child1, child2
        # return survivor, child1



obstacles = []
OBSTACLE_COUNT = 5
OBSTACLE_RADIUS = 40
OBSTACLE_COLOR = (100, 255, 100)

POPULATION_SIZE = 100
MUTATION_RATE = 0.05
GENERATIONS = 5

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
            space_found = distance_sq((x, y), (0, HEIGHT)) > 40000 # 200 pixel buffer
    obstacles.append(Obstacle((x, y), OBSTACLE_RADIUS, OBSTACLE_COLOR))

# canon = Canon((50, HEIGHT - 25), 25, (255, 0, 0), pi / 4, (100, 100, 100))
target = Target((WIDTH - 50, 50), 20, (100, 100, 255))
simulation = Simulation(obstacles, target, WIDTH, HEIGHT, POPULATION_SIZE, MUTATION_RATE, GENERATIONS)


# run simulation

simulation.run(screen)