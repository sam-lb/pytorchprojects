



"""

model inputs:
obstacle positions
target position
canon position
canon angle
canon propulsion

outputs:
fire (boolean: sign(sigmoid y1)): whether to fire the canon
rotate_amount
propulsion_adjustment

fitness:
- for distance from closest projectile to target
+++ for hitting target
- for number of projectiles fired


"""


import torch