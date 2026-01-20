import math

def circle_collision(x1, y1, r1, x2, y2):
    return math.hypot(x2-x1, y2-y1) <= r1
