import random
import cv2

class Fruit:
    def __init__(self, width, height):
        self.r = random.randint(25, 35)
        self.x = random.randint(self.r, width - self.r)
        self.y = height + self.r
        self.speed = random.randint(8, 12)
        self.color = random.choice([
            (0, 0, 255),
            (0, 255, 0),
            (0, 255, 255),
            (255, 0, 0)
        ])
        self.alive = True

    def update(self):
        self.y -= self.speed
        if self.y < -self.r:
            self.alive = False

    def draw(self, frame):
        cv2.circle(frame, (self.x, self.y), self.r, self.color, -1)
        cv2.circle(frame, (self.x, self.y), self.r, (255,255,255), 2)
