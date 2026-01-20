import cv2
import numpy as np
import time
from hand_tracking import HandTracker
from fruit import Fruit
from utils import circle_collision

WIDTH, HEIGHT = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Try to create the MediaPipe hand tracker. If unavailable or missing
# a task model, fall back to a simple motion-based fingertip proxy.
try:
    tracker = HandTracker()
    use_fallback = False
except Exception as e:
    print("HandTracker init failed:", e)
    tracker = None
    use_fallback = True
    # background subtractor for motion-based fallback
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
fruits = []
score = 0
spawn_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    finger = None
    if not use_fallback and tracker is not None:
        tracker.find_hand(frame)
        finger = tracker.get_index_tip(frame)
    else:
        # motion-based fallback: detect largest moving contour and use centroid as finger
        fg = backSub.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 500:
                M = cv2.moments(largest)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    finger = (cx, cy)
                    # draw a visible marker for the fallback fingertip
                    cv2.circle(frame, finger, 12, (0,255,255), -1)

    # spawn fruit
    if time.time() - spawn_time > 1.2:
        fruits.append(Fruit(WIDTH, HEIGHT))
        spawn_time = time.time()

    # update fruits
    for fruit in fruits[:]:
        fruit.update()
        fruit.draw(frame)

        if finger:
            if circle_collision(fruit.x, fruit.y, fruit.r, finger[0], finger[1]):
                fruits.remove(fruit)
                score += 10

        if not fruit.alive:
            fruits.remove(fruit)

    cv2.putText(frame, f"Score: {score}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    cv2.imshow("Gesture Fruit Ninja", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
