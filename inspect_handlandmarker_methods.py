import mediapipe as mp
vision = mp.tasks.vision
cls = vision.HandLandmarker
print([n for n in dir(cls) if not n.startswith('_')])
