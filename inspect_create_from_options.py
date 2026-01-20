import mediapipe as mp
import inspect
vision = mp.tasks.vision
print('create_from_options callable:', callable(vision.HandLandmarker.create_from_options))
print('signature:', inspect.signature(vision.HandLandmarker.create_from_options))
print('doc:', vision.HandLandmarker.create_from_options.__doc__[:1000])
