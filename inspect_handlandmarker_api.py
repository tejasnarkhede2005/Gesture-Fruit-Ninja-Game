import mediapipe as mp
vision = mp.tasks.vision
print('HandLandmarker:', vision.HandLandmarker)
print('Options:', vision.HandLandmarkerOptions)
import inspect
print('\nHandLandmarker signature:')
print(inspect.signature(vision.HandLandmarker))
print('\nHandLandmarkerOptions signature:')
print(inspect.signature(vision.HandLandmarkerOptions))
print('\nRunningMode enum members:', [m for m in dir(vision.RunningMode) if not m.startswith('_')])
