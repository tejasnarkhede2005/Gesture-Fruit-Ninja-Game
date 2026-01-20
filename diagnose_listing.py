import mediapipe as mp
import os

print('mediapipe package file:', mp.__file__)
pkg_dir = os.path.dirname(mp.__file__)
print('package dir:', pkg_dir)
print(os.listdir(pkg_dir))
