import os, mediapipe as mp
pkg_dir = os.path.dirname(mp.__file__)
for root, dirs, files in os.walk(pkg_dir):
    for f in files:
        if 'hand' in f.lower():
            print(os.path.join(root, f))
