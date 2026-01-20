import os
import mediapipe as mp
pkg_dir = os.path.dirname(mp.__file__)
modules_dir = os.path.join(pkg_dir, 'modules')
print('modules dir:', modules_dir)
print(os.listdir(modules_dir))
