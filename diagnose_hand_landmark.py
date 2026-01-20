import os, mediapipe as mp
pkg_dir = os.path.dirname(mp.__file__)
hl_dir = os.path.join(pkg_dir, 'modules', 'hand_landmark')
print('hand_landmark dir:', hl_dir)
print(os.listdir(hl_dir))
for fname in os.listdir(hl_dir):
    if fname.endswith('.py'):
        print('\n---', fname, '---')
        print(open(os.path.join(hl_dir, fname),'r', encoding='utf-8', errors='ignore').read()[:1000])
