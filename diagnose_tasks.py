import mediapipe as mp
from mediapipe import tasks
import inspect

print('tasks dir:', [n for n in dir(tasks) if not n.startswith('_')])
# try to inspect tasks.vision
try:
    from mediapipe.tasks import vision
    print('vision dir:', [n for n in dir(vision) if not n.startswith('_')])
except Exception as e:
    print('vision import failed:', e)
