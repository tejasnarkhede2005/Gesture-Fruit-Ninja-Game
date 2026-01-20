import mediapipe as mp
print(type(mp.tasks))
print('has vision attr:', hasattr(mp.tasks, 'vision'))
if hasattr(mp.tasks, 'vision'):
    print('vision type:', type(mp.tasks.vision))
    try:
        print('vision dir sample:', [n for n in dir(mp.tasks.vision) if not n.startswith('_')][:50])
    except Exception as e:
        print('vision dir failed:', e)
