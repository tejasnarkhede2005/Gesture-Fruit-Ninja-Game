import mediapipe as mp
import sys

print("MP_VERSION:", getattr(mp, "__version__", "N/A"))
print("MP_DIR:", [n for n in dir(mp) if not n.startswith("_")])
print("PY_VER:", sys.version)
