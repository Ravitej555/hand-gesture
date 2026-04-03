import importlib

mods = ['cv2', 'mediapipe', 'mss', 'numpy']
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m}: {e}")

if missing:
    print('IMPORT_FAILED', missing)
else:
    print('IMPORTS_OK')
