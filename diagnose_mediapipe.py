import traceback

try:
    import mediapipe as mp
    print('MEDIAPIPE_IMPORT_OK')
except Exception:
    traceback.print_exc()
