"""Hand-gesture -> game control (MediaPipe + OpenCV)

This file captures camera frames, detects hands with MediaPipe, and
maps gestures to arrow keys. Optional screen-capture based game-state
detection can pause input while the game's image is static (e.g. paused).

The screen-state detector is optional and uses `mss` if available. If
`mss` is not installed the detector is disabled (no extra dependency).
"""

import math
import os
import threading
import queue
import time
from collections import deque
import keyinput
import cv2
import mediapipe as mp 
try:
    import mss
    import numpy as np
    MSS_AVAILABLE = True
except Exception:
    MSS_AVAILABLE = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
font = cv2.FONT_HERSHEY_SIMPLEX

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Toggle heavy drawing on/off to reduce CPU cost
DRAW_LANDMARKS = True

# Optional: set target window substring to find and bring to foreground once at startup.
TARGET_WINDOW_SUBSTRING = os.environ.get('KEYINPUT_TARGET_SUBSTRING', '')
TARGET_HWND = 0

# Optional screen-state detection
# Set env KEYINPUT_ENABLE_SCREEN_DETECTION=1 to enable (requires `mss` and `numpy`).
ENABLE_SCREEN_DETECTION = os.environ.get('KEYINPUT_ENABLE_SCREEN_DETECTION', '0') == '1' and MSS_AVAILABLE
SCREEN_SAMPLE_INTERVAL = 0.8  # seconds between samples
SCREEN_DIFF_THRESH = 12.0     # mean absolute diff threshold to consider as "changed"


def get_window_rect(hwnd):
    """Return (left, top, right, bottom) for a HWND, or None on failure."""
    try:
        import ctypes
        rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        return None


def capture_window_bbox(hwnd):
    """Capture a window by HWND using mss and return a numpy BGR image, or None."""
    if not MSS_AVAILABLE:
        return None
    rect = get_window_rect(hwnd)
    if not rect:
        return None
    left, top, right, bottom = rect
    w = max(1, right - left)
    h = max(1, bottom - top)
    bbox = {'left': left, 'top': top, 'width': w, 'height': h}
    try:
        with mss.mss() as s:
            img = s.grab(bbox)
            arr = np.array(img)  # BGRA
            # convert BGRA -> BGR
            return arr[:, :, :3]
    except Exception:
        return None


def screen_monitor_loop(hwnd, stop_event, state):
    """Thread loop: periodically capture the window and compare frames.
    Updates `state['running'] = True/False` depending on whether image changes."""
    if not MSS_AVAILABLE or hwnd is None:
        return
    prev = None
    while not stop_event.is_set():
        cur = capture_window_bbox(hwnd)
        if cur is None:
            state['running'] = True
            time.sleep(SCREEN_SAMPLE_INTERVAL)
            continue
        gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        if prev is None:
            prev = gray
            state['running'] = True
        else:
            # compute mean absolute difference
            diff = cv2.absdiff(prev, gray)
            mad = float(diff.mean())
            # if small change -> static/paused
            state['running'] = mad >= SCREEN_DIFF_THRESH
            prev = gray
        time.sleep(SCREEN_SAMPLE_INTERVAL)


def start_camera_reader(cap, frame_queue, stop_event):
    """Continuously read frames from the camera and keep the latest frame in the queue."""
    while not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue
        try:
            if frame_queue.full():
                _ = frame_queue.get_nowait()
        except Exception:
            pass
        frame_queue.put(frame)


def processing_loop(frame_queue, stop_event, game_state):
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2) as hands:

        # simple FPS counter (updates once per second)
        fps_counter = 0
        fps_last = time.time()
        fps = 0

        # smoothing history to reduce jitter (store recent midpoints)
        history = deque(maxlen=4)

        # stateful pressed keys to hold while gesture persists
        pressed = {'left': False, 'right': False, 'up': False, 'down': False}

        def apply_action(action):
            # don't send inputs if game is detected as not running
            if not game_state.get('running', True):
                # ensure no keys are held
                for k in list(pressed.keys()):
                    if pressed[k]:
                        try:
                            keyinput.release_key(k)
                        except Exception:
                            pass
                        pressed[k] = False
                return

            keys = ['left', 'right', 'up', 'down']
            for k in keys:
                if k != action and pressed.get(k, False):
                    try:
                        keyinput.release_key(k)
                    except Exception:
                        pass
                    pressed[k] = False
            if action in pressed and not pressed[action]:
                try:
                    keyinput.press_key(action)
                except Exception:
                    pass
                pressed[action] = True

        def release_all():
            for k, v in list(pressed.items()):
                if v:
                    try:
                        keyinput.release_key(k)
                    except Exception:
                        pass
                    pressed[k] = False

        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            t0 = time.time()
            frame = cv2.resize(frame, (640, 480))
            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            imageHeight, imageWidth = rgb.shape[:2]

            frame.flags.writeable = True
            out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            co = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if DRAW_LANDMARKS:
                        mp_drawing.draw_landmarks(
                            out, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    try:
                        wl = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        pixel = mp_drawing._normalized_to_pixel_coordinates(wl.x, wl.y, imageWidth, imageHeight)
                        if pixel is not None:
                            co.append(list(pixel))
                    except Exception:
                        continue

            try:
                desired_action = None
                if len(co) == 2:
                    xm, ym = (co[0][0] + co[1][0]) / 2.0, (co[0][1] + co[1][1]) / 2.0
                    history.append((xm, ym))
                    xm = sum(h[0] for h in history) / len(history)
                    ym = sum(h[1] for h in history) / len(history)

                    # geometry tests (re-used from original logic)
                    radius = 150
                    try:
                        m = (co[1][1] - co[0][1]) / (co[1][0] - co[0][0])
                    except Exception:
                        m = None
                    if m is not None:
                        # compute intersection tests
                        a = 1 + m ** 2
                        b = -2 * xm - 2 * co[0][0] * (m ** 2) + 2 * m * co[0][1] - 2 * m * ym
                        c = xm ** 2 + (m ** 2) * (co[0][0] ** 2) + co[0][1] ** 2 + ym ** 2 - 2 * co[0][1] * ym - 2 * co[0][1] * co[0][0] * m + 2 * m * ym * co[0][0] - 22500
                        disc = b ** 2 - 4 * a * c
                        if disc >= 0:
                            xa = (-b + disc ** 0.5) / (2 * a)
                            xb = (-b - disc ** 0.5) / (2 * a)
                            ya = m * (xa - co[0][0]) + co[0][1]
                            yb = m * (xb - co[0][0]) + co[0][1]
                            if m != 0:
                                ap = 1 + ((-1 / m) ** 2)
                                bp = -2 * xm - 2 * xm * ((-1 / m) ** 2) + 2 * (-1 / m) * ym - 2 * (-1 / m) * ym
                                cp = xm ** 2 + ((-1 / m) ** 2) * (xm ** 2) + ym ** 2 + ym ** 2 - 2 * ym * ym - 2 * ym * xm * (-1 / m) + 2 * (-1 / m) * ym * xm - 22500
                                discp = bp ** 2 - 4 * ap * cp
                                if discp >= 0:
                                    xap = (-bp + discp ** 0.5) / (2 * ap)
                                    xbp = (-bp - discp ** 0.5) / (2 * ap)
                                    yap = (-1 / m) * (xap - xm) + ym
                                    ybp = (-1 / m) * (xbp - xm) + ym

                                    cv2.circle(out, (int(xm), int(ym)), radius=radius, color=(195, 255, 62), thickness=8)
                                    cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), (195, 255, 62), 10)
                                    VERTICAL_DIFF_THRESH = 65
                                    if co[0][0] > co[1][0] and co[0][1] > co[1][1] and co[0][1] - co[1][1] > VERTICAL_DIFF_THRESH:
                                        desired_action = 'left'
                                        cv2.putText(out, "Turn left", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                                    elif co[1][0] > co[0][0] and co[1][1] > co[0][1] and co[1][1] - co[0][1] > VERTICAL_DIFF_THRESH:
                                        desired_action = 'left'
                                        cv2.putText(out, "Turn left", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                                    elif co[0][0] > co[1][0] and co[1][1] > co[0][1] and co[1][1] - co[0][1] > VERTICAL_DIFF_THRESH:
                                        desired_action = 'right'
                                        cv2.putText(out, "Turn right", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                                    elif co[1][0] > co[0][0] and co[0][1] > co[1][1] and co[0][1] - co[1][1] > VERTICAL_DIFF_THRESH:
                                        desired_action = 'right'
                                        cv2.putText(out, "Turn right", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                                    else:
                                        desired_action = 'up'
                                        cv2.putText(out, "keep straight", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                elif len(co) == 1:
                    desired_action = 'down'
                    cv2.putText(out, "keeping back", (50, 50), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

                if desired_action is not None:
                    apply_action(desired_action)
                else:
                    release_all()
            except Exception:
                pass

            t1 = time.time()
            proc_ms = int((t1 - t0) * 1000)
            fps_counter += 1
            if t1 - fps_last >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_last = t1

            cv2.putText(out, f"proc: {proc_ms}ms", (10, out.shape[0] - 10), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, f"fps: {fps}", (10, out.shape[0] - 30), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            # indicate screen-detected state
            if ENABLE_SCREEN_DETECTION:
                state_text = 'running' if game_state.get('running', True) else 'static'
                cv2.putText(out, f"game: {state_text}", (10, 30), font, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Hands', cv2.flip(out, 1))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break


if __name__ == '__main__':
    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    game_state = {'running': True}

    # If configured, locate target window by substring and focus it once at startup
    if TARGET_WINDOW_SUBSTRING:
        try:
            hwnd = keyinput.find_window_by_substring(TARGET_WINDOW_SUBSTRING)
            if hwnd:
                TARGET_HWND = hwnd
                keyinput.focus_window_by_hwnd(hwnd)
        except Exception:
            TARGET_HWND = 0

    # optionally start screen monitor
    screen_thread = None
    if ENABLE_SCREEN_DETECTION and TARGET_HWND:
        screen_thread = threading.Thread(target=screen_monitor_loop, args=(TARGET_HWND, stop_event, game_state), daemon=True)
        screen_thread.start()

    reader = threading.Thread(target=start_camera_reader, args=(cap, frame_queue, stop_event), daemon=True)
    processor = threading.Thread(target=processing_loop, args=(frame_queue, stop_event, game_state), daemon=True)

    reader.start()
    processor.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()

    reader.join(timeout=1.0)
    processor.join(timeout=1.0)
    if screen_thread:
        screen_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
