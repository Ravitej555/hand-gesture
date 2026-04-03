import ctypes
import os
import threading
import time

# When True, do not send real keyboard events. Useful for debugging freezes.
TEST_MODE = os.environ.get("KEYINPUT_TEST_MODE", "0") != "0"

# Use virtual-key codes (VK) which are more compatible with most games
keys = {
    "w": 0x57,  # 'W'
    "a": 0x41,  # 'A'
    "s": 0x53,  # 'S'
    "d": 0x44,  # 'D'
    # Arrow keys (virtual-key codes)
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
}

# Alias common action names (used by steering.py) to WASD so games bound to
# WASD controls (forward=ahead: 'w', left: 'a', right: 'd', back: 's') will
# respond when steering.py calls press_key('up'/'left'/'right'/'down'). This
# preserves the arrow-key mappings above in case they're needed elsewhere.
keys.setdefault('up', keys.get('w', keys.get('up')))
keys.setdefault('down', keys.get('s', keys.get('down')))
keys.setdefault('left', keys.get('a', keys.get('left')))
keys.setdefault('right', keys.get('d', keys.get('right')))

PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def press_key(key):
    # safe guard: if testing, don't send input
    if TEST_MODE:
        # quick debug print without blocking
        try:
            sc = keys[key]
        except Exception:
            sc = None
        print(f"[keyinput TEST_MODE] press {key} (scan={sc})")
        return

    try:
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        # use virtual-key in wVk and no SCANCODE flag
        ii_.ki = KeyBdInput(keys[key], 0, 0x0000, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    except KeyError:
        # unknown key, ignore
        return


def press_and_release(key, hold_ms=0.12):
    """Press a key and release it after hold_ms seconds in a background thread.
    This avoids blocking callers who handle frame processing.
    """
    if TEST_MODE:
        try:
            sc = keys[key]
        except Exception:
            sc = None
        print(f"[keyinput TEST_MODE] press_and_release {key} (scan={sc}) hold={hold_ms}s")
        return

    def _worker():
        try:
            press_key(key)
            time.sleep(hold_ms)
            release_key(key)
        except Exception:
            # be resilient: don't crash worker thread
            return

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def focus_window(title):
    """Try to bring a window with the exact `title` to the foreground.

    Returns True if a window was found and foregrounded, False otherwise.
    """
    if not title:
        return False
    if TEST_MODE:
        print(f"[keyinput TEST_MODE] focus_window {title}")
        return False

    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, title)
        if hwnd == 0:
            return False
        # SW_SHOW = 5
        user32.ShowWindow(hwnd, 5)
        user32.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False


def focus_window_by_hwnd(hwnd):
    """Bring the provided HWND to the foreground. Returns True on success."""
    if not hwnd:
        return False
    if TEST_MODE:
        print(f"[keyinput TEST_MODE] focus_window_by_hwnd {hwnd}")
        return False
    try:
        user32 = ctypes.windll.user32
        user32.ShowWindow(hwnd, 5)
        user32.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False


def find_window_by_substring(substr):
    """Search top-level windows for one whose title contains `substr` (case-insensitive).
    Returns the HWND of the first match or 0 if none found.
    """
    if not substr:
        return 0
    if TEST_MODE:
        print(f"[keyinput TEST_MODE] find_window_by_substring {substr}")
        return 0

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    EnumWindows = user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    GetWindowTextW = user32.GetWindowTextW
    GetWindowTextLengthW = user32.GetWindowTextLengthW
    IsWindowVisible = user32.IsWindowVisible

    found = {'hwnd': 0}

    def foreach_window(hwnd, lParam):
        try:
            if not IsWindowVisible(hwnd):
                return True
            length = GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value
            if substr.lower() in title.lower():
                found['hwnd'] = hwnd
                return False  # stop enumeration
        except Exception:
            pass
        return True

    try:
        EnumWindows(EnumWindowsProc(foreach_window), 0)
    except Exception:
        return 0

    return found['hwnd'] if found['hwnd'] else 0

def release_key(key):
    if TEST_MODE:
        try:
            sc = keys[key]
        except Exception:
            sc = None
        print(f"[keyinput TEST_MODE] release {key} (scan={sc})")
        return

    try:
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        # set KEYEVENTF_KEYUP flag for key release
        ii_.ki = KeyBdInput(keys[key], 0, 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    except KeyError:
        return
