"""
Hand Gesture Cursor Controller — v3
MediaPipe hand tracking + PyAutoGUI, tuned for Mac (mediapipe==0.10.9, Python 3.11).

Gestures (unchanged from v2)
  Move        index finger only      -> cursor follows your hand
  Click       pinch thumb + index    -> quick pinch = click (pinch twice = double-click)
  Drag        pinch + hold + move    -> holds the click down while you move, open to drop
  Right click pinch thumb + middle   -> different finger so it never clashes with a click
  Scroll      index + middle up      -> raise hand = scroll up, lower hand = scroll down
  Zoom        index + middle + ring  -> raise hand = zoom in, lower hand = zoom out
  Pause       open palm (all 5)      -> freezes the cursor so you can rest your hand
  Quit        press Q in the window

What's new in v3
  * One Euro filter    adaptive smoothing: rock-steady when your hand is still,
                       near-zero lag when it moves fast (replaces the fixed EMA)
  * Stable anchor      cursor tracks the index KNUCKLE (landmark 5), not the tip,
                       so curling your finger to pinch no longer drags the cursor
  * Click freeze       cursor locks in place for the first moment of a pinch,
                       so clicks land exactly where you aimed
  * Rotation-proof     finger up/down is judged by distance from the wrist, not
                       "tip above knuckle", so a tilted hand doesn't misread
  * Mode debounce      scroll/zoom/pause must be seen a few frames in a row
                       before they engage — no more flickering between modes
  * Threaded camera    frame grabbing runs on its own thread; the main loop
                       always sees the freshest frame (kills buffer lag)
  * Downscaled inference  MediaPipe runs on a 640px copy of the frame — big CPU
                       drop, no accuracy loss at typical hand distances
  * Quartz fast mouse  on Mac, cursor moves go straight through CoreGraphics
                       instead of pyautogui (which throttles every call)
  * 3D pinch distance  pinch detection uses x, y AND z, more reliable when the
                       hand is angled toward the camera

All thresholds are in the TUNABLES block below.
"""
import time
import math
import threading
import numpy as np
import cv2
import mediapipe as mp
import pyautogui

# ---------------- TUNABLES ----------------
CAM_INDEX       = 0      # change to 1 if the wrong camera opens
CAM_W, CAM_H    = 1280, 720
PROC_WIDTH      = 640    # frame width MediaPipe actually processes (smaller = faster)
ACTIVE_MIN      = 0.15   # active region of the frame mapped to the screen edges
ACTIVE_MAX      = 0.85   # smaller window = less hand travel to cross the screen

# One Euro filter — the two knobs that matter:
EURO_MIN_CUTOFF = 1.0    # lower = smoother when still (but floatier). try 0.5-2.0
EURO_BETA       = 0.004  # higher = snappier during fast moves. try 0.001-0.01

PINCH_ON        = 0.40   # thumb-tip distance (relative to palm) that counts as a pinch
PINCH_OFF       = 0.55   # must open past this to release (hysteresis, stops flicker)
CLICK_MAX_S     = 0.35   # pinch shorter than this = click; longer = drag
CLICK_FREEZE_S  = 0.20   # cursor stays frozen this long at pinch start (no click drift)

FINGER_UP_RATIO = 1.10   # tip must be this much farther from wrist than its PIP joint
MODE_STABLE_N   = 3      # frames a mode must persist before it engages (debounce)

SCROLL_GAIN     = 350    # scroll sensitivity (ticks per normalized hand-height moved)
SCROLL_DEADZONE = 0.004  # hand jitter below this (normalized) doesn't scroll
ZOOM_STEP_PX    = 0.04   # how far the hand moves (normalized) before one zoom step fires
ZOOM_COOLDOWN   = 0.30   # min seconds between zoom steps
ZOOM_IN_KEY     = "="    # Cmd+=  (zoom in in most Mac apps)
ZOOM_OUT_KEY    = "-"    # Cmd+-
# ------------------------------------------

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()


# ---- fast cursor moves: Quartz on Mac, pyautogui everywhere else ----
try:
    from Quartz import (CGEventCreateMouseEvent, CGEventPost,
                        kCGEventMouseMoved, kCGEventLeftMouseDragged,
                        kCGHIDEventTap, kCGMouseButtonLeft)

    def move_cursor(x, y, dragging=False):
        etype = kCGEventLeftMouseDragged if dragging else kCGEventMouseMoved
        CGEventPost(kCGHIDEventTap,
                    CGEventCreateMouseEvent(None, etype, (x, y), kCGMouseButtonLeft))
    FAST_MOUSE = True
except ImportError:                                   # non-Mac fallback
    def move_cursor(x, y, dragging=False):
        pyautogui.moveTo(x, y)
    FAST_MOUSE = False


class OneEuroFilter:
    """Adaptive low-pass filter: heavy smoothing at low speed, light at high speed.
    (Casiez et al. 2012 — the standard for pointing devices.)"""

    def __init__(self, min_cutoff=EURO_MIN_CUTOFF, beta=EURO_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

    def reset(self):
        self.x_prev, self.dx_prev, self.t_prev = None, 0.0, None


class Camera:
    """Grabs frames on a background thread so the main loop never waits on the
    camera and never processes a stale buffered frame."""

    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ok, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.stopped = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def release(self):
        self.stopped = True
        time.sleep(0.05)
        self.cap.release()


mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,       # set to 0 for an extra speed boost if tracking holds up
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)


def dist(a, b):
    """3D distance between two normalized landmarks (z helps with angled hands)."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def fingers_up(lm):
    """(index, middle, ring, pinky) booleans — True when extended.
    A finger counts as up when its tip is meaningfully farther from the wrist
    than its PIP joint, which works at any hand rotation (unlike y-comparisons)."""
    wrist = lm[0]

    def up(tip, pip):
        return dist(lm[tip], wrist) > dist(lm[pip], wrist) * FINGER_UP_RATIO

    return up(8, 6), up(12, 10), up(16, 14), up(20, 18)


def hand_to_screen(lm):
    """Map the index knuckle (landmark 5) to screen coords. The knuckle barely
    moves when you curl the fingertip to pinch, so clicks don't tug the cursor."""
    tx = np.interp(lm[5].x, [ACTIVE_MIN, ACTIVE_MAX], [0, SCREEN_W])
    ty = np.interp(lm[5].y, [ACTIVE_MIN, ACTIVE_MAX], [0, SCREEN_H])
    return tx, ty


# ---- state carried between frames ----
filt_x = OneEuroFilter()
filt_y = OneEuroFilter()
pinching    = False
pinch_start = 0.0
dragging    = False
rc_down     = False      # right-click held flag (edge detection)
scroll_ref  = None       # hand y when scroll mode began
scroll_acc  = 0.0        # fractional scroll ticks carried between frames
zoom_ref    = None       # hand y when zoom mode began
last_zoom   = 0.0
mode        = "move"
cand_mode   = "move"     # debounce: candidate mode + how many frames it's held
cand_count  = 0
fps         = 0.0
last_t      = time.time()

cam = Camera(CAM_INDEX, CAM_W, CAM_H)
if not cam.ok:
    raise SystemExit(
        "Could not read from the camera. Check CAM_INDEX, and that Terminal has "
        "camera permission (System Settings > Privacy & Security > Camera)."
    )

print(f"Hand gesture cursor v3 running (fast mouse: {'Quartz' if FAST_MOUSE else 'pyautogui'}). "
      "Press Q in the camera window to quit.")

while True:
    ok, frame = cam.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)              # mirror so movement matches your hand
    h, w = frame.shape[:2]

    # MediaPipe sees a downscaled, read-only copy — much cheaper, same landmarks
    small = cv2.resize(frame, (PROC_WIDTH, int(h * PROC_WIDTH / w)))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands.process(rgb)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
    last_t = now

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        lm = hand.landmark
        mp_draw.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

        palm = dist(lm[0], lm[9]) or 1e-6   # palm length normalizes for hand distance
        ti = dist(lm[4], lm[8])  / palm     # thumb-index ratio  (small = pinch)
        tm = dist(lm[4], lm[12]) / palm     # thumb-middle ratio (small = right click)
        idx, mid, rng, pky = fingers_up(lm)

        # --- thumb + index pinch: click / drag with hysteresis ---
        if not pinching and ti < PINCH_ON:
            pinching = True
            pinch_start = now
        elif pinching and ti > PINCH_OFF:
            if dragging:
                pyautogui.mouseUp()
                dragging = False
            else:
                pyautogui.click()
            pinching = False

        # --- thumb + middle pinch: right click (fires once per pinch) ---
        if tm < PINCH_ON and not pinching:
            if not rc_down:
                pyautogui.click(button="right")
                rc_down = True
        else:
            rc_down = False

        # --- pinch takes priority over everything ---
        if pinching:
            if not dragging and (now - pinch_start) > CLICK_MAX_S:
                pyautogui.mouseDown()
                dragging = True
            mode = "drag" if dragging else "click"
            # freeze the cursor at pinch onset so the click lands where you aimed;
            # once you're dragging (or the freeze window passes) it moves again
            if dragging or (now - pinch_start) > CLICK_FREEZE_S:
                tx, ty = hand_to_screen(lm)
                cx = min(max(filt_x(tx, now), 0), SCREEN_W - 1)
                cy_ = min(max(filt_y(ty, now), 0), SCREEN_H - 1)
                move_cursor(cx, cy_, dragging)
            scroll_ref = zoom_ref = None
            cand_mode, cand_count = mode, 0

        else:
            # --- classify this frame, then debounce before switching mode ---
            if idx and mid and rng and pky:
                want = "pause"
            elif idx and mid and rng and not pky:
                want = "zoom"
            elif idx and mid and not rng and not pky:
                want = "scroll"
            else:
                want = "move"

            if want == cand_mode:
                cand_count += 1
            else:
                cand_mode, cand_count = want, 1
            if cand_count >= MODE_STABLE_N and mode != cand_mode:
                mode = cand_mode
                scroll_ref = zoom_ref = None   # fresh reference on every mode entry
                scroll_acc = 0.0

            if mode == "pause":
                pass                            # hand at rest, cursor stays put

            elif mode == "zoom":
                cy = lm[9].y
                if zoom_ref is None:
                    zoom_ref = cy
                dy = zoom_ref - cy              # positive = hand moved up
                if abs(dy) > ZOOM_STEP_PX and (now - last_zoom) > ZOOM_COOLDOWN:
                    pyautogui.hotkey("command", ZOOM_IN_KEY if dy > 0 else ZOOM_OUT_KEY)
                    last_zoom = now
                    zoom_ref = cy

            elif mode == "scroll":
                cy = lm[9].y
                if scroll_ref is None:
                    scroll_ref = cy
                dy = scroll_ref - cy            # positive = hand moved up
                if abs(dy) < SCROLL_DEADZONE:
                    dy = 0.0
                scroll_acc += dy * SCROLL_GAIN  # fractional ticks accumulate,
                ticks = int(scroll_acc)         # so slow scrolls still register
                if ticks:
                    pyautogui.scroll(ticks)
                    scroll_acc -= ticks
                scroll_ref = cy

            else:                               # move
                tx, ty = hand_to_screen(lm)
                cx = min(max(filt_x(tx, now), 0), SCREEN_W - 1)
                cy_ = min(max(filt_y(ty, now), 0), SCREEN_H - 1)
                move_cursor(cx, cy_)

    else:
        # hand left the frame -- release anything held so nothing gets stuck
        if dragging:
            pyautogui.mouseUp()
            dragging = False
        pinching = False
        rc_down = False
        scroll_ref = zoom_ref = None
        scroll_acc = 0.0
        filt_x.reset()
        filt_y.reset()
        mode = "no hand"
        cand_mode, cand_count = "move", 0

    # --- on-screen heads-up display ---
    cv2.rectangle(frame, (0, 0), (w, 42), (28, 28, 28), -1)
    cv2.putText(frame, f"MODE: {mode.upper()}    {fps:4.0f} FPS    (Q = quit)",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 170), 2)
    cv2.imshow("Hand Gesture Cursor v3", frame)

    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cam.release()
cv2.destroyAllWindows()
if dragging:
    pyautogui.mouseUp()
print("Stopped.")
