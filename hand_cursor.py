"""Hand Gesture Cursor Control.

Control your Mac's cursor with your hand via the webcam.

- Move your index finger to move the cursor.
- Pinch index finger + thumb together to click.
- Pinch and hold to drag.
- Two-finger pinch (index + middle toward thumb) for right-click.
- Press "q" in the preview window to quit.

Built with MediaPipe Hands, OpenCV, and pyautogui.
"""

import math
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ----------------------------- settings -----------------------------------

CAM_INDEX = 0          # which webcam to use (0 = built-in)
FRAME_W, FRAME_H = 960, 540

SMOOTHING = 0.25       # 0..1, lower = smoother but laggier cursor
FRAME_MARGIN = 0.15    # ignore the outer 15% of the frame so you can
                       # reach screen edges without leaving the camera view

PINCH_CLICK_DIST = 0.045   # normalized distance that counts as a pinch
PINCH_RELEASE_DIST = 0.06  # hysteresis so the click doesn't flicker
DRAG_HOLD_SECONDS = 0.35   # hold a pinch this long to start dragging
RIGHT_CLICK_COOLDOWN = 0.8

pyautogui.FAILSAFE = True   # slam the mouse into a screen corner to abort
pyautogui.PAUSE = 0         # we control timing ourselves

SCREEN_W, SCREEN_H = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def norm_dist(a, b):
    """Distance between two normalized landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)


def map_to_screen(x, y):
    """Map a normalized camera coordinate to a screen coordinate,
    using only the inner region of the frame (FRAME_MARGIN)."""
    x = (x - FRAME_MARGIN) / (1 - 2 * FRAME_MARGIN)
    y = (y - FRAME_MARGIN) / (1 - 2 * FRAME_MARGIN)
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    return x * SCREEN_W, y * SCREEN_H


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise SystemExit(
            "Could not open the webcam. If you're on macOS, make sure your "
            "terminal app has Camera permission in System Settings > "
            "Privacy & Security > Camera."
        )

    smoothed = None          # smoothed cursor position
    pinching = False         # index+thumb currently pinched
    pinch_started = 0.0      # when the current pinch began
    dragging = False
    last_right_click = 0.0

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror so it feels natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            status = "no hand"

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark
                index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]

                # ---- cursor movement (index fingertip) ----
                target = np.array(map_to_screen(index_tip.x, index_tip.y))
                if smoothed is None:
                    smoothed = target
                smoothed = smoothed + SMOOTHING * (target - smoothed)
                pyautogui.moveTo(smoothed[0], smoothed[1])

                # ---- pinch detection with hysteresis ----
                d_click = norm_dist(index_tip, thumb_tip)
                d_right = norm_dist(middle_tip, thumb_tip)
                now = time.time()

                if not pinching and d_click < PINCH_CLICK_DIST:
                    pinching = True
                    pinch_started = now
                elif pinching and d_click > PINCH_RELEASE_DIST:
                    # pinch released
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    else:
                        pyautogui.click()
                    pinching = False

                if pinching and not dragging and (now - pinch_started) > DRAG_HOLD_SECONDS:
                    pyautogui.mouseDown()
                    dragging = True

                # ---- right click: middle finger + thumb pinch ----
                if (
                    not pinching
                    and d_right < PINCH_CLICK_DIST
                    and (now - last_right_click) > RIGHT_CLICK_COOLDOWN
                ):
                    pyautogui.rightClick()
                    last_right_click = now

                if dragging:
                    status = "dragging"
                elif pinching:
                    status = "pinch"
                else:
                    status = "tracking"

                mp_draw.draw_landmarks(
                    frame,
                    result.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
            else:
                smoothed = None
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                pinching = False

            # active-region guide + status text
            x0 = int(FRAME_MARGIN * frame.shape[1])
            y0 = int(FRAME_MARGIN * frame.shape[0])
            x1 = int((1 - FRAME_MARGIN) * frame.shape[1])
            y1 = int((1 - FRAME_MARGIN) * frame.shape[0])
            cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 200, 120), 1)
            cv2.putText(
                frame, status, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 120), 2,
            )

            cv2.imshow("Hand Cursor Control  (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
