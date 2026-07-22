# Obsidian Hand Gesture

Control your computer's cursor with your hand, using nothing but your webcam.
Built with [MediaPipe Hands](https://developers.google.com/mediapipe), OpenCV,
and pyautogui.

The main script is **`obsidian_hand_control_3.py`** (v3). A simpler earlier
version is kept in `hand_cursor.py`.

## Gestures (v3)

| Gesture | Action |
| --- | --- |
| Index finger only | Move cursor |
| Quick thumb + index pinch | Left click (pinch twice = double-click) |
| Pinch + hold + move | Drag, open fingers to drop |
| Thumb + middle pinch | Right click |
| Index + middle up, raise/lower hand | Scroll |
| Index + middle + ring up, raise/lower hand | Zoom in/out (Cmd +/-) |
| Open palm (all 5 fingers) | Pause — freezes the cursor |
| Press `Q` in the preview window | Quit |

## What makes v3 smooth

- **One Euro filter** — adaptive smoothing: rock-steady when your hand is
  still, near-zero lag when it moves fast.
- **Knuckle anchor** — the cursor tracks your index knuckle, not the tip, so
  curling your finger to pinch doesn't drag the cursor off target.
- **Click freeze** — the cursor locks for a moment at pinch onset so clicks
  land exactly where you aimed.
- **Mode debounce** — scroll/zoom/pause must hold for a few frames before they
  engage, so modes never flicker.
- **Threaded camera + downscaled inference** — the main loop always sees the
  freshest frame, and MediaPipe runs on a 640px copy for a big CPU drop.
- **Quartz fast mouse (macOS)** — cursor moves go straight through
  CoreGraphics instead of pyautogui's throttled calls.

## Setup

Requires Python 3.9–3.12 (MediaPipe does not support 3.13 yet; v3 was tuned on
Python 3.11 with mediapipe 0.10.9).

```bash
git clone https://github.com/atlalmond/obsidian-hand-gesture.git
cd obsidian-hand-gesture
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python obsidian_hand_control_3.py
```

### macOS permissions

The first time you run it, grant two permissions to your terminal app in
System Settings → Privacy & Security:

- **Camera** — so the script can see your hand.
- **Accessibility** — so it's allowed to move the cursor and click.

## Tuning

Every threshold lives in the `TUNABLES` block at the top of
`obsidian_hand_control_3.py` — smoothing, pinch sensitivity, scroll gain,
zoom speed, and the active region of the frame.

## License

MIT
