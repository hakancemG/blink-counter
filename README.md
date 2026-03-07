# Blink Counter

A real-time blink detection application that uses your webcam and computer vision to count eye blinks and monitor eye health.

## Features

- **Blink counting** — Tracks and displays the number of eye blinks in real time
- **Eye health alarm** — Plays an audible alert if fewer than 7 blinks are detected in 60 seconds, reminding you to rest your eyes

## Requirements

- Python 3.8+
- Webcam
- Windows (for `winsound` alarm; other platforms may need alternative audio)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python blink_counter.py
```

Press **Q** to quit.

## How It Works

- **MediaPipe Face Landmarker** — Detects 478 facial landmarks in real time
- **Eye Aspect Ratio (EAR)** — Computes the vertical-to-horizontal ratio of eye landmarks; when EAR drops below a threshold, the eye is considered closed
- **Blink detection** — A blink is registered when EAR stays below the threshold for 2+ consecutive frames and then rises again
- **Eye health alarm** — Monitors blink frequency over rolling 60-second windows; triggers a beep if the count falls below 7
