"""
integrated_test.py -- Two-modality integration test.

Runs simultaneously:
  - Visual modality: webcam capture + MediaPipe Face Mesh + EAR drowsiness detection
  - Physiological modality: MAX30102 pulse sensor (BPM via doug-burrell library)

Shared alarm fires when EITHER:
  - Eyes are closed for longer than DROWSY_SECONDS, OR
  - Heart rate is outside the normal range (very low or very high)

This is a Level-2 integration test. It is NOT the full Chapter 4 pipeline:
  - No confidence-weighted fusion (no (r, c) pairs)
  - No per-driver baseline calibration
  - No PERCLOS / MAR temporal aggregation
  - No four-level risk discretisation
  - No substance modality
Those are deferred to the full pipeline build in a later session.

Press 'q' in the OpenCV window or Ctrl+C in the terminal to stop.

Prerequisites:
  - max30102 + hrcalc + heartrate_monitor (from doug-burrell/max30102) on the
    Python path (either pip-installed or in the same directory).
  - mediapipe, opencv-python, simpleaudio, numpy installed.
  - alarm.wav file in the same directory.
  - USB webcam connected.
  - MAX30102 wired to Pi over I2C (verified via 'i2cdetect -y 1' showing 0x57).
"""

import cv2
import time
import math
import threading
import numpy as np
import mediapipe as mp
import simpleaudio as sa

# Import the doug-burrell heartrate monitor (must be on Python path)
from heartrate_monitor import HeartRateMonitor

# Reports alerts to the ISERF web app (non-blocking, edge-triggered)
from reporter import AlertReporter


# ----------------------------
# EAR helpers (from your existing code)
# ----------------------------
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(eye_pts):
    """
    eye_pts: list of 6 (x, y) points:
      p1 (outer corner), p2 (upper-left), p3 (upper-right),
      p4 (inner corner), p5 (lower-right), p6 (lower-left)
    EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    """
    A = euclidean(eye_pts[1], eye_pts[5])
    B = euclidean(eye_pts[2], eye_pts[4])
    C = euclidean(eye_pts[0], eye_pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


# ----------------------------
# Alarm (non-blocking, plays in its own thread)
# ----------------------------
alarm_active = False
alarm_thread = None


def alarm_loop(stop_check, path="./alarm.wav"):
    """Play alarm sound in a loop until stop_check() returns True."""
    try:
        wave = sa.WaveObject.from_wave_file(path)
    except Exception as e:
        print("Could not load alarm.wav: {}. Using console beep instead.".format(e))
        wave = None

    while not stop_check():
        if wave is not None:
            play = wave.play()
            play.wait_done()
        else:
            print("\a", end="", flush=True)
        time.sleep(0.05)


def start_alarm():
    """Start the alarm thread if not already running."""
    global alarm_active, alarm_thread
    if alarm_thread is not None and alarm_thread.is_alive():
        return  # already running
    alarm_active = True
    alarm_thread = threading.Thread(
        target=alarm_loop,
        args=(lambda: not alarm_active,),
        daemon=True,
    )
    alarm_thread.start()


def stop_alarm():
    """Signal the alarm thread to stop."""
    global alarm_active
    alarm_active = False


# ----------------------------
# Main integration loop
# ----------------------------
def main():
    # ---- Configuration ----
    EAR_THRESHOLD = 0.22       # eye closed if EAR < this
    DROWSY_SECONDS = 1.5       # closed for this long => drowsy
    BPM_MIN_NORMAL = 50        # below this => abnormal low
    BPM_MAX_NORMAL = 120       # above this => abnormal high
    BPM_ABNORMAL_SECONDS = 5.0 # sustained out-of-range for this long => alarm
    MIN_FACE_CONF = 0.5
    MIN_TRACK_CONF = 0.5

    # ---- MediaPipe setup ----
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_FACE_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    )

    # Landmark indices for EAR (MediaPipe Face Mesh, refine_landmarks=True)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # ---- Heart rate monitor setup ----
    print("Starting MAX30102 heart rate monitor thread...")
    hrm = HeartRateMonitor(print_raw=False, print_result=False)
    hrm.start_sensor()
    time.sleep(2.0)  # give the sensor a moment to start producing samples

    # ---- Web app reporter (reads config from environment variables) ----
    reporter = AlertReporter()

    # ---- Webcam setup ----
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        hrm.stop_sensor()
        raise RuntimeError("Cannot open webcam. Try changing camera index (0/1/2).")

    # ---- State ----
    eyes_closed_start = None
    bpm_abnormal_start = None

    print("Running. Press 'q' in the window or Ctrl+C in terminal to stop.")
    print()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; stopping.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # ---- Visual modality ----
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            ear_value = None
            eyes_closed = False
            now = time.time()

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark

                def get_point(idx):
                    return (int(lm[idx].x * w), int(lm[idx].y * h))

                left_eye_pts = [get_point(i) for i in LEFT_EYE]
                right_eye_pts = [get_point(i) for i in RIGHT_EYE]

                left_ear = eye_aspect_ratio(left_eye_pts)
                right_ear = eye_aspect_ratio(right_eye_pts)
                ear_value = (left_ear + right_ear) / 2.0

                for p in left_eye_pts + right_eye_pts:
                    cv2.circle(frame, p, 2, (0, 255, 0), -1)

                if ear_value < EAR_THRESHOLD:
                    if eyes_closed_start is None:
                        eyes_closed_start = now
                    if (now - eyes_closed_start) >= DROWSY_SECONDS:
                        eyes_closed = True
                else:
                    eyes_closed_start = None
            else:
                eyes_closed_start = None  # cannot detect, reset

            # ---- Physiological modality ----
            bpm = hrm.bpm  # latest value from the monitor thread
            bpm_abnormal = False
            if bpm > 0:  # only judge when finger is detected
                if bpm < BPM_MIN_NORMAL or bpm > BPM_MAX_NORMAL:
                    if bpm_abnormal_start is None:
                        bpm_abnormal_start = now
                    if (now - bpm_abnormal_start) >= BPM_ABNORMAL_SECONDS:
                        bpm_abnormal = True
                else:
                    bpm_abnormal_start = None
            else:
                bpm_abnormal_start = None  # no finger detected, do not judge

            # ---- Combined alarm decision ----
            should_alarm = eyes_closed or bpm_abnormal

            if should_alarm:
                start_alarm()
            else:
                stop_alarm()

            # ---- Report to the web app (edge-triggered; non-blocking) ----
            reporter.update("drowsiness", active=eyes_closed, ear=ear_value)
            reporter.update("bpm_abnormal", active=bpm_abnormal, bpm=bpm)

            # ---- Overlay text ----
            if ear_value is not None:
                cv2.putText(frame, "EAR: {:.3f}".format(ear_value),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            bpm_text = "BPM: {:.0f}".format(bpm) if bpm > 0 else "BPM: -- (no finger)"
            cv2.putText(frame, bpm_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Status messages
            if eyes_closed:
                cv2.putText(frame, "DROWSY!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            elif eyes_closed_start is not None:
                t_closed = now - eyes_closed_start
                cv2.putText(frame, "Eyes closed: {:.1f}s".format(t_closed),
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 165, 255), 2)

            if bpm_abnormal:
                cv2.putText(frame, "BPM ABNORMAL!", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.putText(frame, "Press 'q' to quit", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Drowsiness + Heart Rate Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Shutting down...")
        stop_alarm()
        cap.release()
        cv2.destroyAllWindows()
        hrm.stop_sensor()
        print("Done.")


if __name__ == "__main__":
    main()
