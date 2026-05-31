import cv2
import time
import math
import numpy as np
import mediapipe as mp
import simpleaudio as sa

from threading import Thread

# Optional alarm sound (works on many systems)
#try:
#    from playsound import playsoun
#    HAVE_PLAYSOUND = True
#except Exception:
#    HAVE_PLAYSOUND = False


# ----------------------------
# EAR helpers
# ----------------------------
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(eye_pts):
    """
    eye_pts: list of 6 (x, y) points in this order:
      p1 (left corner), p2 (upper-left), p3 (upper-right),
      p4 (right corner), p5 (lower-right), p6 (lower-left)
    EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    """
    A = euclidean(eye_pts[1], eye_pts[5])
    B = euclidean(eye_pts[2], eye_pts[4])
    C = euclidean(eye_pts[0], eye_pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


# ----------------------------
# Alarm thread (non-blocking)
# ----------------------------
alarm_on = True

def alarm_sound(path="./alarm.wav"):
    global alarm_on
 #   if not HAVE_PLAYSOUND:
 #       # Fallback "beep" (may or may not beep depending on terminal)
 #       while alarm_on:
 #           print("\a", end="", flush=True)
 #           time.sleep(1.0)
 #       return
    wave = sa.WaveObject.from_wave_file(path)

    while alarm_on:
        play = wave.play()
        play.wait_done()
        time.sleep(0.05)

def main():
    global alarm_on

    EAR_THRESHOLD = 0.22       # typical range ~0.18-0.25
    DROWSY_SECONDS = 1.5       # eyes closed this long => alarm
    MIN_FACE_CONF = 0.5
    MIN_TRACK_CONF = 0.5

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # more accurate eyes/iris
        min_detection_confidence=MIN_FACE_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    )

    # Indices for EAR using MediaPipe FaceMesh landmarks.
    # We will use 6 points per eye (corners + upper/lower points).
    # These are commonly used stable points around the eyelids.
    # Left eye (person's left): use landmarks around left eye region
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Try changing camera index (0/1/2).")

    eyes_closed_start = None

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        ear_value = None

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
                    eyes_closed_start = time.time()
                closed_time = time.time() - eyes_closed_start

                cv2.putText(frame, f"EYES CLOSED: {closed_time:.2f}s",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if closed_time >= DROWSY_SECONDS and not alarm_on:
                    alarm_on = True
                    Thread(target=alarm_sound, daemon=True).start()

            else:
                eyes_closed_start = None
                alarm_on = False

            cv2.putText(frame, f"EAR: {ear_value:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        else:
            eyes_closed_start = None
            alarm_on = False
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, "Press 'q' to quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Drowsiness (EAR)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    alarm_on = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
