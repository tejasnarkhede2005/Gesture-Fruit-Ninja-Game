"""
Headless diagnostic for Gesture Fruit Ninja.
Tries to use HandTracker; if unavailable, falls back to a motion-based fingertip proxy.
Prints frame index, timestamp, and detected (x,y) coordinates to stdout.
"""
import time
import sys
import cv2

try:
    from hand_tracking import HandTracker
    HAS_TRACKER = True
except Exception as e:
    HandTracker = None
    HAS_TRACKER = False
    TRACKER_ERROR = e


def motion_fallback(cap, frames=300):
    print("Using motion fallback (background subtraction).")
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    frame_idx = 0
    while frame_idx < frames:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break
        frame = cv2.flip(frame, 1)
        fg = backSub.apply(frame)
        # morphological clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=2)
        # find contours
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 500:  # threshold
                M = cv2.moments(largest)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    print(f"{frame_idx}\t{time.time():.3f}\tFALLBACK\t{cx}\t{cy}\tarea={int(area)}")
        frame_idx += 1
    print("Fallback finished")


def run_headless(frames=500):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Failed to open camera")
        return

    if HAS_TRACKER:
        try:
            tracker = HandTracker()
            print("Using HandTracker from hand_tracking.py")
        except Exception as e:
            print("HandTracker init failed:", e)
            tracker = None
    else:
        print("hand_tracking.HandTracker import failed:", TRACKER_ERROR)
        tracker = None

    frame_idx = 0
    try:
        if tracker is None:
            motion_fallback(cap, frames=frames)
        else:
            while frame_idx < frames:
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed")
                    break
                frame = cv2.flip(frame, 1)
                try:
                    tracker.find_hand(frame)
                    tip = tracker.get_index_tip(frame, draw=False)
                    if tip:
                        print(f"{frame_idx}\t{time.time():.3f}\tTRACKER\t{tip[0]}\t{tip[1]}")
                except Exception as e:
                    print("Tracker runtime error:", e)
                    # fall back
                    motion_fallback(cap, frames=frames-frame_idx)
                    break
                frame_idx += 1
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        print("Done")


if __name__ == '__main__':
    # optional arg: number of frames
    n = 500
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except:
            pass
    run_headless(frames=n)
