import os
import cv2
import mediapipe as mp
import math


class HandTracker:
    def __init__(self, model_path: str | None = None):
        self.result = None

        # Prefer legacy solutions API when available (easier, no model file required)
        if hasattr(mp, "solutions"):
            self._use_solutions = True
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
            self.draw = mp.solutions.drawing_utils
        else:
            # Use the new tasks API (requires a hand-landmarker model file)
            self._use_solutions = False
            vision = mp.tasks.vision

            # locate model file if not provided
            if model_path is None:
                candidates = [
                    os.path.join(os.path.dirname(__file__), "hand_landmarker.task"),
                    os.path.join(os.path.dirname(__file__), "assets", "hand_landmarker.task"),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        model_path = c
                        break

            if model_path is None:
                raise RuntimeError(
                    "MediaPipe tasks is present but no hand-landmarker model was found. "
                    "Place 'hand_landmarker.task' in the project root or 'assets/' directory, "
                    "or install a mediapipe version that provides 'solutions'."
                )

            try:
                from mediapipe.tasks.python.core.base_options import BaseOptions

                options = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=1,
                    min_hand_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                )

                self.hands = vision.HandLandmarker.create_from_options(options)
                # drawing helpers are not provided for this API in the same way
                self.draw = None
                self._vision = vision
            except Exception as e:
                raise RuntimeError("Failed to create HandLandmarker: " + str(e))

    def find_hand(self, frame):
        if self._use_solutions:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.result = self.hands.process(rgb)
        else:
            # convert BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # try a few ways to build a mediapipe Image from numpy
            img_obj = None
            vision = self._vision
            try:
                # preferred: vision.Image.create_from_ndarray
                if hasattr(vision, "Image") and hasattr(vision.Image, "create_from_ndarray"):
                    img_obj = vision.Image.create_from_ndarray(rgb, vision.ImageFormat.SRGB)
                else:
                    # fallback to core image implementation
                    from mediapipe.tasks.python.vision.core import image as mp_image

                    if hasattr(mp_image.Image, "create_from_ndarray"):
                        img_obj = mp_image.Image.create_from_ndarray(rgb, mp_image.ImageFormat.SRGB)
                    elif hasattr(mp_image.Image, "create_from_array"):
                        img_obj = mp_image.Image.create_from_array(rgb, mp_image.ImageFormat.SRGB)
                    else:
                        raise RuntimeError("Cannot create MediaPipe Image from ndarray")
            except Exception as e:
                raise RuntimeError("Failed to build MediaPipe Image: " + str(e))

            try:
                self.result = self.hands.detect(img_obj)
            except Exception as e:
                raise RuntimeError("HandLandmarker.detect failed: " + str(e))

    def get_index_tip(self, frame, draw=True):
        if self._use_solutions:
            if self.result and self.result.multi_hand_landmarks:
                hand = self.result.multi_hand_landmarks[0]
                h, w, _ = frame.shape
                lm = hand.landmark[8]  # index finger tip
                x, y = int(lm.x * w), int(lm.y * h)

                if draw:
                    self.draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)

                return (x, y)
            return None
        else:
            # tasks API result parsing (HandLandmarkerResult)
            if not self.result:
                return None

            # The result may expose 'hand_landmarks' as a list of landmark lists
            landmarks_list = getattr(self.result, "hand_landmarks", None) or getattr(self.result, "hand_landmark", None)
            if not landmarks_list:
                # sometimes the result object stores landmarks in different attributes
                return None

            hand = landmarks_list[0]
            # landmark sequence may be under 'landmark' or 'landmarks'
            landmarks = getattr(hand, "landmark", None) or getattr(hand, "landmarks", None) or hand
            if not landmarks:
                return None

            h, w, _ = frame.shape
            lm = landmarks[8]
            x, y = int(lm.x * w), int(lm.y * h)

            if draw:
                cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)

            return (x, y)

    def distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
