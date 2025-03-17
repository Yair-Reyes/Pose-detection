import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hands y Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

class PoseDetection:
    def __init__(self):
        print("Pose Detection Ready")
        self.pose = mp_pose.Pose()

    def isChestVisible(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            return left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5
        return False

    def personAngle(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            if all(l.visibility > 0.5 for l in [left_shoulder, right_shoulder, left_hip, right_hip]):
                shoulder_width = np.sqrt((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)
                torso_height = np.sqrt((left_shoulder.x - left_hip.x) ** 2 + (left_shoulder.y - left_hip.y) ** 2)
                if torso_height == 0:
                    return None
                s2t_ratio = shoulder_width / torso_height
                if s2t_ratio > 0.5:
                    return "forward"
                else:
                    return "side"
        return None

    def detectGesture(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
                return "Waving"
            elif left_wrist.y < left_shoulder.y:
                return "Raising Left Arm"
            elif right_wrist.y < right_shoulder.y:
                return "Raising Right Arm"
            elif left_wrist.x < left_elbow.x and left_elbow.x < left_shoulder.x:
                return "Pointing Left"
            elif right_wrist.x > right_elbow.x and right_elbow.x > right_shoulder.x:
                return "Pointing Right"
        return "No Gesture"

    def detectPose(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            if left_hip.y < left_knee.y and right_hip.y < right_knee.y:
                return "Standing"
            elif left_hip.y > left_knee.y and right_hip.y > right_knee.y:
                return "Sitting"
            elif left_knee.y > left_ankle.y and right_knee.y > right_ankle.y:
                return "Lying Down"
        return "Unknown Pose"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose_detection = PoseDetection()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame.")
            break

        fg_mask = bg_subtractor.apply(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(frame_rgb)
        chest_visible = pose_detection.isChestVisible(frame)
        orientation = pose_detection.personAngle(frame)
        gesture = pose_detection.detectGesture(frame)
        pose = pose_detection.detectPose(frame)

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if chest_visible:
            cv2.putText(frame, "Chest Visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if orientation:
            cv2.putText(frame, f"Orientation: {orientation}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {gesture}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Pose: {pose}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Detección de Manos y Pose", frame)
        cv2.imshow("Background Subtraction", fg_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()