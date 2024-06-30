import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the Keras model
model = load_model('model.keras')

# Setup video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=1)

# labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 
               11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 
               21: 'W', 22: 'X', 23: 'Y'}

IMG_SIZE = 128  # Size to which each image will be resized

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(10, 10, 200), thickness=5, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2)
            )

            # Get bounding box coordinates
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = y_max = 0
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y

            # Add padding to the bounding box
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Crop, resize and normalize the hand image
            hand_img = frame[y_min:y_max, x_min:x_max]
            hand_img_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_img_resized = cv2.resize(hand_img_gray, (IMG_SIZE, IMG_SIZE))
            hand_img_normalized = hand_img_resized / 255.0
            hand_img_reshaped = np.reshape(hand_img_normalized, (1, IMG_SIZE, IMG_SIZE, 1))

            # Make prediction
            prediction = model.predict(hand_img_reshaped)
            # Get the predicted class index
            predicted_index = np.argmax(prediction)

            # Check if the predicted index is within the expected range
            if predicted_index in labels_dict:
                predicted_character = labels_dict[predicted_index]
            else:
                predicted_character = 'Unknown'  # if not in any label class

            # Display prediction
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
