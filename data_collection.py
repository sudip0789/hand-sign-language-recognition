import os
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

img_dir = './img'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

dataset_size = 20

for char in range(ord('A'), ord('Y') + 1):
    if chr(char) == 'J':
        continue
    else:
        dir = os.path.join(img_dir, chr(char))
        if not os.path.exists(dir):
            os.makedirs(dir)

    print('Get ready with the handsign of Character {} and then press space'.format(chr(char)))

    done = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image. Check your camera connection.")
            break
        cv2.putText(frame, 'Hit Space to start capturing ', (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (124, 61, 90), 4, cv2.LINE_AA)

        cv2.imshow('Frame', frame) # Display the frame
        
        if cv2.waitKey(25) == 32: # 32 is the ASCII value for the space character 
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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

                # Save the cropped image of the hand only
                cropped_frame = frame[y_min:y_max, x_min:x_max]
                cv2.imwrite(os.path.join(dir, '{}.jpg'.format(counter)), cropped_frame)

                counter += 1
                if counter >= dataset_size:
                    break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()
