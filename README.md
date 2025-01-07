# Real-Time Hand Sign Language Recognition Using CNN and MediaPipe

This project implements a real-time American Sign Language (ASL) recognition system using [OpenCV](https://docs.opencv.org/4.10.0/index.html), [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide), and [Convolutional Neural Networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network). The system captures hand gestures via webcam and identifies ASL alphabets, facilitating communication for individuals who rely on sign language.

## Introduction

American Sign Language (ASL) is a vital means of communication for the deaf and hard-of-hearing community. This project aims to bridge communication gaps by developing a system that recognizes ASL alphabets in real-time, excluding dynamic gestures for 'J' and 'Z'.

## Features

- Real-time hand gesture recognition using webcam input.
- Detection of 24 static ASL alphabet signs.
- Utilizes MediaPipe for efficient hand landmark detection.
- CNN model for accurate gesture classification.
- Keras and TensorFlow for building and training the CNN model.
- OpenCV for image and video processing.

## Data Collection

Run the data collection script to capture hand gesture images for each alphabet and store it in the 'img' folder.
```bash
python data_collection.py
```
This can be customized to not only alphabets but any symbol.

## Results

The CNN model achieves a 99% accuracy on the test set with a classification loss of 0.024.

<a href="https://github.com/sudip0789/hand-sign-language-recognition/blob/main/signRecog.mp4">
  <img src="https://github.com/sudip0789/hand-sign-language-recognition/blob/main/demo/signrecog_img.png" alt="Live Prediction Demo" width="400">
</a>



### For additional details, refer to the detailed [project page](https://sudipdas-projects.netlify.app/hand-sign-language-detection-asl-recognition-system-using-cnn-and-mediapipe/).
