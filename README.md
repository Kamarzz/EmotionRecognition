# Emotion Recognition Project
## ITMO University - Neurotechnology and Affective Computing

### Overview
This project is part of a laboratory work for the course on Neurotechnology and Affective Computing at ITMO University. It focuses on developing a real-time emotion recognition system using facial expression analysis. The system captures video from a webcam, processes it to detect human faces, and then classifies the detected faces into one of several emotion categories.

### Technologies Used
- **Python**: The core programming language for this project.
- **OpenCV (cv2)**: Used for real-time video capture and image processing.
- **PyTorch**: A machine learning library used for building and training the emotion recognition model.
- **PyVirtualCam**: A library for sending processed video frames to a virtual webcam, enabling real-time interaction.
- **Haar Cascades**: For efficient face detection.

### Features
- Real-time emotion recognition from webcam video feed.
- Detection of multiple faces in a single frame.
- Classification of facial expressions into emotions like Happiness, Sadness, Anger, Surprise, etc.
- Display of corresponding emoji on the screen based on the detected emotion.
- Flipping of the video frame for a mirror-like effect.
