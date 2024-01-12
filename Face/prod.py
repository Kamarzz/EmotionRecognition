import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import pyvirtualcam

from Face.model import CNNModel


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        resized_face = cv2.resize(face, (48, 48))
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Добавьте нормализацию здесь, если она использовалась при обучении
        ])
        tensor = transform(resized_face)
        tensor = tensor.unsqueeze(0)
        return tensor
    return None


def overlay_emoji(frame, emoji, position=(50, 50), scale=2 ):
    emoji_height, emoji_width = emoji.shape[:2]
    scaled_emoji = cv2.resize(emoji, (0, 0), fx=scale, fy=scale)
    scaled_height, scaled_width = scaled_emoji.shape[:2]

    overlay = frame[position[1]:position[1] + scaled_height, position[0]:position[0] + scaled_width].copy()
    alpha = scaled_emoji[:, :, 3] / 255.0
    colored = scaled_emoji[:, :, :3]

    # Наложение эмодзи на фон
    for c in range(0, 3):
        overlay[:, :, c] = overlay[:, :, c] * (1 - alpha) + colored[:, :, c] * alpha

    frame[position[1]:position[1] + scaled_height, position[0]:position[0] + scaled_width] = overlay


model = CNNModel()
model_path = 'model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
model.eval()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sad', 'Surprise', 'Neutral']

emojis = {
    'Happiness': cv2.imread('emoji/Happy/icons8-grinning-face-with-smiling-eyes-96.png', cv2.IMREAD_UNCHANGED),
    'Sad': cv2.imread('emoji/Sad/icons8-crying-face-96.png', cv2.IMREAD_UNCHANGED),
    'Angry': cv2.imread('emoji/Angry/icons8-face-with-symbols-on-mouth-96.png', cv2.IMREAD_UNCHANGED),
    'Disgust': cv2.imread('emoji/Disgust/icons8-nauseated-face-96.png', cv2.IMREAD_UNCHANGED),
    'Fear': cv2.imread('emoji/Fear/icons8-anxious-face-with-sweat-96.png', cv2.IMREAD_UNCHANGED),
    'Surprise': cv2.imread('emoji/Surprise/icons8-face-with-open-mouth-96.png', cv2.IMREAD_UNCHANGED),
    'Neutral': cv2.imread('emoji/Neutral/icons8-neutral-face-96.png', cv2.IMREAD_UNCHANGED),
}

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
    while True:
        ret, frame = cap.read()
        if ret:
            frame_processed = preprocess(frame)

            if frame_processed is not None:
                with torch.no_grad():
                    predictions = model(frame_processed)

                predicted_emotion = CLASS_LABELS[torch.argmax(predictions)]

                if predicted_emotion in emojis:
                    overlay_emoji(frame, emojis[predicted_emotion])

            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cam.send(frame)
            cam.sleep_until_next_frame()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
