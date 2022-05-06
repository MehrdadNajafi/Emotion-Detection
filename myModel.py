import math
import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp
from mtcnn import MTCNN

class MyModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.faceDetector_mtcnn = MTCNN()
        
        self.mp_faceDetector = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        
    def detectEmotion_for_SelectedFile(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.faceDetector_mtcnn.detect_faces(image_rgb)
        if faces:
            for face in faces:
                x, y, w, h = face["box"]
                face_pic = image_rgb[y:y+h, x:x+w]
                face_pic = cv2.cvtColor(face_pic, cv2.COLOR_RGB2GRAY)
                face_pic = cv2.resize(face_pic, (64, 64))
                face_pic = face_pic.reshape(1, 64, 64, 1)
                face_pic = face_pic / 255.0
                pred = np.argmax(self.model.predict(face_pic))
                
                if pred == 0:
                    result = "Angry"
                elif pred == 1:
                    result = "Disgusted"
                elif pred == 2:
                    result = "Fearful"
                elif pred == 3:
                    result = "Happy"
                elif pred == 4:
                    result = "Neutral"
                elif pred == 5:
                    result = "Sad"
                elif pred == 6:
                    result = "Surprised"
                
                FONT_SCALE = 2e-3  # Adjust for larger font size in all images
                THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images

                try:
                    height, width, _ = image.shape
                except:
                    height, width = image.shape

                font_scale = min(width, height) * FONT_SCALE
                thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
                
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
                cv2.putText(image, result, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        
        return image
    
    def detectEmotion_for_LiveCam(self, frame):
        with self.mp_faceDetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)
            
            if results.detections:
                for id, detection in enumerate(results.detections):
                    # mp_draw.draw_detection(image, detection)
                    bBox = detection.location_data.relative_bounding_box
                    try:
                        h, w, c = frame.shape
                    except:
                        h, w = frame.shape
                    x, y, w, h = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                    face_pic = frame[y:y+h, x:x+w]
                    face_pic = cv2.cvtColor(face_pic, cv2.COLOR_RGB2GRAY)
                    face_pic = cv2.resize(face_pic, (64, 64))
                    face_pic = face_pic.reshape(1, 64, 64, 1)
                    face_pic = face_pic / 255.0
                    pred = np.argmax(self.model.predict(face_pic))
                    
                    if pred == 0:
                        result = "Angry"
                    elif pred == 1:
                        result = "Disgusted"
                    elif pred == 2:
                        result = "Fearful"
                    elif pred == 3:
                        result = "Happy"
                    elif pred == 4:
                        result = "Neutral"
                    elif pred == 5:
                        result = "Sad"
                    elif pred == 6:
                        result = "Surprised"
                    
                    FONT_SCALE = 2e-3  # Adjust for larger font size in all images
                    THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images

                    try:
                        height, width, _ = frame.shape
                    except:
                        height, width = frame.shape

                    font_scale = min(width, height) * FONT_SCALE
                    thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                    cv2.putText(frame, result, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                    
        return frame