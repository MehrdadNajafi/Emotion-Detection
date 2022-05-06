import math
import cv2
from deepface import DeepFace
from mtcnn import MTCNN
import mediapipe as mp

class DeepFaceModel:
    def __init__(self):
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
                result = DeepFace.analyze(face_pic, ["emotion"], enforce_detection=False)
                result_pred = str(max(result["emotion"], key=result["emotion"].get))

                FONT_SCALE = 2e-3  # Adjust for larger font size in all images
                THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images

                try:
                    height, width, _ = image.shape
                except:
                    height, width = image.shape

                font_scale = min(width, height) * FONT_SCALE
                thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
                
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
                cv2.putText(image, result_pred, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        
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
                    result = DeepFace.analyze(face_pic, ["emotion"], enforce_detection=False)
                    result_pred = str(max(result["emotion"], key=result["emotion"].get))
                    
                    FONT_SCALE = 2e-3  # Adjust for larger font size in all images
                    THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images

                    try:
                        height, width, _ = frame.shape
                    except:
                        height, width = frame.shape

                    font_scale = min(width, height) * FONT_SCALE
                    thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                    cv2.putText(frame, result_pred, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                    
        return frame 