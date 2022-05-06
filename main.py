from PySide6.QtUiTools import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

import sys
import time
import argparse
import cv2

from myModel import MyModel
from deepFaceModel import DeepFaceModel

parser = argparse.ArgumentParser()
parser.add_argument("--inputModelPath", type=str, required=True,
                    help="Input Model Path")
args = parser.parse_args()

class SelectModeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        loader = QUiLoader()
        self.ui = loader.load("ui/selectModeWindow.ui")
        self.ui.show()
        
        cv2.destroyAllWindows()
        
        self.ui.select_file_btn.clicked.connect(self.openSelectFileWindow)
        self.ui.camera_btn.clicked.connect(self.openCameraWindow)
        
    def openSelectFileWindow(self):
        self.ui = SelectFileWindow()
    
    def openCameraWindow(self):
        self.ui = CameraWindow()
        
        
class SelectFileWindow(QDialog):
    def __init__(self):
        super().__init__()
        
        loader = QUiLoader()
        self.ui = loader.load("ui/selectFileWindow.ui")
        self.ui.show()
        
        self.model = MyModel(args.inputModelPath)
        self.deepFace_model = DeepFaceModel()
        
        self.ui.back_btn.clicked.connect(self.backTo_SelectModeWindow)
        self.ui.select_file_btn.clicked.connect(self.selectFile)
        self.ui.process_btn.clicked.connect(self.process)
        
        self.ui.mymodel_rb.setChecked(True)
    
    def backTo_SelectModeWindow(self):
        self.ui = SelectModeWindow()
        
    def selectFile(self):
        try:
            file_name = QFileDialog.getOpenFileName(self, "Open File", "", "Images (*.png *.jpg *.xmp, *.jpeg);; All Files (*)")
            self.img = cv2.imread(str(file_name[0]))
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (600, 400))
            img = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            img = QPixmap(img)
            self.ui.input_label.setPixmap(img)
        except:
            mb = QMessageBox()
            mb.setWindowTitle('Warning')
            mb.setIcon(QMessageBox.Warning)
            mb.setText("Can't load this file, Try again")
            mb.exec()
    
    def process(self):
        try:
            if self.ui.mymodel_rb.isChecked():
                result_image = self.model.detectEmotion_for_SelectedFile(self.img.copy())
            elif self.ui.deepface_rb.isChecked():
                result_image = self.deepFace_model.detectEmotion_for_SelectedFile(self.img.copy())
                
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            result_image = cv2.resize(result_image, (600, 400))
            result_image = QImage(result_image, result_image.shape[1], result_image.shape[0], QImage.Format_RGB888)
            result_image = QPixmap(result_image)
            self.ui.output_label.setPixmap(result_image)
        except:
            mb = QMessageBox()
            mb.setWindowTitle('Warning')
            mb.setIcon(QMessageBox.Warning)
            mb.setText("Can't process on this file, Try again")
            mb.exec()
    
class CameraWindow(QDialog):
    def __init__(self):
        super().__init__()
        
        loader = QUiLoader()
        self.ui = loader.load("ui/cameraWindow.ui")
        self.ui.show()
        
        self.ui.back_btn.clicked.connect(self.backTo_SelectModeWindow)
        
        self.model = MyModel(args.inputModelPath)
        self.deepFace_model = DeepFaceModel()
        
        self.ui.mymodel_rb.setChecked(True)
        try:
            self.video_cap = cv2.VideoCapture(0)
            self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
            self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
            
            while True:
                success, frame = self.video_cap.read()
                if not success:
                    break
                    
                start = time.time()
                
                if self.ui.mymodel_rb.isChecked():
                    result_image = self.model.detectEmotion_for_LiveCam(frame.copy())
                elif self.ui.deepface_rb.isChecked():
                    result_image = self.deepFace_model.detectEmotion_for_SelectedFile(frame.copy())
                
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                result_image = cv2.resize(result_image, (600, 400))
                result_image = QImage(result_image, result_image.shape[1], result_image.shape[0], QImage.Format_RGB888)
                result_image = QPixmap(result_image)
                self.ui.cam_label.setPixmap(result_image)
                
                end = time.time()
                totalTime = end - start
                    
                fps = 1 / totalTime
                self.ui.fps_label.setText(str(int(fps)))
                cv2.waitKey(1)
            
            self.video_cap.release()
            cv2.destroyAllWindows()
        except:
            self.video_cap.release()
            cv2.destroyAllWindows()
            mb = QMessageBox()
            mb.setWindowTitle('Warning')
            mb.setIcon(QMessageBox.Warning)
            mb.setText("Make sure camera is connected and Try again")
            mb.exec()
            self.ui = SelectModeWindow()
            
    def backTo_SelectModeWindow(self):
        self.video_cap.release()
        cv2.destroyAllWindows()
        self.ui = SelectModeWindow()

def exitApp():
    sys.exit()
    
if __name__ == '__main__':
    app = QApplication([])
    app.lastWindowClosed.connect(exitApp)
    selectWindow = SelectModeWindow()
    sys.exit(app.exec())