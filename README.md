# Emotion-Detection
Emotion Detection using Tensorflow, Keras, PySide6, DeepFace
## Information About Application
- It can identify seven states of emotion
  - Angry
  - Disgusted
  - Fearful
  - Happy
  - Neutral
  - Sad
  - Surprised
- Dataset:
  - [Click to See](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)
- Model:
  - [Click to See](https://drive.google.com/drive/folders/18Ia010hOxyKPY8HUF_KwwMJtnNhl3lHP?usp=sharing)
- I used 2 Model for this Application:
  - MyModel (Created by me)
  - [DeepFace](https://github.com/serengil/deepface)
- Model Accuracy:
  - | Model | Accuracy | Loss |
    | :--: | :--: | :--: |
    | MyModel | 66% | 0.9666 |
- Wandb Charts:
  - [Click to See](https://wandb.ai/mehrdadnajafi/EmotionDetection_v5?workspace=user-mehrdadnajafi)
## How To Use
- First, Install requirements:
```
pip install -r requirements.txt
```
- Download the [Model](https://drive.google.com/drive/folders/18Ia010hOxyKPY8HUF_KwwMJtnNhl3lHP?usp=sharing) and run the following command:
```
python main.py --inputModelPath "[Model Path]"
python main.py --inputModelPath "model/model_v3.h5"
```
## Application Demo
![2022-05-09 00_01_40-Settings](https://user-images.githubusercontent.com/88179607/167313416-b3759d69-3307-44f1-99ca-03e1c5290371.png)
![20220509_000020](https://user-images.githubusercontent.com/88179607/167313449-065d6526-227e-4da3-a6b3-12e2ee6d41ce.gif)
![2022-05-09 00_02_36-NVIDIA GeForce Overlay](https://user-images.githubusercontent.com/88179607/167313503-412e2c00-ac30-4080-8cfe-ea6cf9bb92b9.png)
![2022-05-09 00_02_16-NVIDIA GeForce Overlay](https://user-images.githubusercontent.com/88179607/167313504-2075867d-147a-4910-8206-1eddbc79f33d.png)

