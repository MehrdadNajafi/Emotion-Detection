# Emotion-Detection
Emotion Detection using Tensorflow, Keras, PySide6, DeepFace
## Information About Application
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
will be upload the gif and image demo of the Application soon ...
