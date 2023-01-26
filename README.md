This is a DnD dices detection project as a capstone for ML-bookcamp.

### Problem desription
During board games players are constantly throwing a bunch of dices. Throwing dices is a fun and gives an illusion that player is in control of events. But its need to count the numbers on the dices and its boring.

The idea is to create and deploy CNN bot that catches few thrown dices and returns sum of numbers on its upper faces fast. Normally the problem of detection is solved with double-step or single-step region based CNN approach (R-CNN) (there are few implementations across the internet) but these models requires dataset with additional position labels. This project partially uses implementation of these ideas but position detection based on the clusterization approach DBScan.

I splitted the problem into two tasks:
- create prediction for anchor points and bounding boxes of dices on the image
- classify each of anchors with bounding box as dice type or background

Full prediction pipeline is the following

### Dataset description
Original [dataset is taken from Kaggle](https://www.kaggle.com/datasets/ucffool/dice-d4-d6-d8-d10-d12-d20-images) + i append background images additionally. No extra info about dice positions or bounding boxes.

- the original dataset contains dice images on a various backgrounds that labeled with a dice type.
- all training images are 480x480
- all d4, d8, d10, and d12 validation images are 480x480
- most d6 and d20 validation images are 480x480
- for the test purposes i created photos with my own dices on the different backgrounds

### Project Run Guide
#### Requirements
- Ubuntu 22
- docker
- anaconda
- Kubernetes kind, kubectl

#### Clone this repo
`git clone https://github.com/K0nkere/dice-detection-project.git`

#### Download training+validation dataset from my repo
from project folder
`wget https://storage.yandexcloud.net/ybs-123123/dices-dataset.zip`
`unzip dice-dataset.zip`
download dice dataset
unzip dice dataset


#### Repo consist of files
- basic-EDA.ipynb - exploratory data analisys for images of dataset that covers number of samples in each class, mean image per class construction
- dice-detection-model-tuning.ipynb - notebook that covers process parametes tuning for detection and classification models 
- conda-requirements.txt - conda venv for project creation
- test-urls.txt - links to test images, that i've created with my own dices
- test.ipynb - notebook to send tests for prediction service
- kube-deployment.sh - script for deployment with kind and kubectl
- kude-terminate.sh - script for terminating Kubernetes deployment

- models/models-converter.py - convert .h5 models into .tflite
- models/viz-model.tflite - my pretrained detection model
- models/xception-classifier.tflite - my pretrained classification model

- deployment/app.py - flask app script
- deployment/main.py - the core of prediction pipeline
- deployment/requirements.txt - environment for the docker image
- deployment/Dockerfile - for Flask app

### models training script




create conda venv
train models with script (or use pretrained)
convert models with script (or use pretrained)

test Flask app locally
build docker image for Flask app
run image
test Flask app from image



Create conda virtual environmet based on conda-requirements.txt
<>

Download my pretrained .h5 models (optionally, if you want to reproduce all scripts)
<>
<>

### Models convertion 
from project folder run under conda virtual env

`python ./models/model-converter.py` (optionally, i included needed .tflite models in repo)

### Running Flask app locally
from project folder run under conda virtual env

`python ./deployment/app.py`

### Building Flask docker image
from project folder

```docker build -t dice-detection-model:v03 -f deployment/Dockerfile .```
```docker run -it --rm -p 9696:9696 dice-detection-model:v03```

Following instructions are for the image tag 'v03' so if you have used another one commands have to be adjusted with valid tag

Fast Kubernetes deployment
run in terminal from project folder
`bash kube-deployment.sh`

Test with test.ipynb + dont forget that row with `url = "http://localhost:8080/predict"` should be uncommented

Close port-forwarding with Ctrl+C

End of test run
```bash kube-terminate.sh```

### Examples of predictions
![alt text](https://github.com/K0nkere/dice-detection-project/blob/main/pics/predictions_1.png?raw=true)

![alt text](https://github.com/K0nkere/dice-detection-project/blob/cb164f53d5fadbc0b5b7187244d0d4458096b9e3/pics/predictions_2.png)


