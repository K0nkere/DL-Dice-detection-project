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

### Project run Guide
#### Requirements
- Ubuntu 22
- docker
- anaconda
- Kubernetes kind, kubectl

#### 1. Clone this repo
`git clone https://github.com/K0nkere/dice-detection-project.git`

dice-detection-project will be your _project folder_
#### 2. Download training+validation dataset from my repo
from the _project folder_
```
wget https://storage.yandexcloud.net/ybs-123123/dices-dataset.zip
unzip dice-dataset.zip
```
#### 3. Create conda virtual environment
```
conda create -y -n dice-detection python=3.9

conda activate dice-detection

pip3 install -r conda-requirements.txt

Add conda env kernel to kernels list in order to use as notebook kernel
```
conda install -y -c anaconda ipykernel
python -m ipykernel install --user --name=dice-detection
```

Add select conda env kernel as a kernel for notebooks or add as it an interpreter

#### 4. Models
(optional - this step can be skipped, I included my .tflite models in the repo)

Download original .h5 models - run the _project folder_ under conda activated env
```
wget https://storage.yandexcloud.net/ybs-123123/dice-models/xception-classifier-prepr-dr075-0.980.h5 -P models/

wget https://storage.yandexcloud.net/ybs-123123/dice-models/dice-detection-model-dr03-0.729.h5 -P models/
```

run **model-coverter.py** script

`python models/model-converter.py`

#### 5. Docker images
build docker image for Flask app - run from _project folder_
```
docker build -t dice-detection-model:v03 -f deployment/Dockerfile .

docker run -it --rm -p 9696:9696 dice-detection-model:v03
```
and test it with **test.ipynb** notebook - row with `url = "http://localhost:9696/predict"` has to be uncommented


#### 6. Kubernetes deployment
At this stage we have working docker image tagged with "v03". Following scrips are for the image tag 'v03' so if you have used another one commands have to be adjusted with valid tag. Simpy run from the _project folder_
```
bash kube-deployment.sh
```
Test with test.ipynb + dont forget that row with `url = "http://localhost:8080/predict"` should be uncommented

#### 7. End of test run
Close port-forwarding with Ctrl+C and run to delete Kubernetes cluster
```
bash kube-terminate.sh
```

### Repo consist of files
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

### Examples of predictions
![alt text](https://github.com/K0nkere/dice-detection-project/blob/main/pics/predictions_1.png?raw=true)

![alt text](https://github.com/K0nkere/dice-detection-project/blob/cb164f53d5fadbc0b5b7187244d0d4458096b9e3/pics/predictions_2.png)


