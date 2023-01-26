### Download training+validation dataset
from project folder
`wget https://storage.yandexcloud.net/ybs-123123/dices-dataset.zip`
`unzip dice-dataset.zip`

### Model convertion
from project folder
`python ./models/model-converter.py`

### Running Flask app locally
from project folder
`python ./deployment/app.py`

### Building Flask docker image
from project folder
`docker build -t dice-detection-model:v03 -f deployment/Dockerfile .`
`docker run -it --rm -p 9696:9696 dice-detection-model:v03`

Following commands are for the image tag 'v03' so if you have used anoter one its have to be adjusted

Fast Kubernetes deployment
run in terminal from project folder
`bash kube-deployment.sh`
Test with test.ipynb + dont forget row with url = "http://localhost:8080/predict" shoud be uncomment

Close port-forwarding with Ctrl+C
End of test run
`bash kube-terminate.sh`