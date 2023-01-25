### Download training+validation dataset
from project folder
wget https://storage.yandexcloud.net/ybs-123123/dices-dataset.zip
unzip dice-dataset.zip

### Model convertion
from project folder
python ./models/model-converter.py

### Building Flask docker image
from project folder
docker build -t dice-detection-model:v03 -f deployment/Dockerfile .
docker run -it --rm -p 9696:9696 dice-detection-model:v03