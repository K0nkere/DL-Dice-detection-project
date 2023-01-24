### Download training+validation dataset
wget https://storage.yandexcloud.net/ybs-123123/dices-dataset.zip
unzip dice-dataset.zip

### Building Flask docker image
docker build -t dice-detection-model:v03 -f deployment/Dockerfile .
docker run -it --rm -p 9696:9696 dice-detection-model:v03