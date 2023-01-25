FROM python:3.9.12-slim

WORKDIR /app

RUN mkdir models
COPY ["deployment/app.py", "deployment/requirements.txt", "deployment/main.py", "./"]

COPY ["models/xception-classifier.tflite", "models/viz-model.tflite", "./models/"]

RUN pip install -U pip
RUN pip install -r requirements.txt
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "app:app"]
