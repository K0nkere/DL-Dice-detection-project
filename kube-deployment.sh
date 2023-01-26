#!/usr/bin/env bash
kind create cluster
kubectl cluster-info --context kind-kind

kind load docker-image dice-detection-model:v03

sleep 2

kubectl apply -f deployment/kube-deployment.yaml
kubectl get pod

sleep 2

kubectl apply -f deployment/kube-service.yaml
kubectl get service

sleep 2

kubectl autoscale deployment detection-model --name detection-model-hpa --cpu-percent=20 --min=1 --max=3
kubectl port-forward service/detection-model 8080:80