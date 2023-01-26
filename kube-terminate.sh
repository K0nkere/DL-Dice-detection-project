#!/usr/bin/env bash
kubectl delete -f deployment/kube-deployment.yaml
kubectl delete -f deployment/kube-service.yaml
kind delete cluster