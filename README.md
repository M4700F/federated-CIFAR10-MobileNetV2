# Federated CIFAR-10 with MobileNetV2

This repository demonstrates a **basic federated learning workflow** using [Flower](https://flower.dev/) and **TensorFlow**.  
I use **MobileNetV2** as the model and **CIFAR-10** as the dataset for image classification.

## Features
- Federated learning setup with Flower
- MobileNetV2 for image classification
- CIFAR-10 dataset
- Simple client-server implementation

## Requirements
- Python 3.8+
- TensorFlow
- Flower (`flwr`)

## How to Run
1. **Start the server**:
```bash
python server.py
```
2. **Start one or more clients**:
```bash
python client.py
```
References

Flower Official Website

MobileNetV2 Paper

CIFAR-10 Dataset

## References
- [Flower Official Website](https://flower.dev/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)



