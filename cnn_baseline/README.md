# Attribute Classification

## Description
Train face attribute classifier used to evalute face attribute translation GAN models.

The classifier is a ResNet34 model.

We adopt **Free Adversarial Learning** to improve the model's ability of generalization and robustness to noise.

## Train

Train CelebA attribute classifier:
```bash
CUDA_VISIBLE_DEVICES=0, 1 python main.py --config configs/cebela_classification_train.yaml
```


## Test


## Results


### CelebA

| Accuracy | CelebA |
|:---------------------:|:--------:|
| Avg. | 94.24 |
| Bangs | 96.30 |
| Black Hair | 88.69 |
| Blond Hair | 94.49 |
| Brown Hair | 85.39 |
| Gray Hair | 98.90 |
| Bushy Eyebrows | 91.39 |
| Eyeglasses | 100.0 |
| Male | 98.60 |
| Mouth Slightly Open | 93.79 |
| Mustache | 98.00 |
| Pale Skin | 95.20 |
| Young | 90.29 |
