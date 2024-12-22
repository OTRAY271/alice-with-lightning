# alice-with-lightning

PyTorch Lightning + Hydra + WandB

Unofficial implementation of unsupervised [Adversarially Learned Inference](https://arxiv.org/abs/1606.00704) (ALI)

## Setup

```bash
poetry install --no-root
poetry run wandb login
```

## Usage

```bash
poetry run python src/train.py experiment={cifar10|celeba|mnist}
```

## Results

### CIFAR-10 (6475 epochs)

samples:\
![samples](https://github.com/user-attachments/assets/a56dedc5-2c10-4e57-b249-0d1f1226c696)

reconstructions:\
![reconstructions](https://github.com/user-attachments/assets/23015d08-d43c-4f8e-8e47-55ef5494d882)


### CelebA (123 epochs)

samples:\
![samples](https://github.com/user-attachments/assets/405030cd-a59c-44b6-aa31-94b4f1ad77ea)


reconstructions:\
![reconstructions](https://github.com/user-attachments/assets/6b1cb750-dde4-44a1-9fb6-bdff39bf88d6)


### MNIST (100 epochs)

samples:\
![samples](https://github.com/user-attachments/assets/b7c4cb63-3901-4435-9b76-310d06dd71dd)

reconstructions:\
![reconstructions](https://github.com/user-attachments/assets/4dad7157-e32e-4e03-8ac8-2e52712be2e5)


## Reference

Dumoulin, V., Belghazi, I., Poole, B., Mastropietro, O., Lamb, A., Arjovsky, M., & Courville, A. (2016). Adversarially learned inference. arXiv preprint arXiv:1606.00704.