# alice-with-lightning

PyTorch Lightning + Hydra + WandB

Unofficial implementation of unsupervised [Adversarially Learned Inference with Conditional Entropy](https://arxiv.org/abs/1709.01215) (ALICE)

## Setup

```bash
poetry install --no-root
poetry run wandb login
```

## Usage

```bash
# explicit ALICE
poetry run python src/train.py experiment=alice/explicit/mnist

# implicit ALICE
poetry run python src/train.py experiment=alice/implicit/mnist
```

## Results

### MNIST (200 epochs, explicit)

samples:\
![samples](https://github.com/user-attachments/assets/ce33d27e-8d5a-432d-99bc-2314eb809345)

reconstructions:\
![reconstructions](https://github.com/user-attachments/assets/e08a39fa-065f-41fa-9c47-339fb2101ab4)


### MNIST (200 epochs, implicit)

samples:\
![samples](https://github.com/user-attachments/assets/23fa89c3-5c06-49a2-bc02-a75f331accc8)

reconstructions:\
![reconstructions](https://github.com/user-attachments/assets/e22b5045-c1a2-4388-9c6d-b3d25134c826)


## Reference

Li, C., Liu, H., Chen, C., Pu, Y., Chen, L., Henao, R., & Carin, L. (2017). Alice: Towards understanding adversarial learning for joint distribution matching. Advances in neural information processing systems, 30.