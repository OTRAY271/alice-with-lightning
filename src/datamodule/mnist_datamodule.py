import lightning as L
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
from torchvision import datasets, transforms


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 0,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()

        self.dataloader_config = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True,
        )

    def setup(self, stage: str) -> None:
        mnist_train_val = datasets.MNIST(
            self.data_dir, download=True, transform=self.transform, train=True
        )

        train_indices, val_indices = train_test_split(
            range(len(mnist_train_val)),
            train_size=55000,
            stratify=mnist_train_val.targets,
            random_state=torch.random.initial_seed(),
        )
        self.mnist_train = data.Subset(mnist_train_val, train_indices)
        self.mnist_val = data.Subset(mnist_train_val, val_indices)

        self.mnist_test = datasets.MNIST(
            self.data_dir, download=True, transform=self.transform, train=False
        )

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.mnist_train, shuffle=True, **self.dataloader_config)

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.mnist_val, **self.dataloader_config)

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.mnist_test, **self.dataloader_config)
