import lightning as L
from torch.utils import data
from torchvision import datasets, transforms


class CelebADataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 0,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        )

        self.dataloader_config = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True,
        )

    def setup(self, stage: str) -> None:
        self.celeba_train = datasets.CelebA(
            self.data_dir, download=True, transform=self.transform, split="train"
        )

        self.celeba_val = datasets.CelebA(
            self.data_dir, download=True, transform=self.transform, split="valid"
        )

        self.celeba_test = datasets.CelebA(
            self.data_dir, download=True, transform=self.transform, split="test"
        )

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.celeba_train, shuffle=True, **self.dataloader_config
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.celeba_val, **self.dataloader_config)

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.celeba_test, **self.dataloader_config)
