import os

from torchvision import datasets
from torchvision.transforms import ToTensor

BASE_DIR = os.path.dirname(__file__)

training_data = datasets.FashionMNIST(
    # Where to store the dataset
    root=BASE_DIR,
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root=BASE_DIR,
    train=False,
    download=True,
    transform=ToTensor()
)
