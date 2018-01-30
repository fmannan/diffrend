"""Datset loader module."""
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms


class Dataset_load():
    """Load a dataset."""

    def __init__(self, opt):
        """Constructor."""
        self.opt = opt
        self.initialized = False

    def initialize(self):
        """Initialize."""
        if self.opt.dataset in ['imagenet', 'folder', 'lfw']:
            # folder dataset
            self.dataset = dset.ImageFolder(
                root=self.opt.dataroot,
                transform=transforms.Compose([
                    transforms.Scale(self.opt.imageSize),
                    transforms.CenterCrop(self.opt.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
        elif self.opt.dataset == 'lsun':
            self.dataset = dset.LSUN(
                db_path=self.opt.dataroot, classes=['bedroom_train'],
                transform=transforms.Compose([
                    transforms.Scale(self.opt.imageSize),
                    transforms.CenterCrop(self.opt.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
        elif self.opt.dataset == 'cifar10':
            self.dataset = dset.CIFAR10(
                root=self.opt.dataroot, download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.opt.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
        elif self.opt.dataset == 'fake':
            self.dataset = dset.FakeData(
                image_size=(3, self.opt.imageSize, self.opt.imageSize),
                transform=transforms.ToTensor())
        assert self.dataset

        # Load dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.opt.batchSize, shuffle=True,
            num_workers=int(self.opt.workers))

    def get_dataloader(self):
        """Get the dataset."""
        if not self.initialized:
            self.initialize()
        return self.dataloader
