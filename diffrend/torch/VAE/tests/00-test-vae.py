from __future__ import print_function
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

from diffrend.torch.VAE.experiment import Experiment
from diffrend.torch.VAE.parameters import get_args
from diffrend.torch.VAE.tests.losses import vae_loss
from diffrend.torch.VAE.vaes import VAE_Mnist


args = get_args(description='VAE MNIST Example')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


model = VAE_Mnist()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

exp = Experiment(model, args)
exp.add_dataset(train_loader, test_loader)
exp.add_loss_optim(vae_loss, optimizer)


for epoch in range(1, args.epochs + 1):
    exp.train(epoch)
    exp.test(epoch)
    sample = Variable(torch.randn(64, 20))
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, 28, 28),
               'results/sample_' + str(epoch) + '.png')