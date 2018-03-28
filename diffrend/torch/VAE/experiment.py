import torch
from torch.autograd import Variable
from torchvision.utils import save_image


class Experiment(object):
    def __init__(self, model, args):
        self.model = model
        self.args = args

        self.train_loader = None
        self.test_loader = None

        self.loss_function = None
        self.optimizer = None

    def add_dataset(self, train, test = None):
        self.train_loader = train
        if self.test is not None:
            self.test_loader = test

    def add_loss_optim(self, loss, optim):
        self.loss_function = loss
        self.optimizer = optim

    def check_completeness(self, training = True):
        if self.loss_function is None or self.test_loader is None:
            raise Exception("You have to add a loss function and "
                            "a test set loader via Experiment.add_loss_optim() and "
                            "Experiment.add_dataset() respectively")
        if training and (self.optimizer is None or self.train_loader is None):
            raise Exception("You have to add an optimizer and "
                            "a training set loader via Experiment.add_loss_optim() and "
                            "Experiment.add_dataset() respectively")

    def train(self, epoch):
        self.check_completeness(training=True)
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = Variable(data)
            if self.args.cuda:
                data = data.cuda()

            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.data[0]
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader),
                           loss.data[0] / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.check_completeness(training=False)
        self.model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(self.test_loader):
            if self.args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self.model(data)
            test_loss += self.loss_function(recon_batch, data, mu, logvar).data[0]
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(self.args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.data.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))