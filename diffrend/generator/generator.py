"""Generator."""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from parameters import Parameters
from datasets import Dataset_load
from networks import create_networks

# Get parameters
opt = Parameters().parse()

# Load dataset
dataloader = Dataset_load(opt).get_dataloader()

# Create the networks
netG, netD = create_networks(opt)

# Create the criterion
criterion = nn.BCELoss()

# Create the tensors
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, int(opt.nz), 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, int(opt.nz), 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

# Move everything to the GPU
if not opt.no_cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# Setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if not opt.no_cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, int(opt.nz), 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())  # Do not backpropagate through generator
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # Fake labels are real for generator cost
        labelv = Variable(label.fill_(real_label))
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        if i % 10 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f'
                  ' D(G(z)): %.4f / %.4f' % (epoch, opt.niter, i,
                                             len(dataloader),
                                             errD.data[0], errG.data[0], D_x,
                                             D_G_z1, D_G_z2), end="\r")
        if i % 100 == 0:
            vutils.save_image(real_cpu, '%s/real_samples.png' % opt.outf,
                              normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(
                fake.data,
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                normalize=True)

    # Do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    print ("Epoch ", epoch, "finished")
