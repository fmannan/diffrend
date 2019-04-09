from diffrend.torch.utils import tch_var_f, get_data, grad_spatial2d
from diffrend.torch.LightFieldNet import LFNetV0
from diffrend.torch.render import render_scene
from diffrend.utils.sample_generator import uniform_sample_sphere
from torch import optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def lf_renderer_v0(pos, normal, lfnet, num_samples=10):
    pos_all = pos.reshape((-1, 3))
    normal_all = normal.reshape((-1, 3))
    pixel_colors = []

    for idx in range(pos_all.shape[0]):
        dir_sample = uniform_sample_sphere(radius=1.0, num_samples=num_samples, axis=normal_all[idx], angle=np.pi / 2)
        inp = tch_var_f(np.concatenate((np.tile(pos_all[idx], (num_samples, 1)),
                                        dir_sample), axis=-1))
        Li = lfnet(inp)
        cos_theta = torch.sum(inp[:, 3:6] * tch_var_f(normal_all[idx]), dim=-1)
        rgb = torch.sum(cos_theta[:, np.newaxis] * Li, dim=0)
        pixel_colors.append(rgb)

    im = torch.cat(pixel_colors, dim=0).reshape(pos.shape)
    return im


def lf_renderer(pos, normal, lfnet, num_samples=20):
    """This is a simpler version of lf_renderer_v0 where the same direction samples are used
    for all surfels. The samples are on a uniform sphere and so this renderer also supports
    transmissive medium.

    Args:
        pos:
        normal:
        lfnet:
        num_samples:

    Returns:

    """
    pos_all = pos.reshape((-1, 3))
    normal_all = tch_var_f(normal.reshape((-1, 3)))

    spherical_samples = uniform_sample_sphere(radius=1.0, num_samples=num_samples)

    inp = tch_var_f(np.concatenate((np.tile(pos_all[:, np.newaxis, :], (1, num_samples, 1)),
                                    np.tile(spherical_samples[np.newaxis, :, :], (pos_all.shape[0], 1, 1))),
                                   axis=-1))
    Li = lfnet(inp)
    cos_theta = torch.sum(inp[:, :, 3:6] * normal_all[:, np.newaxis, :], dim=-1)
    nonzero_mask = (cos_theta > 0).float()
    pos_cos_theta = cos_theta * nonzero_mask
    im = torch.sum(pos_cos_theta[..., np.newaxis] * Li, dim=1).reshape(pos.shape)

    return im


def test_render(scene, lfnet, num_samples=200):
    res = render_scene(scene)

    pos = get_data(res['pos'])
    normal = get_data(res['normal'])

    im = lf_renderer(pos, normal, lfnet, num_samples=num_samples)
    im_ = get_data(im)
    im_ = im_ / im_.max()

    plt.figure()
    plt.imshow(im_)
    plt.show()


def optimize_lfnet(scene, lfnet, max_iter=2000, num_samples=120, lr=1e-3,
                   print_interval=10, imsave_interval=100,
                   out_dir='./tmp_lf_opt'):
    """
    Args:
        scene: scene file
        lfnet: Light Field Network

    Returns:

    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    res = render_scene(scene)

    pos = get_data(res['pos'])
    normal = get_data(res['normal'])

    opt_vars = lfnet.parameters()
    criterion = torch.nn.MSELoss(size_average=True).cuda()
    optimizer = optim.Adam(opt_vars, lr=lr)
    lr_scheduler = StepLR(optimizer, step_size=500, gamma=0.8)

    loss_per_iter = []
    target_im = res['image']
    target_im_grad = grad_spatial2d(target_im.mean(dim=-1)[..., np.newaxis])
    h1 = plt.figure()
    plt.figure(h1.number)
    plt.imshow(get_data(target_im))
    plt.title('Target')
    plt.savefig(out_dir + '/Target.png')

    for iter in range(max_iter):
        im_est = lf_renderer(pos, normal, lfnet, num_samples=num_samples)
        im_est_grad = grad_spatial2d(im_est.mean(dim=-1)[..., np.newaxis])
        optimizer.zero_grad()
        loss = criterion(im_est * 255, target_im * 255) + criterion(target_im_grad * 100, im_est_grad * 100)

        loss_ = get_data(loss)
        loss_per_iter.append(loss_)
        if iter % print_interval == 0 or iter == max_iter - 1:
            print('{}. Loss: {}'.format(iter, loss_))
        if iter % imsave_interval == 0 or iter == max_iter - 1:
            im_out_ = get_data(im_est)
            im_out_ = np.uint8(255 * im_out_ / im_out_.max())

            plt.figure(h1.number)
            plt.imshow(im_out_)
            plt.title('%d. loss= %f' % (iter, loss_))
            plt.savefig(out_dir + '/fig_%05d.png' % iter)

        loss.backward()
        lr_scheduler.step()
        optimizer.step()

    plt.figure()
    plt.plot(loss_per_iter, linewidth=2)
    plt.xlabel('Iteration', fontsize=14)
    plt.title('Loss', fontsize=12)
    plt.grid(True)
    plt.savefig(out_dir + '/loss.png')


def main():
    import argparse
    parser = argparse.ArgumentParser(usage='Light Field Renderer Demo.')
    parser.add_argument('--scene', type=str, default='../../scenes/halfbox_sphere_cube.json',
                        help='Scene file to test rendering.')
    parser.add_argument('--maxiter', type=int, default=2000, help='Max iter.')
    parser.add_argument('--outdir', type=str, default='./tmp_lf_opt',
                        help='Output directory.')
    parser.add_argument('--test-render', action='store_true', help='Test render a scene.')

    args = parser.parse_args()
    print(args)
    lfnet = LFNetV0(in_ch=6, out_ch=3, chunks=4).cuda()
    if args.test_render:
        test_render(args.scene, lfnet)
    else:
        optimize_lfnet(args.scene, lfnet, args.maxiter)


if __name__ == '__main__':
    main()




