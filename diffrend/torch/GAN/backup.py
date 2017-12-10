def different_views(filename, out_dir, num_samples, radius, cam_dist,  width, height,
                               fovy, focal_length,batch_size):
    """
    Randomly generate N samples on a surface and render them. The samples include position and normal, the radius is set
    to a constant.
    """
    obj = load_model(filename)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    r = np.ones(num_samples) * radius

    large_scene = copy.deepcopy(SCENE_BASIC)

    large_scene['camera']['viewport'] = [0, 0, width, height]
    large_scene['camera']['fovy'] = np.deg2rad(fovy)
    large_scene['camera']['focal_length'] = focal_length
    large_scene['objects']['disk']['radius'] = tch_var_f(r)
    large_scene['objects']['disk']['material_idx'] = tch_var_l(np.zeros(num_samples, dtype=int).tolist())
    large_scene['materials']['albedo'] = tch_var_f([[0.6, 0.6, 0.6]])
    large_scene['tonemap']['gamma'] = tch_var_f([1.0])  # Linear output

    # generate camera positions on a sphere
    cam_pos = uniform_sample_sphere(radius=cam_dist, num_samples=batch_size)
    data=[]
    for idx in range(batch_size):
        v, vn = uniform_sample_mesh(obj, num_samples=num_samples)

        # normalize the vertices
        v = (v - np.mean(v, axis=0)) / (v.max() - v.min())

        large_scene['objects']['disk']['pos'] = tch_var_f(v)
        large_scene['objects']['disk']['normal'] = tch_var_f(vn)

        large_scene['camera']['eye'] = tch_var_f(cam_pos[idx])
        suffix = '_{}'.format(idx)

        # main render run
        res = render(large_scene)
        if CUDA:
            im = res['image']
        else:
            im = res['image']

        if CUDA:
            depth = res['depth']
        else:
            depth = res['depth']

        depth[depth >= large_scene['camera']['far']] = depth.min()
        im_depth = np.uint8(255. * (depth - depth.min()) / (depth.max() - depth.min()))
        data.append(im_depth)

    return np.array(data)
