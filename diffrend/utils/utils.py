import numpy as np


def get_param_value(key, dict_var, default_val, required=False):
    if key in dict_var:
        return dict_var[key]
    elif required:
        raise ValueError('Missing required key {}'.format(key))

    return default_val


def contrast_stretch_percentile(im, nbins, hist_range, new_range=None,
                                low=0.05, high=0.95):
    hist, bins = np.histogram(im.ravel(), nbins, hist_range)
    cdf = hist.cumsum() / hist.sum()
    min_val = sum(cdf <= low)
    max_val = sum(cdf <= high)
    # print(UB, min_val, max_val)
    im = np.clip(im, min_val, max_val).astype(np.float32)
    if max_val > min_val:
        im = (im - min_val) / (max_val - min_val)

    if new_range is not None:
        im = im * (max(new_range) - min(new_range)) + min(new_range)

    return im
