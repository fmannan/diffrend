
def get_param_value(key, dict_var, default_val, required=False):
    if key in dict_var:
        return dict_var[key]
    elif required:
        raise ValueError('Missing required key {}'.format(key))

    return default_val