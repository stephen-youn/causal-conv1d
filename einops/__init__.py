def rearrange(x, pattern):
    pattern = pattern.replace(' ', '')
    if pattern == 'bsd->bds':
        return x.permute(0, 2, 1)
    elif pattern == 'bds->bsd':
        return x.permute(0, 2, 1)
    else:
        raise NotImplementedError(f"pattern {pattern} not supported")
