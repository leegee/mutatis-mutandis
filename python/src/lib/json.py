import numpy as np

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(sanitize(v) for v in obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    return obj
