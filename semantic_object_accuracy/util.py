import pickle


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def load_file(path):
    with open(path, "rb") as f:
        _file = pickle.load(f)
    return _file


def get_label(path):
    idx = path.find("label_")
    try:
        label = int(path[idx + 6 : idx + 8])
    except Exception:
        label = int(path[idx + 6 : idx + 7])
    return label
