import sys


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise Exception('Boolean expected')


class Tee(object):
    def __init__(self, filename):
        self.file_name = filename
        with open(self.file_name, "w") as f:
            pass
        self.stdout = sys.stdout

    def close(self):
        sys.stdout = self.stdout

    def write(self, data):
        with open(self.file_name, "a") as f:
            f.write(data)
        self.stdout.write(data)

    def flush(self):
        self.stdout.flush()