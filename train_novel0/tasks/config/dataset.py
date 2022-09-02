
################################################################################
# classification head parameters
################################################################################
# dataset (to find parser)

labels = "kitti"
scans = "kitti"
workers = 8        # number of threads to get data
max_points = 150000  # max of any scan in dataset

class sensor:
    name = "HDL64"
    type = "spherical"  # projective
    fov_up = 3
    fov_down = -25

    class img_prop:
        width = 2048
        height = 64
    img_means = [
        12.12,
        10.88,
        0.23,
        -1.04,
        0.21,
    ]  # range,x,y,z,signal
    img_stds = [
        12.32,
        11.47,
        6.91,
        0.86,
        0.16,
    ]
    # range,x,y,z,signal
