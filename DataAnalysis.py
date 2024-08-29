import numpy as np

# reading a .bin file
scan = np.fromfile('/Users/YehonatanMileguir/GOOSE/goose_3d_train/lidar/train/2022-07-27_hoehenkirchner_forst/2022-07-27_hoehenkirchner_forst__0023_1658909692343408380_vls128.bin', dtype=np.float32)
scan = scan.reshape((-1, 4))

# put in attribute
points = scan[:, 0:3]    # get xyz
remissions = scan[:, 3]  # get remission

label = np.fromfile('/Users/YehonatanMileguir/GOOSE/goose_3d_train/labels/train/2022-07-27_hoehenkirchner_forst/2022-07-27_hoehenkirchner_forst__0023_1658909692343408380_goose.label', dtype=np.uint32)
label = label.reshape((-1))

# extract the semantic and instance label IDs
sem_label = label & 0xFFFF  # semantic label in lower half
inst_label = label >> 16    # instance id in upper half

