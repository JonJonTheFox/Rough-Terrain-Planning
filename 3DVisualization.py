import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt

def load_point_cloud(file_path):
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    print(f"Loaded point cloud with shape: {point_cloud.shape}")  # Debug print
    return point_cloud

def load_label(file_path):
    label = np.fromfile(file_path, dtype=np.uint32).reshape(-1)
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16    # instance id in upper half
    print(f"Loaded labels with shape: {sem_label.shape}")  # Debug print
    return sem_label, inst_label

def get_label_colors(sem_labels):
    colors = np.zeros((sem_labels.shape[0], 3))
    unique_labels = np.unique(sem_labels)
    colormap = plt.get_cmap('hsv')
    num_colors = len(unique_labels)
    
    for i, label in enumerate(unique_labels):
        colors[sem_labels == label] = colormap(i / num_colors)[:3]
        
    return colors

def visualize_with_plotly(point_cloud, sem_labels):
    if point_cloud.shape[0] != sem_labels.shape[0]:
        print(f"Point cloud and label sizes do not match! Point cloud: {point_cloud.shape[0]}, Labels: {sem_labels.shape[0]}")
        min_size = min(point_cloud.shape[0], sem_labels.shape[0])
        point_cloud = point_cloud[:min_size, :]
        sem_labels = sem_labels[:min_size]

    label_colors = get_label_colors(sem_labels)
    trace = go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=['rgb({}, {}, {})'.format(*[int(c * 255) for c in color]) for color in label_colors],
            opacity=0.8
        )
    )

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[trace], layout=layout)
    pio.show(fig)

point_cloud_path = '/Users/YehonatanMileguir/GOOSE/goose_3d_train/lidar/train/2022-07-27_hoehenkirchner_forst/2022-07-27_hoehenkirchner_forst__0023_1658909692343408380_vls128.bin'
label_path = '/Users/YehonatanMileguir/GOOSE/goose_3d_train/labels/train/2022-07-27_hoehenkirchner_forst/2022-07-27_hoehenkirchner_forst__0023_1658909692343408380_goose.label'

# Load the point cloud data
point_cloud = load_point_cloud(point_cloud_path)

# Load the labels
sem_labels, inst_labels = load_label(label_path)

# Visualize the point cloud with semantic labels using Plotly
visualize_with_plotly(point_cloud, sem_labels)

point_cloud_path = '/Users/YehonatanMileguir/GOOSE/goose_3d_train/lidar/train/2022-07-27_hoehenkirchner_forst/2022-07-27_hoehenkirchner_forst__0002_1658909536721242062_vls128.bin'
label_path = '/Users/YehonatanMileguir/GOOSE/goose_3d_train/labels/train/2022-07-27_hoehenkirchner_forst/2022-07-27_hoehenkirchner_forst__0002_1658909536721242062_goose.label'

# Load the point cloud data
point_cloud = load_point_cloud(point_cloud_path)

# Load the labels
sem_labels, inst_labels = load_label(label_path)

# Visualize the point cloud with semantic labels using Plotly
visualize_with_plotly(point_cloud, sem_labels)
