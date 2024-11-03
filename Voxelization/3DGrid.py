import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import import_helper as ih
from matplotlib.patches import Patch

class PointCloudVisualizer:
    def __init__(self):
        self.metadata_map = {}
        self.rgb_map = {}
        self.load_metadata()

    def load_metadata(self):
        """Load and prepare metadata for visualization."""
        try:
            metadata = ih.load_label_metadata()
            self.metadata_map = ih.get_metadata_map(metadata)
            self.rgb_map = ih.get_rgb_map(metadata)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            raise

    def create_color_legend(self, labels):
        """Create a color legend for the given labels."""
        legend_elements = []
        for label in np.unique(labels):
            color = np.array(self.rgb_map.get(label, (128, 128, 128))) / 255.0
            legend_elements.append(Patch(facecolor=color, edgecolor='black',
                                         label=self.metadata_map.get(label, 'Unknown')))
        return legend_elements

    def plot_pointcloud(self, short_name, index):
        """Plot a 3D pointcloud with colored labels and a color legend."""
        try:
            points, labels = ih.import_pc_and_labels(short_name, index)
            
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Color points based on labels
            colors = [self.rgb_map.get(label, (128, 128, 128)) for label in labels]
            colors = np.array(colors) / 255.0  # Normalize RGB values
            
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'PointCloud Visualization: {short_name} - Index {index}')
            
            # Add color legend
            legend_elements = self.create_color_legend(labels)
            ax.legend(handles=legend_elements, title="Labels", loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting pointcloud: {e}")

    def plot_label_distribution(self, short_name, index):
        """Plot the distribution of labels for a specific pointcloud with a color legend."""
        try:
            _, labels = ih.import_pc_and_labels(short_name, index)
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_names = [self.metadata_map.get(label, 'Unknown') for label in unique_labels]
            
            fig, ax = plt.subplots(figsize=(15, 8))
            colors = [self.rgb_map.get(label, (128, 128, 128)) for label in unique_labels]
            colors = np.array(colors) / 255.0  # Normalize RGB values
            
            bars = ax.bar(label_names, counts, color=colors)
            ax.set_xticklabels(label_names, rotation=90)
            ax.set_title(f'Label Distribution: {short_name} - Index {index}')
            ax.set_ylabel('Count')
            ax.set_xlabel('Labels')
            
            # Add color legend
            legend_elements = self.create_color_legend(unique_labels)
            ax.legend(handles=legend_elements, title="Labels", loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting label distribution: {e}")

    def plot_2d_skyview(self, short_name, index):
        """Plot a 2D skyview of the pointcloud, colored by labels, with a color legend."""
        try:
            points, labels = ih.import_pc_and_labels(short_name, index)
            
            fig, ax = plt.subplots(figsize=(15, 12))
            
            colors = [self.rgb_map.get(label, (128, 128, 128)) for label in labels]
            colors = np.array(colors) / 255.0  # Normalize RGB values
            
            scatter = ax.scatter(points[:, 0], points[:, 1], c=colors, s=1)
            
            ax.set_title(f'2D Skyview: {short_name} - Index {index}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.axis('equal')
            
            # Add color legend
            legend_elements = self.create_color_legend(labels)
            ax.legend(handles=legend_elements, title="Labels", loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting 2D skyview: {e}")

def main():
    visualizer = PointCloudVisualizer()
    
    # Example usage
    short_name = "flight"  # Replace with your desired dataset
    index = 0  # Replace with your desired index
    
    #visualizer.plot_pointcloud(short_name, index)
    #visualizer.plot_label_distribution(short_name, index)
    visualizer.plot_2d_skyview(short_name, index)

if __name__ == "__main__":
    main()