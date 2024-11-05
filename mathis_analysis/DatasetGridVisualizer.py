import numpy as np
import matplotlib.pyplot as plt
import import_helper as ih
import os
from multiprocessing import Pool, cpu_count
from PIL import Image


class DatasetGridVisualizer:
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

    def create_output_folder(self, short_name):
        """Create a folder to save images if it doesn't exist."""
        dir_path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(dir_path, f'point_cloud_visualizations.local/{short_name}_images')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")
        return folder_path
    
    def get_distribution_file(self, short_name):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(dir_path, 'label_plots_output.label/')
        file_path = os.path.join(folder_path, f'{short_name}_distribution.png')
        if not os.path.exists(file_path):
            print(f"ERROR : Label distribution plot not found: {file_path}")
        return file_path
        
    def plot_single_pointcloud(self, args):
        """Plot and save a single point cloud as an image, but skip if the image already exists."""
        short_name, index, folder_path = args
        try:
            # Define the path where the image will be saved
            image_path = os.path.join(folder_path, f'{short_name}_pc_{index}.png')

            # Check if the image already exists
            if os.path.exists(image_path):
                #print(f"Image {image_path} already exists. Skipping...")
                return  # Skip the plotting and saving process

            # If the image doesn't exist, proceed to generate and save it
            points, labels = ih.import_pc_and_labels(short_name, index)

            # Map labels to colors
            colors = [self.rgb_map.get(label, (128, 128, 128)) for label in labels]
            colors = np.array(colors) / 255.0  # Normalize RGB values

            # Create plot for the current point cloud
            plt.figure(figsize=(5, 5))
            plt.scatter(points[:, 0], points[:, 1], c=colors, s=0.1)
            plt.title(f'Index: {index}')
            plt.axis('off')

            # Save individual plot as an image in the folder
            plt.savefig(image_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved point cloud {index} to {image_path}")
        except Exception as e:
            print(f"Error plotting point cloud {index} for {short_name}: {e}")
            
    def create_html_grid(self, short_name, folder_path, ncols=4):
        """Create an HTML file to display the images in a grid, with a legend and a toggleable label distribution plot."""
        # Get the list of all image files in the folder and sort them alphabetically
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        # Remove the first one (big picture)
        image_files = image_files[1:]
        total_images = len(image_files)

        if total_images == 0:
            print(f"No images found in {folder_path}")
            return

        # Create the HTML file
        html_path = os.path.join(folder_path, f'{short_name}_big_picture.html')
        with open(html_path, 'w') as f:
            # Write the HTML header with CSS and JavaScript for toggle behavior
            f.write('<html>\n<head>\n<title>Point Cloud Visualizations</title>\n')
            f.write('<style>\n')
            f.write('body { font-family: Arial, sans-serif; margin-left: 250px; }\n')  # Adjust margin for the fixed sidebar
            f.write('.grid-container {\n display: grid;\n grid-template-columns: ' + ' '.join(['auto'] * ncols) + ';\n grid-gap: 10px;\n}\n')
            f.write('.grid-item { text-align: center; }\n')
            f.write('.grid-item img { width: 100%; height: auto; }\n')
            f.write('.sidebar {\n position: fixed;\n top: 0;\n left: 0;\n width: 230px;\n background-color: white;\n padding: 10px;\n border-right: 1px solid #ddd;\n height: 100%;\n overflow-y: auto;\n}\n')
            f.write('.legend-item { margin-bottom: 10px; }\n')
            f.write('.legend-color { width: 20px; height: 20px; display: inline-block; vertical-align: middle; }\n')
            f.write('.legend-label { display: inline-block; margin-left: 10px; vertical-align: middle; }\n')
            f.write('.distribution { display: none; position: fixed; bottom: 10px; left: 10px; width: 950px; background-color: white; border: 1px solid #ddd; padding: 10px; cursor: pointer; }\n')  # Added cursor: pointer
            f.write('</style>\n')

            # JavaScript for toggling the distribution plot
            f.write('<script>\n')
            f.write('function toggleDistribution() {\n')
            f.write('  var dist = document.getElementById("distribution");\n')
            f.write('  if (dist.style.display === "none") {\n')
            f.write('    dist.style.display = "block";\n')
            f.write('  } else {\n')
            f.write('    dist.style.display = "none";\n')
            f.write('  }\n')
            f.write('}\n')

            # JavaScript to close the distribution plot when clicked
            f.write('function hideDistribution() {\n')
            f.write('  var dist = document.getElementById("distribution");\n')
            f.write('  dist.style.display = "none";\n')
            f.write('}\n')
            f.write('</script>\n')

            f.write('</head>\n<body>\n')

            # Add the sidebar with the button at the top
            f.write('<div class="sidebar">\n')
            f.write('<button onclick="toggleDistribution()" style="margin-bottom: 10px;">Toggle Label Distribution</button>\n')  # Button at the top

            # Add the legend title and items below the button
            f.write('<h2>Legend</h2>\n')
            for label, color in self.rgb_map.items():
                color_hex = '#{:02x}{:02x}{:02x}'.format(*color)  # Convert RGB to hex
                label_name = self.metadata_map.get(label, f'Unknown ({label})')
                f.write(f'<div class="legend-item"><span class="legend-color" style="background-color: {color_hex};"></span>')
                f.write(f'<span class="legend-label">{label_name}</span></div>\n')

            # Close the sidebar div
            f.write('</div>\n')

            # Add the grid for images
            f.write(f'<h1>{short_name} Point Cloud Visualizations</h1>\n')
            f.write('<div class="grid-container">\n')

            # Write each image in a grid item
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                f.write(f'<div class="grid-item"><img src="{img_file}" alt="{img_file}"><br>{img_file}</div>\n')

            # Close the grid container
            f.write('</div>\n')

            # Add the distribution plot, initially hidden, with an onclick event to close it
            dist_plot_path = self.get_distribution_file(short_name)
            if os.path.exists(dist_plot_path):
                f.write(f'<div id="distribution" class="distribution" onclick="hideDistribution()">\n')  # Added onclick
                f.write(f'<h3>Label Distribution</h3>\n')
                f.write(f'<img src="{dist_plot_path}" alt="Label Distribution" style="width: 100%;">\n')
                f.write('</div>\n')
            else:
                print(f"ERROR: Label distribution plot not found: {dist_plot_path}")

            # Close the HTML tags
            f.write('</body>\n</html>')

        print(f"HTML grid with legend and distribution plot saved to {html_path}")




    def plot_dataset_grid(self, short_name):
        """Plot all point clouds for a dataset as individual images using multiprocessing."""
        max_pcs = ih.get_max_pointclouds_count(short_name)

        # Create output folder
        folder_path = self.create_output_folder(short_name)

        # Prepare arguments for each process
        tasks = [(short_name, index, folder_path) for index in range(max_pcs)]

        # Use multiprocessing to speed up the process
        num_workers = min(cpu_count(), len(tasks))  # Limit number of workers to available CPUs or task count
        print(f"Using {num_workers} workers to process {max_pcs} point clouds...")
        
        with Pool(num_workers) as pool:
            pool.map(self.plot_single_pointcloud, tasks)

        # After processing, create the HTML grid of all images
        self.create_html_grid(short_name, folder_path)
    
    def create_big_picture_unused(self, short_name, folder_path, nrows=3, ncols=3):
        """Create and display a grid of all the saved point cloud images."""
        # Get the list of all image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        total_images = len(image_files)

        if total_images == 0:
            print(f"No images found in {folder_path}")
            return

        # Calculate the grid size based on the number of images
        grid_size = min(nrows * ncols, total_images)
        print(f"Creating a big picture with the first {grid_size} images...")

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
        axes = axes.flatten()  # Flatten the grid for easy iteration

        for i, ax in enumerate(axes):
            if i < total_images:
                img_path = os.path.join(folder_path, image_files[i])
                img = Image.open(img_path)
                ax.imshow(img)
                ax.axis('off')  # Hide axes
                ax.set_title(image_files[i], fontsize=8)  # Optional: Add image file name
            else:
                ax.axis('off')  # Hide axes for any empty slots

        # Adjust layout and show the grid of images
        plt.tight_layout()
        plt.show()

        # Optionally, save the big picture
        big_picture_path = os.path.join(folder_path, f'{short_name}_big_picture.png')
        fig.savefig(big_picture_path, dpi=300)
        print(f"Big picture saved to {big_picture_path}")

    def visualize_all_datasets(self):
        """Visualize all datasets in the DATASET_MAP."""
        for short_name in ih.DATASET_MAP.keys():
            print(f"Visualizing dataset: {short_name}")
            self.plot_dataset_grid(short_name)
            print(f"Completed visualization for {short_name}")

def main():
    visualizer = DatasetGridVisualizer()
    visualizer.visualize_all_datasets()

if __name__ == "__main__":
    main()
