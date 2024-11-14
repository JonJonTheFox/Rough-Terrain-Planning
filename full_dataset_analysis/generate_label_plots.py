#!/usr/bin/env python3
"""
Label Distribution Analysis Tool

This script analyzes and visualizes label distributions across multiple datasets.
It generates various plots to help understand the distribution of labels within
and across datasets.

Author: [Your Name]
Date: 2024-10-08
"""

import os
import logging
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import matplotlib.pyplot as plt
import import_helper as ih

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LabelAnalyzer:
    """Class to handle label analysis across datasets."""
    
    def __init__(self, output_dir: str = "label_plots_output.local"):
        """
        Initialize the LabelAnalyzer.
        
        Args:
            output_dir: Directory to save output plots
        """
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(file_dir, output_dir)
        self.datasets_label_counts: Dict = {}
        self.metadata_map: Dict = {}
        self.rgb_map: Dict = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_metadata(self):
        """Load and prepare metadata for visualization."""
        try:
            metadata = ih.load_label_metadata()
            self.metadata_map = ih.get_metadata_map(metadata)
            self.rgb_map = ih.get_rgb_map(metadata)
            logger.info("Metadata loaded successfully")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def aggregate_label_counts_for_dataset(self, short_name: str) -> Dict[int, int]:
        """
        Aggregate label counts for a specific dataset.
        
        Args:
            short_name: Short name identifier for the dataset
            
        Returns:
            Dictionary of label counts for the dataset
        """
        max_pcs = ih.get_max_pointclouds_count(short_name)
        label_counts = {}
        
        for index in range(max_pcs):
            try:
                _, labels = ih.import_pc_and_labels(short_name, index)
                unique_labels, counts = np.unique(labels, return_counts=True)
                
                for label, count in zip(unique_labels, counts):
                    label_counts[label] = label_counts.get(label, 0) + count
            except Exception as e:
                logger.error(f"Error processing point cloud {index} from dataset {short_name}: {e}")
                
        return label_counts
    
    def aggregate_labels_across_datasets(self):
        """Aggregate label counts across all available datasets."""
        for short_name in ih.DATASET_MAP.keys():
            logger.info(f"Processing dataset: {short_name}")
            try:
                label_counts = self.aggregate_label_counts_for_dataset(short_name)
                self.datasets_label_counts[short_name] = label_counts
            except Exception as e:
                logger.error(f"Error processing dataset {short_name}: {e}")
    
    def plot_individual_distributions(self):
        """Generate individual distribution plots for each dataset."""
        for short_name, label_counts in self.datasets_label_counts.items():
            plt.figure(figsize=(15, 8))
            try:
                self._plot_single_distribution(label_counts, short_name)
                plt.savefig(os.path.join(self.output_dir, f'{short_name}_distribution.png'))
                plt.close()
            except Exception as e:
                logger.error(f"Error plotting distribution for {short_name}: {e}")
                plt.close()
    
    def plot_comparative_distribution(self):
        """Generate a comparative distribution plot across all datasets."""
        try:
            self._create_comparative_plot()
            plt.savefig(os.path.join(self.output_dir, 'comparative_distribution.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error creating comparative plot: {e}")
            plt.close()
    
    def plot_stacked_distribution(self):
        """Generate a stacked distribution plot."""
        try:
            self._create_stacked_plot()
            plt.savefig(os.path.join(self.output_dir, 'stacked_distribution.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error creating stacked plot: {e}")
            plt.close()
    
    def plot_heatmap(self):
        """Generate a heatmap visualization."""
        try:
            self._create_heatmap()
            plt.savefig(os.path.join(self.output_dir, 'label_heatmap.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            plt.close()
    
    def _plot_single_distribution(self, label_counts: Dict[int, int], short_name: str):
        """Helper method to plot distribution for a single dataset."""
        sorted_label_counts = dict(sorted(label_counts.items()))
        labels, counts = zip(*sorted_label_counts.items())
        label_names = [self.metadata_map.get(label, 'Unknown') for label in labels]
        
        colors = [(r/255, g/255, b/255) for label in labels 
                 for r, g, b in [self.rgb_map.get(label, (128, 128, 128))]]
        
        plt.bar(label_names, counts, color=colors)
        plt.xticks(rotation=90)
        plt.title(f'Label Distribution for Dataset: {short_name}')
        plt.ylabel('Count')
        plt.xlabel('Labels')
        plt.tight_layout()
    
    def _create_comparative_plot(self):
        """Helper method to create comparative distribution plot."""
        all_labels = set()
        for label_counts in self.datasets_label_counts.values():
            all_labels.update(label_counts.keys())
        
        labels = sorted([label for label in all_labels if label in self.metadata_map])
        label_names = [self.metadata_map[label] for label in labels]
        
        plt.figure(figsize=(20, 10))
        datasets = sorted(self.datasets_label_counts.keys())
        num_datasets = len(datasets)
        bar_width = 0.8 / num_datasets
        
        for idx, label in enumerate(labels):
            for i, dataset in enumerate(datasets):
                count = self.datasets_label_counts[dataset].get(label, 0)
                offset = (i - num_datasets / 2) * bar_width + bar_width / 2
                position = idx + offset
                plt.bar(position, count, width=bar_width, label=dataset if idx == 0 else "")
        
        plt.xticks(range(len(labels)), label_names, rotation=90)
        plt.xlabel('Labels')
        plt.ylabel('Counts')
        plt.title('Label Distribution Across All Datasets')
        plt.legend()
        plt.tight_layout()

    def _create_stacked_plot(self):
        """Helper method to create stacked distribution plot."""
        all_labels = sorted(set(
            label for label in self.metadata_map 
            if any(label in counts for counts in self.datasets_label_counts.values())
        ))
        label_names = [self.metadata_map[label] for label in all_labels]

        datasets = sorted(self.datasets_label_counts.keys())
        counts_per_label = {
            label: [self.datasets_label_counts[dataset].get(label, 0) 
                   for dataset in datasets] 
            for label in all_labels
        }

        plt.figure(figsize=(20, 10))
        bottom_counts = np.zeros(len(all_labels))

        for idx, dataset in enumerate(datasets):
            counts = [counts_per_label[label][idx] for label in all_labels]
            plt.bar(label_names, counts, bottom=bottom_counts, label=dataset)
            bottom_counts += counts

        plt.xticks(rotation=90, fontsize=10)
        plt.xlabel('Labels', fontsize=12)
        plt.ylabel('Counts', fontsize=12)
        plt.title('Stacked Label Distribution Across All Datasets', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

    def _create_heatmap(self):
        """Helper method to create heatmap visualization."""
        all_labels = sorted(set(
            label for label in self.metadata_map 
            if any(label in counts for counts in self.datasets_label_counts.values())
        ))
        label_names = [self.metadata_map[label] for label in all_labels]

        datasets = sorted(self.datasets_label_counts.keys())
        data_matrix = np.array([
            [self.datasets_label_counts[dataset].get(label, 0) for label in all_labels]
            for dataset in datasets
        ])

        plt.figure(figsize=(20, 10))
        im = plt.imshow(data_matrix, aspect='auto', cmap='twilight_shifted')

        plt.xticks(np.arange(len(label_names)), label_names, rotation=90, fontsize=10)
        plt.yticks(np.arange(len(datasets)), datasets, fontsize=10)

        plt.xlabel('Labels', fontsize=12)
        plt.ylabel('Datasets', fontsize=12)
        plt.title('Heatmap of Label Distribution Across All Datasets', fontsize=14)

        plt.colorbar(im, label='Counts')
        plt.tight_layout()

def main():
    """Main execution function."""
    try:
        # Initialize analyzer
        analyzer = LabelAnalyzer()
        
        # Load necessary data
        logger.info("Loading metadata...")
        analyzer.load_metadata()
        
        # Aggregate data
        logger.info("Aggregating label counts across datasets...")
        analyzer.aggregate_labels_across_datasets()
        
        # Generate plots
        logger.info("Generating individual distribution plots...")
        analyzer.plot_individual_distributions()
        
        logger.info("Generating comparative distribution plot...")
        analyzer.plot_comparative_distribution()
        
        logger.info("Generating stacked distribution plot...")
        analyzer.plot_stacked_distribution()
        
        logger.info("Generating heatmap visualization...")
        analyzer.plot_heatmap()
        
        logger.info("Analysis complete. Plots have been saved to the output directory.")
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    main()