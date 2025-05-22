"""
Example script demonstrating how to use the PyTorch-based brain generator.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the PyTorch brain generator
from SynthSeg.brain_generator_torch import BrainGenerator


def main():
    """
    Main function to demonstrate brain generation with PyTorch.
    """
    # Path to label maps directory
    labels_dir = "../data/training_label_maps"
    
    # Create brain generator
    generator = BrainGenerator(
        labels_dir=labels_dir,
        batchsize=1,
        n_channels=1,
        flipping=True,
        scaling_bounds=0.15,
        rotation_bounds=15,
        nonlin_std=3.0,
        bias_field_std=0.5,
        randomise_res=True
    )
    
    # Generate a synthetic brain
    image, labels = generator.generate_brain()
    
    # Display the results
    if image.ndim == 4:  # If batchsize > 1
        image = image[0]
        labels = labels[0]
    
    if image.ndim == 4:  # If n_channels > 1
        image = image[0]
    
    # Get middle slices
    slice_idx = image.shape[0] // 2
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot image
    axes[0].imshow(image[slice_idx], cmap='gray')
    axes[0].set_title('Synthetic Image')
    axes[0].axis('off')
    
    # Plot labels
    axes[1].imshow(labels[slice_idx], cmap='jet')
    axes[1].set_title('Label Map')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('synthetic_brain_torch.png')
    plt.close()
    
    print("Generated synthetic brain image and saved as 'synthetic_brain_torch.png'")


if __name__ == "__main__":
    main()