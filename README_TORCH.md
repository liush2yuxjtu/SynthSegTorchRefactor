# SynthSeg PyTorch Implementation

This repository contains a PyTorch-based implementation of the SynthSeg brain generator. The implementation is a standalone version that does not depend on the original TensorFlow/Keras code.

## Overview

The PyTorch implementation provides the same functionality as the original SynthSeg brain generator, but uses PyTorch for all operations. The main components are:

1. `BrainGenerator`: The main class for generating synthetic brain images from label maps.
2. `BrainGeneratorModel`: A PyTorch module that implements the image generation pipeline.
3. `BrainGeneratorDataset`: A PyTorch dataset for loading and processing label maps.

## Key Features

- **Pure PyTorch Implementation**: All operations are implemented using PyTorch, with no dependencies on TensorFlow or Keras.
- **Modular Design**: The implementation is organized into small, focused modules that can be easily reused or modified.
- **GPU Support**: The implementation can run on both CPU and GPU.
- **Same Functionality**: The implementation provides the same functionality as the original SynthSeg brain generator.

## Usage

Here's a simple example of how to use the PyTorch-based brain generator:

```python
from SynthSeg.brain_generator_torch import BrainGenerator

# Create brain generator
generator = BrainGenerator(
    labels_dir="path/to/label/maps",
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
```

See the `examples/generate_brain_torch.py` script for a complete example.

## Implementation Details

The PyTorch implementation includes the following modules:

- **Spatial Transformations**: Random spatial deformations, cropping, and flipping.
- **Intensity Transformations**: Gaussian mixture model sampling, bias field corruption, and intensity augmentation.
- **Resolution Simulation**: Gaussian blurring and resolution mimicking.
- **Label Processing**: Label conversion and mapping.

Each module is implemented as a PyTorch `nn.Module` subclass, making it easy to integrate with other PyTorch models.

## Requirements

- PyTorch >= 1.7.0
- NumPy
- Matplotlib (for visualization)

## Citation

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib