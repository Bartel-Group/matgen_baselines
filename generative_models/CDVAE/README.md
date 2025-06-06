# Instructions for CDVAE use

Here we provide scripts to interface with the Crystal Diffusion Variational AutoEncoder (CDVAE).

All code included here is adapted from the CDVAE package:

* Preprint: [Crystal Diffusion Variational Autoencoder for Periodic Material Generation](https://arxiv.org/abs/2110.06197)
* Repo: [https://github.com/txie-93/cdvae](https://github.com/txie-93/cdvae)

## Setup

### Environment Setup

Follow instructions on the original [CDVAE repository](https://github.com/txie-93/cdvae) for environment setup and installation. If the environment provided with that repository does not work on your system, consider using the ```env.yml``` included with this folder.

## Pre-trained Models

The package includes a pre-trained model for stability optimization:

* ```pre-trained/``` - Model for generating stable crystal structures

## Configuration

Edit config.json to specify your generation parameters:

```json
{
    "task": {
        "type": "generation",
        "parameters": {
            "num_samples": 256,
            "batch_size": 16
        }
    },
    "langevin_dynamics": {
        "n_step_each": 100,
        "step_lr": 1e-4,
        "min_sigma": 0.0,
        "save_traj": false
    },
    "output": {
        "format": "cif",
        "save_path": "generated_materials/"
    }
}
```

## Configuration Options

### Task Types

* ```"generation"``` - Generate novel materials by sampling from latent space
* ```"reconstruction"``` - Reconstruct materials from test set
* ```"optimization"``` - Generate materials optimized for specific properties

### Task Parameters

For generation:

* ```num_samples```: Number of materials to generate
* ```batch_size```: Batch size for generation

### Langevin Dynamics Parameters

* ```n_step_each```: Number of steps for each noise level
* ```step_lr```: Step size for Langevin dynamics
* ```min_sigma```: Minimum noise level to use
* ```save_traj```: Whether to save the entire trajectory

## Usage

### Materials Generation

To generate new materials using the pre-trained model:

```bash
python generate_cdvae.py config.json
```

This will:

* Load the pre-trained CDVAE model
* Generate materials according to your configuration
* Save structures as CIF files in the specified output directory
* Provide evaluation metrics for generated structures

### Output

After successful generation, you'll find:

* ```generated_materials/``` directory containing CIF files
* ```generation_metrics.json``` with validity and diversity metrics
* ```generated_structures.pt``` with raw structure data

