import json
import os
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import argparse
import yaml

from eval_utils import load_model, get_crystals_list
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter


def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        print(config_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return None


def monkey_patch_gemnet_decoder():
    """
    Monkey patch the GemNetTDecoder to use the correct path.
    This is necessary to overwrite the hard-coded path in the
    checkpoint from our training run.
    """
    try:
        from cdvae.pl_modules.decoder import GemNetTDecoder

        # Store the original __init__ method
        original_init = GemNetTDecoder.__init__

        def patched_init(self, *args, **kwargs):
            # Override the scale_file parameter
            project_root = os.environ.get('PROJECT_ROOT', '')
            correct_path = f"{project_root}/cdvae/pl_modules/gemnet/gemnet-dT.json"

            if 'scale_file' in kwargs:
                print(f"Patching scale_file from {kwargs['scale_file']} to {correct_path}")
                kwargs['scale_file'] = correct_path

            # Call the original __init__
            return original_init(self, *args, **kwargs)

        # Replace the __init__ method
        GemNetTDecoder.__init__ = patched_init
        print("Successfully monkey-patched GemNetTDecoder")

    except Exception as e:
        print(f"Failed to monkey-patch GemNetTDecoder: {e}")


def load_cdvae_model(model_path="pre-trained/"):
    """Load the pre-trained CDVAE model with monkey-patched decoder."""
    try:
        # Convert to absolute path
        model_path = Path(model_path).resolve()

        # Apply monkey patch before loading
        monkey_patch_gemnet_decoder()

        print(f"Loading model from: {model_path}")
        model, test_loader, cfg = load_model(model_path, load_data=False)

        if torch.cuda.is_available():
            model.to('cuda')

        return model, test_loader, cfg

    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTo use CDVAE, you need to train a model first:")
        print("1. Navigate to the CDVAE repository")
        print("2. Run: python cdvae/run.py data=perov expname=my_model")
        print("3. After training, use the model path in $HYDRA_JOBS/singlerun/YYYY-MM-DD/my_model")
        return None, None, None


def generate_materials(model, config):
    """Generate materials using CDVAE."""
    task_config = config['task']
    ld_config = config.get('langevin_dynamics', {})

    # Set up Langevin dynamics parameters
    ld_kwargs = SimpleNamespace(
        n_step_each=ld_config.get('n_step_each', 100),
        step_lr=ld_config.get('step_lr', 1e-4),
        min_sigma=ld_config.get('min_sigma', 0.0),
        save_traj=ld_config.get('save_traj', False),
        disable_bar=ld_config.get('disable_bar', False)
    )

    if task_config['type'] == 'generation':
        return generation_task(model, ld_kwargs, task_config['parameters'])
    elif task_config['type'] == 'optimization':
        return optimization_task(model, ld_kwargs, task_config['parameters'])
    else:
        raise ValueError(f"Unknown task type: {task_config['type']}")

def generation_task(model, ld_kwargs, params):
    """Generate materials by sampling from latent space."""

    try:
        num_samples = params['num_samples']  # Will raise KeyError if missing
        batch_size = params['batch_size']    # Will raise KeyError if missing
    except KeyError as e:
        raise ValueError(f"Required parameter missing from config: {e}")

    all_structures = []
    samples_generated = 0

    while samples_generated < num_samples:
        # Calculate how many samples to generate in this batch
        current_batch_size = min(batch_size, num_samples - samples_generated)

        print(f"Generating batch with {current_batch_size} samples ({samples_generated + current_batch_size}/{num_samples})")

        # Sample from latent space
        z = torch.randn(current_batch_size, model.hparams.latent_dim, device=model.device)

        # Generate structures
        outputs = model.langevin_dynamics(z, ld_kwargs)

        # Convert to crystal structures
        crystal_list = get_crystals_list(
            outputs['frac_coords'],
            outputs['atom_types'],
            outputs['lengths'],
            outputs['angles'],
            outputs['num_atoms']
        )

        all_structures.extend(crystal_list)
        samples_generated += current_batch_size

    return all_structures


def optimization_task(model, ld_kwargs, params):
    """Generate materials through property optimization."""
    num_starting_points = params.get('num_starting_points', 100)
    num_gradient_steps = params.get('num_gradient_steps', 5000)
    lr = params.get('lr', 1e-3)

    # Initialize latent variables
    z = torch.randn(num_starting_points, model.hparams.latent_dim, device=model.device)
    z.requires_grad = True

    optimizer = torch.optim.Adam([z], lr=lr)
    model.freeze()

    print("Optimizing structures...")
    for step in range(num_gradient_steps):
        optimizer.zero_grad()

        # Compute property loss (assuming minimization)
        if hasattr(model, 'fc_property'):
            loss = model.fc_property(z).mean()
        else:
            # If no property predictor, optimize for stability in latent space
            loss = torch.norm(z, dim=1).mean()

        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"Step {step}/{num_gradient_steps}, Loss: {loss.item():.4f}")

    # Generate final structures
    outputs = model.langevin_dynamics(z, ld_kwargs)

    crystal_list = get_crystals_list(
        outputs['frac_coords'],
        outputs['atom_types'],
        outputs['lengths'],
        outputs['angles'],
        outputs['num_atoms']
    )

    return crystal_list


def save_structures_as_cif(crystal_list, output_dir):
    """Save crystal structures as CIF files."""
    os.makedirs(output_dir, exist_ok=True)

    valid_count = 0

    for i, crystal_dict in enumerate(crystal_list):
        try:
            # Create pymatgen Structure
            structure = Structure(
                lattice=Lattice.from_parameters(
                    *(crystal_dict['lengths'].tolist() + crystal_dict['angles'].tolist())
                ),
                species=crystal_dict['atom_types'],
                coords=crystal_dict['frac_coords'],
                coords_are_cartesian=False
            )

            # Write CIF file
            cif_writer = CifWriter(structure)
            cif_file = os.path.join(output_dir, f"generated_{i:04d}.cif")
            cif_writer.write_file(cif_file)
            valid_count += 1

        except Exception as e:
            print(f"Failed to save structure {i}: {e}")
            continue

    print(f"Successfully saved {valid_count}/{len(crystal_list)} structures as CIF files")
    return valid_count


def compute_generation_metrics(crystal_list):
    """Compute basic metrics for generated structures."""
    total_structures = len(crystal_list)
    valid_structures = 0
    volumes = []
    num_atoms_list = []

    for crystal_dict in crystal_list:
        try:
            structure = Structure(
                lattice=Lattice.from_parameters(
                    *(crystal_dict['lengths'].tolist() + crystal_dict['angles'].tolist())
                ),
                species=crystal_dict['atom_types'],
                coords=crystal_dict['frac_coords'],
                coords_are_cartesian=False
            )

            valid_structures += 1
            volumes.append(structure.volume)
            num_atoms_list.append(len(structure))

        except Exception:
            continue

    metrics = {
        'total_generated': total_structures,
        'valid_structures': valid_structures,
        'validity_rate': valid_structures / total_structures if total_structures > 0 else 0,
        'avg_volume': np.mean(volumes) if volumes else 0,
        'avg_num_atoms': np.mean(num_atoms_list) if num_atoms_list else 0,
        'volume_std': np.std(volumes) if volumes else 0,
    }

    return metrics


def main(config_path='config.json'):
    """Main function to run CDVAE generation."""

    # Load configuration
    config = load_config(config_path)
    if config is None:
        return

    # Load model
    print("Loading CDVAE model...")
    model, test_loader, cfg = load_cdvae_model()
    if model is None:
        return

    # Generate materials
    print("Generating materials...")
    crystal_list = generate_materials(model, config)

    # Save output
    output_config = config.get('output', {})
    output_dir = output_config.get('save_path', 'generated_materials/')

    if output_config.get('format', 'cif').lower() == 'cif':
        valid_count = save_structures_as_cif(crystal_list, output_dir)

    # Compute and save metrics
    metrics = compute_generation_metrics(crystal_list)

    metrics_file = os.path.join(output_dir, 'generation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Generation completed!")
    print(f"Total structures generated: {metrics['total_generated']}")
    print(f"Valid structures: {metrics['valid_structures']}")
    print(f"Validity rate: {metrics['validity_rate']:.2%}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    main(config_path)
