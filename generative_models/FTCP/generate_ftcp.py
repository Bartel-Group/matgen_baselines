import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from model import *
from utils import *
from sampling import get_info
import joblib
import json
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return None

def load_vae_models(task_type):
    """Load the trained VAE models based on task type."""
    base_path = {
        'stability': 'pre-trained/Stability',
        'band_gap': 'pre-trained/Bandgap',
        'bulk_modulus': 'pre-trained/Bulk_Modulus'
    }[task_type]

    try:
        vae = load_model(f'{base_path}/vae_model.h5', compile=False)
        encoder = load_model(f'{base_path}/encoder_model.h5', compile=False)
        decoder = load_model(f'{base_path}/decoder_model.h5', compile=False)
        return vae, encoder, decoder
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

def recreate_scaler(dataframe, task_type):
    """Recreate the scaler using the original training data based on task"""
    props = {
        'stability': ['formation_energy_per_atom'],
        'band_gap': ['formation_energy_per_atom', 'band_gap'],
        'bulk_modulus': ['formation_energy_per_atom', 'bulk_modulus']
    }[task_type]

    Y = dataframe[props].values
    scaler_y = MinMaxScaler()
    scaler_y.fit(Y)
    return scaler_y

def generate_latent_samples_stability(X_train, encoder, params):
    """Generate samples for stability task."""
    if params['random_seed'] is not None:
        np.random.seed(params['random_seed'])

    train_indices = np.random.choice(len(X_train), params['n_samples'], replace=False)
    selected_points = X_train[train_indices]
    latent_points = encoder.predict(selected_points)
    latent_samples = np.tile(latent_points, (params['n_perturb'], 1))

    gaussian_noise = np.random.normal(0, 1, latent_samples.shape)
    perturbed_samples = latent_samples + gaussian_noise * params['noise_scale']

    return perturbed_samples, train_indices, selected_points

def generate_latent_samples_bandgap(X_train, y_train, encoder, scaler_y, params, task_params):
    """Generate samples for band gap task."""
    band_gap_min = task_params['target'] - task_params['threshold']
    band_gap_max = task_params['target'] + task_params['threshold']

    scaled_ranges = scaler_y.transform([[0, band_gap_min], [0, band_gap_max]])
    scaled_min = scaled_ranges[0][1]
    scaled_max = scaled_ranges[1][1]

    print(f"Targeting band gap range: {band_gap_min:.2f} to {band_gap_max:.2f} eV")
    print(f"Scaled band gap range: {scaled_min:.3f} to {scaled_max:.3f}")

    band_gaps = y_train[:, 1]
    valid_indices = np.where((band_gaps >= scaled_min) & (band_gaps <= scaled_max))[0]

    if len(valid_indices) == 0:
        raise ValueError(f"No training points found with band gaps between "
                       f"{band_gap_min} and {band_gap_max} eV")

    print(f"Found {len(valid_indices)} valid training points")

    n_samples = min(params['n_samples'], len(valid_indices))
    train_indices = np.random.choice(valid_indices, n_samples, replace=False)

    selected_points = X_train[train_indices]
    latent_points = encoder.predict(selected_points)
    latent_samples = np.tile(latent_points, (params['n_perturb'], 1))

    gaussian_noise = np.random.normal(0, 1, latent_samples.shape)
    perturbed_samples = latent_samples + gaussian_noise * params['noise_scale']

    return perturbed_samples, train_indices, selected_points

def generate_latent_samples_bulk(X_train, y_train, encoder, params, task_params):
    """Generate samples for bulk modulus task."""
    bulk_modulus = y_train[:, 1]

    if task_params['objective'] == 'maximize':
        threshold = np.percentile(bulk_modulus, 100 - task_params['percentile'])
        valid_indices = np.where(bulk_modulus >= threshold)[0]
    else:  # minimize
        threshold = np.percentile(bulk_modulus, task_params['percentile'])
        valid_indices = np.where(bulk_modulus <= threshold)[0]

    print(f"Found {len(valid_indices)} materials in the {'top' if task_params['objective'] == 'maximize' else 'bottom'} "
          f"{task_params['percentile']}% of bulk modulus")
    print(f"Bulk modulus threshold (scaled): {threshold:.3f}")

    n_samples = min(params['n_samples'], len(valid_indices))
    train_indices = np.random.choice(valid_indices, n_samples, replace=False)

    selected_points = X_train[train_indices]
    latent_points = encoder.predict(selected_points)

    print("\nSelected materials bulk modulus values (scaled):")
    for idx, bulk_val in zip(train_indices, bulk_modulus[train_indices]):
        print(f"Index {idx}: {bulk_val:.3f}")

    latent_samples = np.tile(latent_points, (params['n_perturb'], 1))
    gaussian_noise = np.random.normal(0, 1, latent_samples.shape)
    perturbed_samples = latent_samples + gaussian_noise * params['noise_scale']

    return perturbed_samples, train_indices, selected_points

def decode_samples(perturbed_samples, decoder):
    """Decode the perturbed samples back to the original space."""
    return decoder.predict(perturbed_samples, verbose=1)

def main(config_path='config.json'):
    # Load configuration
    config = load_config(config_path)
    if config is None:
        return

    task_type = config['task']['type']

    # Load training data based on task
    X_train = np.load('pre-trained/data/X_train.npy')
    y_train = np.load('pre-trained/data/y_train.npy')

    # Load models based on task
    vae, encoder, decoder = load_vae_models(task_type)
    if vae is None:
        return

    # Load and create scaler if needed
    scaler_y = None
    if task_type in ['band_gap', 'bulk_modulus']:
        dataframe = pd.read_csv(f'pre-trained/data/dataframe_{task_type}.csv')
        scaler_y = recreate_scaler(dataframe, task_type)

    # Generate samples based on task
    if task_type == 'stability':
        perturbed_samples, train_indices, original_points = generate_latent_samples_stability(
            X_train, encoder, config['sampling_params'])
    elif task_type == 'band_gap':
        perturbed_samples, train_indices, original_points = generate_latent_samples_bandgap(
            X_train, y_train, encoder, scaler_y, config['sampling_params'], config['task']['parameters'])
    else:  # bulk_modulus
        perturbed_samples, train_indices, original_points = generate_latent_samples_bulk(
            X_train, y_train, encoder, config['sampling_params'], config['task']['parameters'])

    # Decode the samples
    generated_samples = decode_samples(perturbed_samples, decoder)

    # Process FTCP representation
    FTCP_representation = np.load('pre-trained/data/FTCP_rep.npy')
    FTCP_representation = pad(FTCP_representation, 2)
    _, scaler_X = minmax(FTCP_representation)

    # Un-normalize
    generated_samples = inv_minmax(generated_samples, scaler_X)
    generated_samples[generated_samples < 0.1] = 0

    # Generate chemical info and CIF files
    print("Generating structure files...")
    element_data = joblib.load('data/element.pkl')

    pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor = get_info(
        generated_samples,
        max_elms=5,
        max_sites=40,
        elm_str=element_data,
        to_CIF=True,
    )

    print(f"Successfully generated {len(pred_formula)} structure files")

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    main(config_path)
