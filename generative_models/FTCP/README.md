# Instructions for FTCP use

Here we provide scripts to interface with the Fourier Transform of Crystal Properties (FTCP) representation for generative modeling.

All code is adapted from the below paper and GitHub repository:

* Paper: [Ren, Z., Tian, S. I. P., Noh, J., Oviedo, F., Xing, G., Li, J., Liang, Q., Zhu, R., Aberle, A. G., Sun, S., Wang, X., Liu, Y., Li, Q., Jayavelu, S., Hippalgaonkar, K., Jung, Y., & Buonassisi, T. (2022). An invertible crystallographic representation for general inverse design of inorganic crystals with targeted properties. *Matter*, 5(1), 314-335.](https://doi.org/10.1016/j.matt.2021.11.032)

* Repo: [https://github.com/PV-Lab/FTCP](https://github.com/PV-Lab/FTCP)

## Setup

### Environment Setup

Create the required environment using the provided environment file. We recommend mamba for this, which can better handle the older versions of Python.

```bash
mamba env create -f environment.yml
mamba activate ftcp
```

### Pre-trained Models

Our pre-trained models can be downloaded as follows:

```bash
pip install gdown
python download_pretrained.py
unzip pre-trained.zip
```

This will include models for three tasks:

* ```pre-trained/Bandgap/``` - Models for band gap optimization
* ```pre-trained/Bulk_Modulus/``` - Models for bulk modulus optimization
* ```pre-trained/Stability/``` - Models for stability optimization

### Configuration

Edit ```config.json``` to specify your generation parameters. Below are three examples.

#### For stability optimization:

```json
{
    "task": {
        "type": "stability",
        "parameters": {}
    },
    "sampling_params": {
        "n_samples": 500,
        "n_perturb": 1,
        "noise_scale": 0.2
    }
}
```

#### For band gap near 3 eV

```json
{
    "task": {
        "type": "band_gap",
        "parameters": {
            "target": 3.0,
            "threshold": 0.5
        }
    },
    "sampling_params": {
        "n_samples": 500,
        "n_perturb": 1,
        "noise_scale": 0.2
    }
}
```

#### For high bulk modulus

```json
{
    "task": {
        "type": "bulk_modulus",
        "parameters": {
            "objective": "maximize",
            "percentile": 10
        }
    },
    "sampling_params": {
        "n_samples": 500,
        "n_perturb": 1,
        "noise_scale": 0.2
    }
}
```

### Configuration Options

#### Task Types

* ```"stability"``` - Generate thermodynamically stable materials
* ```"band_gap"``` - Generate materials with specific band gaps
* ```"bulk_modulus"``` - Generate materials with optimized mechanical properties

#### Task Parameters

For ```band_gap```:

* ```target```: Target band gap in eV
* ```threshold```: Acceptable range around target (±threshold)

For ```bulk_modulus```:

* ```objective```: "maximize" or "minimize"
* ```percentile```: Top/bottom percentile to sample from (e.g., 5 for top 5%)

#### Sampling Parameters

* ```n_samples```: Number of training materials to select as starting points
* ```n_perturb```: Number of perturbations per selected material
* ```noise_scale```: Standard deviation of Gaussian noise for perturbation
* ```random_seed```: Random seed for reproducibility (set to null for random)

**Important note on the noise scale:** This parameter has a very noticable effect on the tradeoff between stability and novelty. Making ```noise_scale``` a small value (like 0.1) will cause most generated materials to closely resemble known materials. The resulting stability rate may be high, but the corresponding novelty rate will be low. In contrast, making ```noise_scale``` larger (say, 0.8) will create "newer" materials that also tend to look more unreasonable and will generally be much less stable.

### Usage

#### Materials Generation (Inference)

To generate new materials using pre-trained models:

```bash
python generate_ftcp.py config.json
```

This will:

* Load the appropriate pre-trained VAE models based on your task
* Select training materials matching your criteria
* Generate perturbed samples in latent space
* Decode samples back to FTCP representations
* Extract chemical formulas and structural information
* Generate CIF files in the designed_CIFs/ directory

#### Required Files for Inference

The below files are required for inference and currently available in ```pre-trained/data```:

* ```X_train.npy``` - Training FTCP representations
* ```y_train.npy``` - Training property labels
* ```FTCP_rep.npy``` - Full FTCP representation for scaling
* ```dataframe_[task_type].csv``` - Training dataframe for the specific task

#### Output

##### Generated Files

After successful generation, you'll find:

* ```designed_CIFs/``` directory containing CIF files for generated structures

### Model Training

#### Data Preparation

Prepare your materials database with the following columns:

* ```cif```: Crystal structure in CIF format
* ```formation_energy_per_atom```: Formation energy per atom
* ```band_gap```: Band gap (for band gap task)
* ```bulk_modulus```: Bulk modulus (for bulk modulus task)

Generate FTCP representations:

```python
from data import FTCP_represent
import pandas as pd

# Load your dataframe
df = pd.read_csv('your_materials_data.csv')

# Generate FTCP representations
ftcp_data = FTCP_represent(df, max_elms=5, max_sites=40)
np.save('FTCP_rep.npy', ftcp_data)
```

Prepare training data:

```python
from utils import pad, minmax

# Pad and normalize FTCP data
ftcp_padded = pad(ftcp_data, 2)
X_train, scaler_X = minmax(ftcp_padded)

# Prepare property labels
props = ['formation_energy_per_atom', 'band_gap']  # Adjust based on task
y_train = df[props].values

# Save training data
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
```

#### Training VAE Models

```python
from model import FTCP
from sklearn.preprocessing import MinMaxScaler

# Normalize properties
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)

# Create and compile model
VAE, encoder, decoder, regression, vae_loss = FTCP(
    X_train, y_train_scaled, 
    coeffs=(2, 10)  # KL divergence and property loss coefficients
)

VAE.compile(optimizer='adam', loss=vae_loss)

# Train the model
history = VAE.fit([X_train, y_train_scaled], X_train, 
                  epochs=100, batch_size=32, validation_split=0.2)

# Save trained models
VAE.save('vae_model.h5')
encoder.save('encoder_model.h5')
decoder.save('decoder_model.h5')
```
