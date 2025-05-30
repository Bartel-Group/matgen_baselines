# MatterGen Materials Generation Package

Here we provide instructions on using MatterGen, from the below paper and repository:

* Paper: [Zeni, C., Pinsler, R., Zügner, D., et al. (2025). A generative model for inorganic materials design. *Nature*.](https://www.nature.com/articles/s41586-025-08628-5)

* Repo: [https://github.com/microsoft/mattergen](https://github.com/microsoft/mattergen)

## Setup

### Environment Setup

Follow the instructions on [MatterGen](https://github.com/microsoft/mattergen) and install the environment using uv (or pip):

```bash
pip install uv
uv venv .venv --python 3.10 
source .venv/bin/activate
uv pip install -e .
```

### Pre-trained Model

Our pre-trained model (on MP-20 only) for stability optimization can be downloaded as follows:

```bash
pip install gdown
python download_pretrained.py
unzip pre-trained.zip
```

## Usage

### Unconditional Generation

Generate (nominally) stable materials without property constraints, for example:

```bash
export RESULTS_PATH=results/
mattergen-generate $RESULTS_PATH --model_path=pre-trained --batch_size=128 --num_batches=4
```

### Property-Conditioned Generation

Please refer to [https://github.com/microsoft/mattergen](https://github.com/microsoft/mattergen) for models relevant to this task.

## Output

Generated materials are saved as:
- `generated_crystals_cif.zip` - Individual CIF files
- `generated_crystals.extxyz` - Single file with all structures
- `generated_trajectories.zip` - Full denoising trajectories (optional)
