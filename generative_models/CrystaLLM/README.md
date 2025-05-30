# Instructions for CrystaLLM use

Here we provide scripts for interfacing with CrystaLLM, from the paper and repository below.

* Paper: [Antunes, L.M., Butler, K.T., & Grau-Crespo, R. (2024). Crystal structure generation with autoregressive large language modeling. *Nature Communications*, 15, 10865.](https://www.nature.com/articles/s41467-024-54639-7)

* Repo: [https://github.com/lantunes/CrystaLLM](https://github.com/lantunes/CrystaLLM)

## Setup

### Environment Setup

Follow instructions on the [CrystaLLM repo](https://github.com/lantunes/CrystaLLM) to set up your environment and install the package.

### Pre-trained Models

Download the large CrystaLLM model trained on the MP-20 dataset:

```bash
python download_pretrained.py
```

This will automatically download `crystallm_v1_large.tar.gz` to the `pre-trained/` directory and extract it.

## Usage

### Basic Structure Generation

#### 1. Create Prompts

Generate prompts for desired compositions using the composition formula:

```bash
# Create prompt for composition (elements sorted by electronegativity)
python bin/make_prompt_file.py Na2Cl2 my_prompt.txt

# Create prompt with specific space group
python bin/make_prompt_file.py Na2Cl2 my_sg_prompt.txt --spacegroup P4/nmm
```

#### 2. Generate Structures

Use the sampling script to generate CIF files:

```bash
python bin/sample.py \
    out_dir=crystallm_v1_small \
    start=FILE:my_prompt.txt \
    num_samples=10 \
    top_k=10 \
    max_new_tokens=3000 \
    device=cpu \
    target=file
```

#### 3. Post-process Generated Files

Raw model output needs post-processing:

```bash
python bin/postprocess.py raw_cifs_folder processed_cifs_folder
```

### Programmatic Generation

You can also use Python to accomplish this programatically:

```python
import subprocess
import os
from pymatgen.core import Structure

# Create output directory
os.makedirs('generated_structures', exist_ok=True)

# List of compositions to generate (examples below)
compositions = ['Na2Cl2', 'Ca1Ti1O3', 'Ba1Ti1O3']

for composition in compositions:
    # Create prompt
    subprocess.run([
        "python", "bin/make_prompt_file.py", 
        composition, "temp_prompt.txt"
    ])
    
    # Generate structure
    result = subprocess.run([
        "python", "bin/sample.py",
        "out_dir=crystallm_v1_small",
        "start=FILE:temp_prompt.txt",
        "num_samples=1",
        "top_k=10",
        "max_new_tokens=3000",
        "device=cpu",
        "target=file"
    ])
    
    # Move generated file
    if os.path.exists('sample_1.cif'):
        os.rename('sample_1.cif', f'generated_structures/{composition}.cif')
```

### Generation Parameters

Key sampling parameters:

- `num_samples`: Number of structures to generate
- `top_k`: Limits vocabulary to top k most likely tokens
- `max_new_tokens`: Maximum length of generated sequence
- `temperature`: Controls randomness (default 1.0)
- `device`: Use 'cpu' or 'cuda' for GPU acceleration
- `target`: Set to 'file' to save CIF files locally

