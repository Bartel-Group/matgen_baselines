# Baselines for generative materials modeling

This package facilitates the generation of inorganic crystalline materials through random enumeration and ion exchange. It supports generating materials optimized for thermodynamic stability, targeting a specific electronic band gap, or achieving a desired bulk modulus. These metrics establish benchmarks for evaluating the performance of generative AI models in materials science.

## Installation

The package can be installed as follows:

```bash
git clone https://github.umn.edu/bartel-group/matgen_baselines.git
cd matgen_baselines
python -m pip install .
```

For ML predictions of band gap and bulk modulus, the user should also install [CGCNN](https://github.com/txie-93/cgcnn). Since this package is not available on PyPI, the following procedure can be used:

```bash
git clone https://github.com/txie-93/cgcnn.git
mv cgcnn base_cgcnn # install directory
mv cgcnn_setup.py base_cgcnn/setup.py
cd base_cgcnn
python -m pip install -e .
cd ../
```

## Usage

The package supports two primary generation methods with optional ML-based filtering:

### Generation Methods

1. **Random Enumeration (`random_enum`)**
   - Generates materials by randomly combining elements with prototype structures (from [AFLOW](https://aflowlib.org/prototype-encyclopedia/))
   - The code ensures charge balance in the resulting compositions using [pymatgen](https://pymatgen.org/)
   - Can be combined with ML filtering to target specific properties
   - Best for broad exploration of chemical space

2. **Ion Exchange (`ion_exchange`)**
   - Generates materials by performing ion substitutions on known materials
   - Can directly target specific properties during generation
   - Can optionally use ML filtering for additional verification
   - Best for targeted exploration around known stable compounds

### ML Filtering

ML filtering can be applied to either generation method to:
- Predict stability using [CHGNet](https://chgnet.lbl.gov/)
- Predict band gaps using [CGCNN](https://github.com/txie-93/cgcnn)
- Predict bulk modulus using [CGCNN](https://github.com/txie-93/cgcnn)

## Configuration

Tasks are configured in a config.json file. Here are all the supported combinations with example configurations:

### 1. Basic Random Enumeration
Generate structures without any property targeting:
```json
{
    "method": "random_enum",
    "num_strucs": 500,
    "filepath": "Randomly-Enumerated"
}
```

### 2. Random Enumeration with ML Filtering
Generate structures and filter for stability or specific properties using ML:

```json
{
    "method": "random_enum",
    "num_strucs": 500,
    "ml_filter": {
        "type": "stability",  # or "band_gap" or "bulk_modulus"
        "threshold": 0.0,     # for stability: maximum allowed energy above hull (in eV/atom)
        "target": 3,         # for properties: desired band_gap (eV) or bulk_modulus (GPa)
        "threshold": 0.5     # for properties: allowed deviation from target
    },
    "filepath": "Random-Enum-ML-Filtered/Property"
}
```

### 3. Ion Exchange with Direct Property Targeting
Generate structures using ion exchange, targeting stability or properties during generation:

```json
{
    "method": "ion_exchange",
    "num_strucs": 500,
    "filter_type": "stability",
    "threshold": 0.0,
    "filepath": "Ion-Exchanged/Property"
}
```

### 4. Ion Exchange with Additional ML Verification
Generate structures using ion exchange, then verify with ML:

```json
{
    "method": "ion_exchange",
    "num_strucs": 500,
    "filter_type": "stability",
    "threshold": 0.0,
    "ml_filter": {
        "type": "stability",
        "threshold": 0.0
    },
    "filepath": "Ion-Exchange-ML-Filtered/Stable"
}
```

## Full Example Configuration

Here's a complete example showing all possible combinations:

```json
{
    "mp_api_key": "YOUR_MP_API_KEY_HERE",
    "tasks": [
        {
            "method": "random_enum",
            "num_strucs": 500,
            "filepath": "Randomly-Enumerated"
        },
        {
            "method": "random_enum",
            "num_strucs": 500,
            "ml_filter": {
                "type": "stability",
                "threshold": 0.0
            },
            "filepath": "Random-Enum-ML-Filtered/Stable"
        },
        {
            "method": "random_enum",
            "num_strucs": 500,
            "ml_filter": {
                "type": "band_gap",
                "target": 3,
                "threshold": 0.5
            },
            "filepath": "Random-Enum-ML-Filtered/Bandgap"
        },
        {
            "method": "random_enum",
            "num_strucs": 500,
            "ml_filter": {
                "type": "bulk_modulus",
                "target": 400,
                "threshold": 200
            },
            "filepath": "Random-Enum-ML-Filtered/Bulk-Modulus"
        },
        {
            "method": "ion_exchange",
            "num_strucs": 500,
            "filter_type": "stability",
            "threshold": 0.0,
            "filepath": "Ion-Exchanged/Stable"
        },
        {
            "method": "ion_exchange",
            "num_strucs": 500,
            "filter_type": "stability",
            "threshold": 0.0,
            "ml_filter": {
                "type": "stability",
                "threshold": 0.0
            },
            "filepath": "Ion-Exchange-ML-Filtered/Stable"
        },
        {
            "method": "ion_exchange",
            "num_strucs": 500,
            "filter_type": "band_gap",
            "target": 3,
            "threshold": 0.5,
            "filepath": "Ion-Exchanged/Bandgap"
        },
        {
            "method": "ion_exchange",
            "num_strucs": 500,
            "filter_type": "bulk_modulus",
            "target": 400,
            "threshold": 200,
            "filepath": "Ion-Exchanged/Bulk-Modulus"
        }
    ]
}
```

## Running Generation

Once your config.json is set up, start generation with:

```bash
python generate.py
```

Generated structures will be saved as CIF files in the specified output directories.

## Calculating Decomposition Energies

The package includes a standalone script `calc_decomp.py` for calculating decomposition energies of materials using the Materials Project database as a reference. This script processes a CSV file containing material compositions and their computed energies, calculating the energy above hull or decomposition energy for each entry.

> **⚠️ IMPORTANT: Input energies must be:**
> - In units of eV/atom
> - From GGA/GGA+U calculations (or MLPs trained on GGA/GGA+U calculations) with Materials Project corrections already applied. Energies obtained from CHGNet are directly compatible.

### Usage

1. Add your Materials Project API key to the script:
   - Open `calc_decomp.py`
   - In the `calculate_decomp_energies()` function, set the API key associated with your [Materials Project](https://materialsproject.org/api) account

2. Prepare your input CSV file with the following format:
   ```csv
   composition,energy_per_atom
   Fe3Al,-7.4878
   AlFe2,-7.0365
   ```

3. Run the script:
   ```bash
   python calc_decomp.py input_file.csv
   ```

4. The script will:
   - Process each composition
   - Calculate decomposition energies using Materials Project data
   - Add results to a new 'decomp_energy' column
   - Save the updated data back to your input file

### Output Format

The script will update your input file with a new column:
```csv
composition,energy_per_atom,decomp_energy
Fe3Al,-7.4878,0.0
AlFe2,-7.0365,0.1
```

- Positive decomposition energies indicate the material is unstable
- Zero or negative values indicate the material is stable
- Values are in eV/atom

## How to Cite

If you use this code, please consider citing the below paper (available on [arXiv](https://arxiv.org/abs/2501.02144)):

```bibtex
@article{szymanski_2025_matgen_baselines,
    title={Establishing baselines for generative discovery of inorganic crystals},
    DOI={10.48550/arXiv.2501.02144},
    journal={arXiv},
    author={Szymanski, Nathan J. and Bartel, Christopher J.},
    year={2025}
}
```

You may also consider citing the below papers that this package relies on:

```bibtex
@article{ong_2013_pymatgen,
    title={Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis},
    author={Ong, Shyue Ping and Richards, William Davidson and Jain, Anubhav and Hautier, Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, Dan and Chevrier, Vincent L. and Persson, Kristin A. and Ceder, Gerbrand},
    journal={Computational Materials Science},
    volume={68},
    pages={314--319},
    year={2013},
    DOI={10.1016/j.commatsci.2012.10.028}
}

@article{jain_2013_materials,
    title={Commentary: The Materials Project: A materials genome approach to accelerating materials innovation},
    author={Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and Persson, Kristin A.},
    journal={APL Materials},
    volume={1},
    pages={011002},
    year={2013},
    DOI={10.1063/1.4812323}
}

@article{eckert_2024_aflow_library,
    title={The AFLOW library of crystallographic prototypes: Part 4},
    author={Eckert, Hagen and Divilov, Simon and Mehl, Michael J. and Hicks, David and Zettel, Adam C. and Esters, Marco and Campilongo, Xiomara and Curtarolo, Stefano},
    journal={Computational Materials Science},
    volume={240},
    pages={112988},
    year={2024},
    DOI={10.1016/j.commatsci.2024.112988}
}

@article{hautier_2011_substitutions,
    title={Data Mined Ionic Substitutions for the Discovery of New Compounds},
    author={Hautier, Geoffroy and Fischer, Chris and Ehrlacher, Virginie and Jain, Anubhav and Ceder, Gerbrand},
    journal={Inorganic Chemistry},
    volume={50},
    number={2},
    pages={656--663},
    year={2011},
    DOI={10.1021/ic102031h}
}

@article{deng_2023_chgnet,
    title={CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling},
    author={Deng, B. and Zhong, P. and Jun, K. and others},
    journal={Nature Machine Intelligence},
    volume={5},
    pages={1031--1041},
    year={2023},
    DOI={10.1038/s42256-023-00716-3}
}

@article{xie_2018_cgcnn,
    title={Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties},
    author={Xie, Tian and Grossman, Jeffrey C.},
    journal={Physical Review Letters},
    volume={120},
    number={145301},
    year={2018},
    DOI={10.1103/PhysRevLett.120.145301}
}
```
