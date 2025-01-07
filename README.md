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
