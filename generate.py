"""
Structure generation pipeline for materials discovery.

This module orchestrates the generation of new crystal structures using various methods
(random enumeration, ion exchange) and applies ML-based filtering for property targeting.
"""

from __future__ import annotations

# Standard library imports
import warnings
import sys
import io
import logging
import json
import os
from typing import Dict, List, Union, Optional, Tuple, Any
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass

# Third-party imports
from tqdm import tqdm
from pymatgen.core import Structure
from chgnet.model import CHGNet, StructOptimizer
from mp_api.client import MPRester

# Local imports
from matgen_baselines.analysis.random_enum import (
    StructureGenerator,
    PROTOTYPES
)
from matgen_baselines.analysis.ion_exchange import (
    SubstitutionConfig,
    generate_structures as generate_ion_exchange
)
from matgen_baselines.analysis.ml_filtration import (
    analyze_structure,
    StructureAnalyzer
)

# Configure warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="chgnet")

# Configure logging
logging.basicConfig(
    filename='generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Type aliases
TaskConfig = Dict[str, Union[str, int, float, Dict[str, Any]]]
Config = Dict[str, Union[str, List[TaskConfig]]]

# Constants
DEFAULT_MAX_ATTEMPTS: int = 500
DEFAULT_MIN_ELEMENTS: int = 3
DEFAULT_MAX_ELEMENTS: int = 5

@contextmanager
def suppress_stdout() -> None:
    """Temporarily suppress stdout."""
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = stdout

def log_to_file(message: str) -> None:
    """Write message to log file."""
    logging.info(message)

@dataclass
class GenerationConfig:
    """Configuration for structure generation."""
    method: str
    num_strucs: int
    filepath: str
    filter_type: Optional[str] = None
    target: Optional[float] = None
    threshold: Optional[float] = None
    ml_filter: Optional[Dict[str, Union[str, float]]] = None

class StructurePipeline:
    """Manages structure generation and ML filtering pipeline."""

    def __init__(self, mp_api_key: str):
        """Initialize pipeline with MP API key."""
        self.mp_api_key = mp_api_key
        self.chgnet_model: Optional[CHGNet] = None
        self.mpr: Optional[MPRester] = None
        self.optimizer: Optional[StructOptimizer] = None
        self.structure_generator = StructureGenerator()
        self.ml_tested = 0
        self.ml_passed = 0

    def initialize_ml_models(self) -> None:
        """Initialize ML models for structure analysis."""
        with suppress_stdout():
            log_to_file("Initializing CHGNet model and optimizer")
            self.chgnet_model = CHGNet.load()
            self.mpr = MPRester(self.mp_api_key)
            self.optimizer = StructOptimizer()

    def process_structure(
        self,
        structure: Structure,
        config: GenerationConfig,
        prototype_label: str,
        num_generated: int,
        pbar: tqdm
    ) -> bool:
        """Process and save a single structure."""
        if config.ml_filter:
            self.ml_tested += 1
            passes_filter, relaxed_structure = analyze_structure(
                structure,
                config.ml_filter['type'],
                (config.ml_filter.get('target'), config.ml_filter['threshold']),
                self.chgnet_model,
                self.mpr,
                optimizer=self.optimizer
            )
            if not passes_filter:
                log_to_file("Structure failed ML filter")
                pbar.set_description(
                    f"Generating structures ({self.ml_passed}/{self.ml_tested} passed ML filter)"
                )
                return False
                
            self.ml_passed += 1
            pbar.set_description(
                f"Generating structures ({self.ml_passed}/{self.ml_tested} passed ML filter)"
            )
            
            if relaxed_structure is not None:
                structure = relaxed_structure

        formula = structure.composition.reduced_formula
        suffix = f"_{prototype_label}" if prototype_label else ""
        output_path = Path(config.filepath) / f"{formula}{suffix}.cif"
        self.structure_generator.save_structure(structure, output_path)
        pbar.update(1)
        log_to_file(f"Generated structure {num_generated + 1}: {formula}")
        return True

    def generate_random_enum_structures(
        self,
        config: GenerationConfig,
        pbar: tqdm
    ) -> None:
        """Generate structures using random enumeration."""
        prototype_labels = list(PROTOTYPES.keys())
        num_generated = 0

        for prototype_label in prototype_labels:
            if num_generated >= config.num_strucs:
                break

            try:
                with suppress_stdout():
                    log_to_file(f"Trying prototype: {prototype_label}")
                    structures = self.structure_generator.retrieve_structures(
                        PROTOTYPES[prototype_label]['Compound Form'],
                        PROTOTYPES[prototype_label]['Space Group'],
                        self.mp_api_key
                    )
                if not structures:
                    log_to_file(f"No structures found for prototype {prototype_label}")
                    continue

                for _ in range(DEFAULT_MAX_ATTEMPTS):
                    if num_generated >= config.num_strucs:
                        break

                    composition = self.structure_generator.generate_random_composition(
                        DEFAULT_MIN_ELEMENTS, DEFAULT_MAX_ELEMENTS
                    )
                    structure = self.structure_generator.generate_from_prototype(
                        composition, PROTOTYPES[prototype_label], structures
                    )

                    if structure:
                        if self.process_structure(
                            structure, config, prototype_label, num_generated, pbar
                        ):
                            num_generated += 1

            except Exception as e:
                log_to_file(f"Error with prototype {prototype_label}: {str(e)}")
                continue

    def generate_ion_exchange_structures(
        self,
        config: GenerationConfig,
        pbar: tqdm
    ) -> None:
        """Generate structures using ion exchange."""
        ion_config = SubstitutionConfig(
            target_num_strucs=config.num_strucs,
            filepath=config.filepath,
            filter_type=config.filter_type or "stability",
            target_params=(config.target, config.threshold) if config.target else None,
            show_progress=True
        )
        generate_ion_exchange(ion_config, self.mp_api_key)

def load_config(config_path: str = 'config.json') -> Config:
    """Load and validate configuration from JSON file."""
    with open(config_path) as f:
        config = json.load(f)

    required_keys = {'mp_api_key', 'tasks'}
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise KeyError(f"Configuration missing required keys: {missing_keys}")

    return config

def get_task_description(config: GenerationConfig) -> str:
    """Generate descriptive string for current task."""
    if config.ml_filter:
        filter_type = config.ml_filter['type']
        if filter_type == "stability":
            return f"Begin {config.method} with ML stability filtering"
        target = config.ml_filter.get('target', '')
        return f"Begin {config.method} with ML {filter_type} filtering (target: {target})"
    
    if config.filter_type:
        return f"Begin {config.method} targeting {config.filter_type}"
    
    return f"Begin {config.method} without filtering"

def run_task(task_config: TaskConfig, mp_api_key: str) -> None:
    """Execute single structure generation task."""
    config = GenerationConfig(
        method=task_config['method'],
        num_strucs=task_config['num_strucs'],
        filepath=task_config['filepath'],
        filter_type=task_config.get('filter_type'),
        target=task_config.get('target'),
        threshold=task_config.get('threshold'),
        ml_filter=task_config.get('ml_filter')
    )
    print(get_task_description(config))
    
    pipeline = StructurePipeline(mp_api_key)
    if config.ml_filter and config.ml_filter['type'] == 'stability':
        pipeline.initialize_ml_models()

    os.makedirs(config.filepath, exist_ok=True)
    pbar = tqdm(
        total=config.num_strucs,
        desc="Generating structures" if not config.ml_filter else "Generating structures (0/0 passed ML filter)"
    )
    
    try:
        if config.method == 'random_enum':
            pipeline.generate_random_enum_structures(config, pbar)
        elif config.method == 'ion_exchange':
            pipeline.generate_ion_exchange_structures(config, pbar)
    finally:
        pbar.close()
        if config.ml_filter:
            print(f"\nML Filtering Summary:")
            print(f"Total structures tested: {pipeline.ml_tested}")
            print(f"Structures passing filter: {pipeline.ml_passed}")
            pass_rate = (pipeline.ml_passed/pipeline.ml_tested*100) if pipeline.ml_tested > 0 else 0
            print(f"Pass rate: {pass_rate:.1f}%")

def main() -> None:
    """Entry point for structure generation pipeline."""
    config = load_config()
    mp_api_key = config['mp_api_key']

    for task_config in config['tasks']:
        run_task(task_config, mp_api_key)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
