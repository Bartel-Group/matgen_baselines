"""
ML-based structure filtering and analysis module for materials discovery.

This module provides functionality for analyzing crystal structures using ML models
and phase diagram analysis, supporting stability, band gap, and bulk modulus predictions.
"""

from __future__ import annotations

# Standard library imports
import warnings
import os
import json
import sys
import io
from typing import Dict, Optional, Tuple, Any, List, Union
from itertools import combinations
from contextlib import contextmanager

# Third-party imports
from pymatgen.core import Structure, Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from chgnet.model import CHGNet, StructOptimizer
from mp_api.client import MPRester

# Local imports
from .cgcnn_predict import predict_property

# Configure warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="chgnet")

# File paths and constants
CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))
ELEM_ENERGIES_PATH: str = os.path.join(CURRENT_DIR, '../data/element_energies.json')
MP_ENERGIES_PATH: str = os.path.join(CURRENT_DIR, '../data/mp_formation_energies.json')
PRETRAINED_DIR: str = os.path.join(CURRENT_DIR, 'pre-trained')

# Load reference data
with open(ELEM_ENERGIES_PATH, 'r') as f:
    ELEM_ENERGIES: Dict[str, float] = json.load(f)

with open(MP_ENERGIES_PATH, 'r') as f:
    MP_ENERGIES: Dict[str, Dict[str, Union[str, float]]] = json.load(f)

@contextmanager
def suppress_stdout() -> None:
    """Context manager to suppress stdout and warnings."""
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = stdout

class StructureAnalyzer:
    """Class for analyzing crystal structures using ML models and phase diagrams."""

    @staticmethod
    def compute_formation_energy(structure: Structure, energy_per_atom: float) -> float:
        """Compute formation energy per atom."""
        n_atoms = len(structure)
        total_energy = n_atoms * energy_per_atom
        for element, amount in structure.composition.items():
            total_energy -= amount * ELEM_ENERGIES[str(element)]
        return total_energy / n_atoms

    @staticmethod
    def create_computed_entry(structure: Structure, formation_energy: float) -> ComputedEntry:
        """Create ComputedEntry for phase diagram analysis."""
        return ComputedEntry(
            composition=structure.composition,
            energy=formation_energy * len(structure),
            parameters={'formation_energy_per_atom': formation_energy}
        )

    @staticmethod
    def get_mp_entries_for_chemsys(elements: List[str]) -> List[ComputedEntry]:
        """Get all relevant MP entries for a chemical system."""
        entries = []
        for mp_id, data in MP_ENERGIES.items():
            try:
                comp = Composition(data['formula'])
                comp_elements = set(str(el) for el in comp.elements)
                if comp_elements.issubset(set(elements)):
                    energy = data['e_form'] * comp.num_atoms
                    entry = ComputedEntry(
                        composition=comp,
                        energy=energy,
                        entry_id=mp_id
                    )
                    entries.append(entry)
            except Exception:
                continue
        return entries

    @staticmethod
    def check_pretrained_files(model_type: str) -> Tuple[str, str]:
        """Verify and return paths for pretrained model files."""
        model_path = os.path.join(PRETRAINED_DIR, f'{model_type}.pth.tar')
        atom_init_path = os.path.join(PRETRAINED_DIR, 'atom_init.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(atom_init_path):
            raise FileNotFoundError(f"atom_init.json not found at {atom_init_path}")
            
        return model_path, atom_init_path

    def analyze_structure_stability(
        self,
        structure: Structure,
        chgnet_model: CHGNet,
        mpr: MPRester,
        threshold: float = 0.1,
        verbose: bool = False,
        optimizer: Optional[StructOptimizer] = None
    ) -> Tuple[bool, Optional[Structure]]:
        """Analyze thermodynamic stability of a structure."""
        try:
            structure = structure.remove_oxidation_states()
            if optimizer is None:
                optimizer = StructOptimizer()

            with suppress_stdout():
                relax_result = optimizer.relax(structure)

            relaxed_structure = relax_result["final_structure"]
            energy_per_atom = relax_result['trajectory'].energies[-1] / len(relaxed_structure)
            formation_energy = self.compute_formation_energy(relaxed_structure, energy_per_atom)
            struct_entry = self.create_computed_entry(relaxed_structure, formation_energy)

            elements = [str(el) for el in relaxed_structure.composition.elements]
            entries = self.get_mp_entries_for_chemsys(elements)

            if not entries:
                return False, relaxed_structure

            pd = PhaseDiagram(entries)
            decomp = pd.get_decomposition(relaxed_structure.composition)
            hull_energy = sum(amt * pd.get_form_energy_per_atom(entry)
                            for entry, amt in decomp.items())
            
            energy_above_hull = formation_energy - hull_energy
            return energy_above_hull <= threshold, relaxed_structure

        except Exception as e:
            if verbose:
                print(f"Error analyzing stability: {str(e)}")
            return False, None

    def analyze_band_gap(
        self,
        structure: Structure,
        target_gap: float,
        threshold: float
    ) -> bool:
        """Analyze if predicted band gap matches target."""
        try:
            model_path, _ = self.check_pretrained_files('band-gap')
            predicted_gap = predict_property(structure, model_path)
            return abs(predicted_gap - target_gap) <= threshold
        except Exception as e:
            print(f"Error predicting band gap: {str(e)}")
            return False

    def analyze_bulk_modulus(
        self,
        structure: Structure,
        target_modulus: float,
        threshold: float
    ) -> bool:
        """Analyze if predicted bulk modulus matches target."""
        try:
            model_path, _ = self.check_pretrained_files('bulk-moduli')
            predicted_modulus = 100 * predict_property(structure, model_path)
            return abs(predicted_modulus - target_modulus) <= threshold
        except Exception as e:
            print(f"Error predicting bulk modulus: {str(e)}")
            return False

    def analyze_structure(
        self,
        structure: Structure,
        filter_type: str,
        target_params: Optional[Tuple[float, float]],
        chgnet_model: Optional[CHGNet] = None,
        mpr: Optional[MPRester] = None,
        verbose: bool = False,
        optimizer: Optional[StructOptimizer] = None
    ) -> Tuple[bool, Optional[Structure]]:
        """Analyze a structure based on specified filter type."""
        if filter_type == "stability":
            if chgnet_model is None or mpr is None:
                raise ValueError("CHGNet model and MPRester required for stability analysis")
            threshold = target_params[1] if target_params else 0.1
            return self.analyze_structure_stability(
                structure, chgnet_model, mpr, threshold, verbose, optimizer
            )
        
        if filter_type == "band_gap":
            if target_params is None:
                raise ValueError("Target parameters required for band gap analysis")
            target, threshold = target_params
            return self.analyze_band_gap(structure, target, threshold), None
        
        if filter_type == "bulk_modulus":
            if target_params is None:
                raise ValueError("Target parameters required for bulk modulus analysis")
            target, threshold = target_params
            return self.analyze_bulk_modulus(structure, target, threshold), None
        
        raise ValueError(f"Unknown filter type: {filter_type}")

# Initialize global analyzer instance
analyzer = StructureAnalyzer()
analyze_structure = analyzer.analyze_structure
