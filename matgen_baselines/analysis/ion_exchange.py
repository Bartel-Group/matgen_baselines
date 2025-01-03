"""
Ion exchange structure generation module for materials discovery.

This module provides functionality for generating new crystal structures through ion
substitution, targeting specific properties like stability, band gap, or bulk modulus.
"""

from __future__ import annotations

# Standard library imports
import warnings
import os
from typing import List, Optional, Tuple, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager

# Third-party imports
import sys
import io
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.analysis.structure_prediction import substitutor
from pymatgen.analysis.bond_valence import BVAnalyzer
from mp_api.client import MPRester

# Local imports
from ..utils.helpers import load_json, save_structure_to_cif

# Configure warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="chgnet")
warnings.filterwarnings("ignore", category=UserWarning)

# Type aliases
MPEntry = Any  # Type for Materials Project entries
PropertyTarget = Optional[Tuple[float, float]]

# Constants
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR.parent / 'data'
COMMON_OXI = load_json(DATA_DIR / 'element_oxidation_states.json')

@dataclass
class SubstitutionConfig:
    """Configuration for ion exchange structure generation."""
    target_num_strucs: int = 2000
    min_elements: int = 3
    max_elements: int = 5
    max_attempts: int = 500
    limit_per_struc: int = 10
    filepath: str = 'Ion-Exchanged'
    filter_type: str = "stability"
    target_params: Optional[Tuple[float, float]] = None
    show_progress: bool = False

class OutputManager:
    """Utility class for managing output suppression."""
    
    @staticmethod
    @contextmanager
    def suppress_stdout():
        """Context manager to suppress stdout."""
        stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = stdout

class IonExchanger:
    """Handles ion exchange structure generation and property targeting."""
    _instance = None

    def __init__(self, mp_api_key: Optional[str] = None):
        """Initialize ion exchanger with API key."""
        self.mp_api_key = mp_api_key
        
    @classmethod
    def get_instance(cls, mp_api_key: Optional[str] = None) -> 'IonExchanger':
        """Get or create singleton instance with API key."""
        if cls._instance is None:
            cls._instance = cls(mp_api_key)
        elif mp_api_key is not None:
            cls._instance.mp_api_key = mp_api_key
        return cls._instance

    def _check_api_key(self) -> str:
        """Verify and return valid API key."""
        api_key = self.mp_api_key or os.environ.get('MP_API_KEY')
        if not api_key:
            raise ValueError("No MP API key provided.")
        return api_key

    @staticmethod
    def format_species(element: str, oxidation_state: int) -> str:
        """Format element and oxidation state into species string."""
        sign = '+' if oxidation_state > 0 else '-'
        if abs(oxidation_state) == 1:
            return f"{element}{sign}"
        return f"{element}{abs(oxidation_state)}{sign}"

    def add_oxidation_states(self, structure: Structure) -> Optional[Structure]:
        """Add oxidation states to structure using BVAnalyzer."""
        try:
            return BVAnalyzer().get_oxi_state_decorated_structure(structure)
        except Exception:
            return None

    def substitute_structure(self, structure: Structure) -> List[Structure]:
        """Generate new structures through ion substitution."""
        structure_with_oxi = self.add_oxidation_states(structure)
        if not structure_with_oxi:
            return []

        sub = substitutor.Substitutor()
        final_structures: List[Structure] = []

        # Get species strings
        species_strings = [
            self.format_species(site.specie.element.symbol, int(site.specie.oxi_state))
            for site in structure_with_oxi
        ]
        unique_species = list(set(species_strings))

        # Try substitutions
        for species in unique_species:
            element = ''.join(c for c in species if not c.isdigit() and c not in '+-')
            
            for new_el, ox_states in COMMON_OXI.items():
                for ox_state in ox_states:
                    new_species = self.format_species(new_el, ox_state)
                    
                    if new_species == species:
                        continue
                        
                    modified_species = [
                        new_species if sp == species else sp 
                        for sp in unique_species
                    ]
                    
                    # Check for duplicates
                    if len(set(modified_species)) != len(modified_species):
                        continue

                    try:
                        generated = sub.pred_from_structures(
                            modified_species,
                            [{'structure': structure_with_oxi, 'id': 'my-id'}]
                        )
                        final_structures.extend(s.final_structure for s in generated)
                    except Exception as e:
                        if "species in target_species are not allowed" not in str(e):
                            continue

        return final_structures

    def get_mp_entries(
        self,
        min_elements: int,
        max_elements: int,
        stability: Optional[float] = None,
        band_gap_target: PropertyTarget = None,
        bulk_modulus_target: PropertyTarget = None
    ) -> List[MPEntry]:
        """Retrieve Materials Project entries matching criteria."""
        with OutputManager.suppress_stdout(), MPRester(self._check_api_key()) as mpr:
            # Base query
            query: Dict[str, Any] = {"num_elements": (min_elements, max_elements)}
            if stability is not None:
                query["energy_above_hull"] = (None, stability)
            if band_gap_target:
                target, threshold = band_gap_target
                query["band_gap"] = (target - threshold, target + threshold)

            # Get entries
            entries = mpr.summary.search(
                fields=["structure", "energy_above_hull", "band_gap", "bulk_modulus"],
                **query
            )

            # Sort/filter entries
            if band_gap_target:
                target = band_gap_target[0]
                entries.sort(key=lambda x: abs(x.band_gap - target))
            elif bulk_modulus_target:
                target, threshold = bulk_modulus_target
                filtered = []
                for entry in entries:
                    if not (entry.bulk_modulus and isinstance(entry.bulk_modulus, dict)):
                        continue
                    bulk_mod = entry.bulk_modulus.get('vrh')
                    if bulk_mod is None:
                        continue
                    if target - threshold <= bulk_mod <= target + threshold:
                        filtered.append(entry)
                filtered.sort(key=lambda x: abs(x.bulk_modulus['vrh'] - target))
                return filtered

            return entries

    def generate_structures(self, config: SubstitutionConfig) -> None:
        """Generate structures according to configuration."""
        os.makedirs(config.filepath, exist_ok=True)

        # Get entries based on filter type
        if config.filter_type == "band_gap":
            mp_entries = self.get_mp_entries(
                config.min_elements, config.max_elements,
                band_gap_target=config.target_params
            )
        elif config.filter_type == "bulk_modulus":
            mp_entries = self.get_mp_entries(
                config.min_elements, config.max_elements,
                bulk_modulus_target=config.target_params
            )
        else:  # stability
            stability = config.target_params[0] if config.target_params else 0.0
            mp_entries = self.get_mp_entries(
                config.min_elements, config.max_elements,
                stability=stability
            )

        if not mp_entries:
            print("No materials found matching the criteria.")
            return

        # Generate structures
        with tqdm(
            total=config.target_num_strucs,
            desc="Generating structures",
            disable=not config.show_progress
        ) as pbar:
            self._generate_from_entries(mp_entries, config, pbar)

    def _generate_from_entries(
        self,
        entries: List[MPEntry],
        config: SubstitutionConfig,
        pbar: tqdm
    ) -> None:
        """Generate structures from MP entries."""
        num_generated = 0
        attempts = 0

        for entry in entries:
            if num_generated >= config.target_num_strucs:
                break
            if attempts >= config.max_attempts:
                break

            attempts += 1
            substituted = self.substitute_structure(entry.structure)

            for idx, structure in enumerate(substituted):
                if idx >= config.limit_per_struc:
                    break
                if num_generated >= config.target_num_strucs:
                    break

                formula = structure.composition.reduced_formula
                output_path = os.path.join(config.filepath, f"{formula}.cif")
                save_structure_to_cif(structure, output_path)
                num_generated += 1
                pbar.update(1)

# Functions for backward compatibility
def get_exchanger(mp_api_key: Optional[str] = None) -> IonExchanger:
    """Get the global ion exchanger instance."""
    return IonExchanger.get_instance(mp_api_key)

def generate_structures(config: SubstitutionConfig, mp_api_key: Optional[str] = None) -> None:
    """Generate structures using ion exchange."""
    exchanger = get_exchanger(mp_api_key)
    exchanger.generate_structures(config)

if __name__ == "__main__":
    config = SubstitutionConfig()
    # MP_API_KEY should be set in environment or passed explicitly
    generate_structures(config)
