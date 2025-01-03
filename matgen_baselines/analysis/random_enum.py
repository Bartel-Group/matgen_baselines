"""
Random structure enumeration module for materials discovery.

This module provides functionality for generating new crystal structures by
randomly combining elements within prototype structures from the Materials Project.
"""

from __future__ import annotations

# Standard library imports
import warnings
import random
import os
import json
import re
import sys
import io
import builtins
from itertools import product
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Set, Iterator, Any, Union, TypeVar

# Third-party imports
from pymatgen.core import Structure, Composition
from pymatgen.io.cif import CifWriter
from mp_api.client import MPRester
from tqdm.std import tqdm as real_tqdm

# Configure warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="chgnet")

# Type aliases
T = TypeVar('T')
PrototypeInfo = Dict[str, str]
OxiStates = Dict[str, List[int]]
Composition_t = Dict[str, float]

# Constants and file paths
CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))
COMMON_OXI_PATH: str = os.path.join(CURRENT_DIR, '../data/element_oxidation_states.json')
PROTOTYPES_PATH: str = os.path.join(CURRENT_DIR, '../data/prototypes.json')

# Load static data
with open(COMMON_OXI_PATH, 'r') as f:
    COMMON_OXI: OxiStates = json.load(f)

with open(PROTOTYPES_PATH, 'r') as f:
    PROTOTYPES: Dict[str, PrototypeInfo] = json.load(f)

class OutputSuppressor:
    """Utility class for suppressing output and tqdm progress bars."""
    
    @staticmethod
    @contextmanager
    def suppress_stdout() -> Iterator[None]:
        """Suppress stdout temporarily."""
        stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = stdout

    @staticmethod
    @contextmanager
    def suppress_output() -> Iterator[None]:
        """Suppress both stdout and tqdm."""
        original_print = builtins.print
        
        def no_print(*args, **kwargs) -> None:
            pass
            
        def fake_tqdm(*args, **kwargs) -> Any:
            if len(args) > 0:
                return args[0]
            return kwargs.get('iterable', None)
        
        builtins.print = no_print
        sys.modules['tqdm'] = type('FakeTqdm', (), {'tqdm': fake_tqdm})
        
        try:
            yield
        finally:
            builtins.print = original_print
            sys.modules['tqdm'] = real_tqdm

class StructureGenerator:
    """Class for generating crystal structures using prototype templates."""

    @staticmethod
    def parse_formula(formula: str) -> Dict[str, int]:
        """Parse chemical formula into element counts."""
        pattern = re.compile(r'([A-Z][a-z]*)(\d*)')
        parts = pattern.findall(formula)
        elements: Dict[str, int] = {}
        for element, count in parts:
            elements[element] = int(count) if count else 1
        return elements

    @staticmethod
    def is_valid_formula(formula: str, oxi_states: OxiStates) -> bool:
        """Check if formula can form a charge-neutral compound."""
        elements = StructureGenerator.parse_formula(formula)
        all_combinations: List[List[Tuple[int, ...]]] = []

        for element, count in elements.items():
            if element not in oxi_states:
                return False
            states = [(count * state,) for state in oxi_states[element]]
            all_combinations.append(states)

        return any(sum(sum(x) for x in combo) == 0 
                  for combo in product(*all_combinations))

    @staticmethod
    def generate_random_composition(
        min_elements: int,
        max_elements: int
    ) -> Composition_t:
        """Generate a random composition with specified number of elements."""
        num_elements = min_elements + int.from_bytes(os.urandom(4), 'big') % (max_elements - min_elements + 1)
        elements = list(COMMON_OXI.keys())
        selected_elements = []

        for _ in range(num_elements):
            index = int.from_bytes(os.urandom(4), 'big') % len(elements)
            selected_elements.append(elements.pop(index))

        composition = {
            elem: int.from_bytes(os.urandom(4), 'big') / (2**32 - 1) + 0.1
            for elem in selected_elements
        }
        total = sum(composition.values())
        return {elem: amount / total for elem, amount in composition.items()}

    def swap_species(
        self,
        structure: Structure,
        target_composition: Composition_t,
        min_num_elems: int = 3
    ) -> List[Structure]:
        """Generate new structures by swapping species."""
        template_struc = structure.copy()
        elements_in_structure = [str(el) for el in structure.composition.elements]
        target_elements = list(target_composition.keys())
        
        all_replacements = [
            list(product([el], target_elements)) 
            for el in elements_in_structure
        ]
        all_replacements = list(product(*all_replacements))

        valid_structures: List[Structure] = []
        for replacement_set in all_replacements:
            structure = template_struc.copy()
            for from_el, to_el in replacement_set:
                structure.replace_species({from_el: to_el})
            
            proposed_elems = set(str(el) for el in structure.composition.elements)
            if (self.is_valid_formula(structure.composition.reduced_formula, COMMON_OXI) 
                and len(proposed_elems) >= min_num_elems):
                valid_structures.append(structure)
                
        return valid_structures

    def retrieve_structures(
        self,
        cmpd_form: str,
        spacegroup_num: Union[str, int],
        mp_api_key: str
    ) -> List[Structure]:
        """Retrieve structures from Materials Project."""
        with OutputSuppressor.suppress_output():
            spacegroup_num = int(spacegroup_num)
            with MPRester(mp_api_key) as mpr:
                docs = mpr.summary.search(
                    formula=cmpd_form,
                    energy_above_hull=(None, 1.0),
                    spacegroup_number=spacegroup_num,
                    fields=['structure']
                )
            return [entry.structure for entry in docs]

    def generate_from_prototype(
        self,
        target_composition: Composition_t,
        prototype_info: PrototypeInfo,
        structures: List[Structure],
        min_num_elems: int = 3
    ) -> Optional[Structure]:
        """Generate a new structure using a prototype template."""
        if not structures:
            return None

        structure = random.choice(structures)
        modified_structures = self.swap_species(
            structure, target_composition, min_num_elems
        )

        return random.choice(modified_structures) if modified_structures else None

    @staticmethod
    def save_structure(structure: Structure, output_path: str) -> None:
        """Save structure to CIF file."""
        CifWriter(structure).write_file(output_path)

    def generate_structures(
        self,
        target_num_strucs: int = 1000,
        min_elements: int = 3,
        max_elements: int = 5,
        max_attempts: int = 500,
        filepath: str = 'Randomly-Enumerated',
        mp_api_key: Optional[str] = None,
        show_progress: bool = False
    ) -> None:
        """Generate random structures based on prototypes."""
        if mp_api_key is None:
            mp_api_key = os.environ.get('MP_API_KEY')
            if mp_api_key is None:
                raise ValueError("No MP API key provided.")

        os.makedirs(filepath, exist_ok=True)
        num_generated = 0
        prototype_labels = list(PROTOTYPES.keys())
        random.shuffle(prototype_labels)

        with tqdm(total=target_num_strucs, 
                 desc="Generating structures", 
                 disable=not show_progress) as pbar:
            
            for prototype_label in prototype_labels:
                if num_generated >= target_num_strucs:
                    break

                try:
                    structures = self.retrieve_structures(
                        PROTOTYPES[prototype_label]['Compound Form'],
                        PROTOTYPES[prototype_label]['Space Group'],
                        mp_api_key
                    )
                except Exception:
                    continue

                if not structures:
                    continue

                for _ in range(max_attempts):
                    if num_generated >= target_num_strucs:
                        break

                    try:
                        composition = self.generate_random_composition(
                            min_elements, max_elements
                        )
                        structure = self.generate_from_prototype(
                            composition, 
                            PROTOTYPES[prototype_label], 
                            structures
                        )

                        if structure:
                            formula = structure.composition.reduced_formula
                            output_path = os.path.join(
                                filepath, f'{formula}_{prototype_label}.cif'
                            )
                            self.save_structure(structure, output_path)
                            num_generated += 1
                            pbar.update(1)
                    except Exception:
                        continue

# Initialize global generator instance
generator = StructureGenerator()
generate_structures = generator.generate_structures
