#!/usr/bin/env python3
"""
Materials Project Novelty Assessment Tool

This script checks the novelty of crystal structures by comparing them against
the Materials Project database. A structure is considered novel if:

1. No materials with the same composition exist in MP
   ...OR...
2. No materials with matching structure exist in MP for the same composition

Usage: python assess_mp_novelty.py
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from mp_api.client import MPRester


class NoveltyAssessment:
    """Class to assess novelty of crystal structures against Materials Project database."""
    
    def __init__(self, mp_api_key: str = 'YOUR_API_KEY'):
        """
        Initialize the novelty assessment tool.
        
        Args:
            mp_api_key: Materials Project API key for database access
        """
        self.mpr = MPRester(mp_api_key)
        # Structure matcher with loose constraints for better matching
        self.matcher = StructureMatcher(
            ltol=0.25,      # Lattice tolerance
            stol=0.4,       # Site tolerance  
            angle_tol=10.0, # Angle tolerance
            attempt_supercell=True
        )
        self.results = []
    
    def check_structure_novelty(self, structure: Structure, filename: str) -> Dict[str, Any]:
        """
        Check if a structure is novel compared to Materials Project database.
        
        Args:
            structure: Pymatgen Structure object
            filename: Name of the structure file
            
        Returns:
            Dictionary containing novelty assessment results
        """
        structure.remove_oxidation_states()
        formula = structure.composition.reduced_formula
        space_group = structure.get_space_group_info()[1]
        
        print(f"Checking novelty for {filename}: {formula} (SG: {space_group})")
        
        try:
            # Get all MP entries with same chemical system
            elements = [str(el) for el in structure.composition.elements]
            mp_entries = self.mpr.get_entries_in_chemsys(elements)
            
            # Filter for exact composition match
            same_comp_entries = [
                entry for entry in mp_entries
                if Composition(entry.formula).reduced_formula == formula
            ]
            
            result = {
                'filename': filename,
                'formula': formula,
                'space_group': space_group,
                'mp_entries_total': len(mp_entries),
                'mp_entries_same_composition': len(same_comp_entries),
                'novel': False,
                'matching_mp_ids': [],
                'error': None
            }
            
            # If no composition match found, it's novel
            if not same_comp_entries:
                result['novel'] = True
                result['reason'] = 'No materials with same composition in MP'
                print(f"  ✓ Novel: No {formula} found in MP database")
            else:
                # Check structure matching with all entries of same composition
                novel = True
                matching_ids = []
                
                for entry in same_comp_entries:
                    if self.matcher.fit(structure, entry.structure):
                        novel = False
                        matching_ids.append(entry.entry_id)
                
                result['novel'] = novel
                result['matching_mp_ids'] = matching_ids
                
                if novel:
                    result['reason'] = f'Structure differs from {len(same_comp_entries)} MP entries with same composition'
                    print(f"  ✓ Novel: Structure unique among {len(same_comp_entries)} MP entries")
                else:
                    result['reason'] = f'Structure matches MP entry/entries: {", ".join(matching_ids)}'
                    print(f"  ✗ Not novel: Matches {len(matching_ids)} MP entries")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            print(f"  ✗ {error_msg}")
            return {
                'filename': filename,
                'formula': formula,
                'space_group': space_group,
                'novel': None,
                'error': error_msg
            }
    
    def process_structures_directory(self, structures_dir: str = "Structures") -> None:
        """
        Process all CIF files in the structures directory.
        
        Args:
            structures_dir: Path to directory containing structure files
        """
        structures_path = Path(structures_dir)
        
        if not structures_path.exists():
            raise FileNotFoundError(f"Structures directory not found: {structures_dir}")
        
        # Find all CIF files
        cif_files = list(structures_path.glob("*.cif"))
        
        if not cif_files:
            print(f"No CIF files found in {structures_dir}")
            return
        
        print(f"Found {len(cif_files)} CIF files to process\n")
        
        # Process each structure file
        for cif_file in sorted(cif_files):
            try:
                structure = Structure.from_file(str(cif_file))
                result = self.check_structure_novelty(structure, cif_file.name)
                self.results.append(result)
                
            except Exception as e:
                error_result = {
                    'filename': cif_file.name,
                    'novel': None,
                    'error': f"Failed to load structure: {str(e)}"
                }
                self.results.append(error_result)
                print(f"  ✗ Error loading {cif_file.name}: {str(e)}")
            
            print()  # Add blank line between entries
    
    def save_results(self, output_file: str = "novelty_results.json") -> None:
        """
        Save results to JSON file and print summary.
        
        Args:
            output_file: Name of output file
        """
        # Add metadata
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_structures': len(self.results),
                'novel_count': sum(1 for r in self.results if r.get('novel') is True),
                'not_novel_count': sum(1 for r in self.results if r.get('novel') is False),
                'error_count': sum(1 for r in self.results if r.get('novel') is None)
            },
            'results': self.results
        }
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Print summary
        print("="*60)
        print("NOVELTY ASSESSMENT SUMMARY")
        print("="*60)
        print(f"Total structures processed: {output_data['metadata']['total_structures']}")
        print(f"Novel structures: {output_data['metadata']['novel_count']}")
        print(f"Non-novel structures: {output_data['metadata']['not_novel_count']}")
        print(f"Errors: {output_data['metadata']['error_count']}")
        print(f"\nResults saved to: {output_file}")
        
        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("-" * 60)
        for result in self.results:
            status = "NOVEL" if result.get('novel') else "NOT NOVEL" if result.get('novel') is False else "ERROR"
            print(f"{result['filename']:<20} | {status:<10} | {result.get('formula', 'N/A')}")
            if result.get('error'):
                print(f"                     | ERROR: {result['error']}")


def main():
    """Main function to run novelty assessment."""
    print("Materials Project Novelty Assessment Tool")
    print("="*50)
    
    # Initialize assessment tool
    assessor = NoveltyAssessment()
    
    # Process all structures
    assessor.process_structures_directory()
    
    # Save and display results
    assessor.save_results()


if __name__ == "__main__":
    main()
