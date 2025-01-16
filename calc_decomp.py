from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from mp_api.client import MPRester
import pandas as pd
import sys

def calculate_decomp_energies(input_file, api_key=None):
    """
    Calculate decomposition energies for compositions in the input_file.
    
    Args:
        input_file (str): Path to CSV file with columns: composition, energy_per_atom
                         Optional column: decomp_energy
        api_key (str): Materials Project API key. Must be provided.
    """
    if api_key is None:
        print("Error: Materials Project API key must be provided")
        sys.exit(1)

    # Read input file
    df = pd.read_csv(input_file)
    required_cols = ['composition', 'energy_per_atom']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input file must contain columns: {required_cols}")
        sys.exit(1)

    # Check if decomposition energies are already computed
    if 'decomp_energy' in df.columns and df['decomp_energy'].notna().all():
        print("Decomposition energies already computed for all entries.")
        return
    
    # Initialize MPRester
    try:
        mpr = MPRester(api_key)
    except Exception as e:
        print(f"Error initializing MPRester: Invalid API key or connection error")
        sys.exit(1)
    
    # Initialize list to store decomposition energies
    decomp_energies = []
    
    # Process each composition
    for idx, row in df.iterrows():
        if 'decomp_energy' in df.columns and pd.notna(row.get('decomp_energy')):
            decomp_energies.append(row['decomp_energy'])
            continue
            
        try:
            composition = Composition(row['composition'])
            energy = float(row['energy_per_atom']) * composition.num_atoms
            
            target_entry = ComputedEntry(composition, energy)
            elements = [str(el) for el in composition.elements]
            mp_entries = mpr.get_entries_in_chemsys(elements)
            
            phase_diagram = PhaseDiagram(mp_entries + [target_entry])
            decomp_energy = phase_diagram.get_decomp_and_e_above_hull(target_entry)[1]
            
            if decomp_energy == 0:
                decomp_energy = phase_diagram.get_equilibrium_reaction_energy(target_entry)
                
            decomp_energies.append(round(float(decomp_energy), 3))
            print(f"Processed {row['composition']}: {decomp_energies[-1]} eV/atom")
            
        except Exception as e:
            print(f"Error processing {row['composition']}: {e}")
            decomp_energies.append(None)
    
    df['decomp_energy'] = decomp_energies
    df.to_csv(input_file, index=False)
    print(f"\nResults saved to {input_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py input_file.csv")
        print("\nInput file format (CSV):")
        print("composition,energy_per_atom[,decomp_energy]")
        print("Example:")
        print("Fe3Al,-7.4878")
        print("AlFe2,-7.0365")
        sys.exit(1)
    
    try:
        calculate_decomp_energies(sys.argv[1])
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
