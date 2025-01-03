import json
import os
from typing import Any, Dict, Union, TypeVar
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

# Type variable for generic JSON data
JsonType = TypeVar('JsonType')

def load_json(filename: str) -> JsonType:
    """
    Load and parse JSON data from a file.
    
    Args:
        filename (str): Path to the JSON file to load
        
    Returns:
        JsonType: Parsed JSON data (can be dict, list, or primitive types)
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        json.JSONDecodeError: If the file contains invalid JSON
        
    Note:
        This function uses a generic type variable to allow for different
        JSON data structures while maintaining type safety.
    """
    with open(filename, 'r') as f:
        return json.load(f)

def save_structure_to_cif(structure: Structure, output_path: str) -> None:
    """
    Save a pymatgen Structure object to a CIF file.
    
    Args:
        structure (Structure): Pymatgen Structure object to save
        output_path (str): Path where the CIF file should be saved
        
    Raises:
        ValueError: If the structure is invalid or cannot be converted to CIF format
        OSError: If there are permission issues or the directory doesn't exist
        
    Note:
        Creates parent directories if they don't exist. The CIF file will be
        overwritten if it already exists at the specified path.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    writer = CifWriter(structure)
    writer.write_file(output_path)
