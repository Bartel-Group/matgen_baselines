"""
CGCNN-based property prediction module for crystal structures.

This module provides functionality for predicting material properties using
the Crystal Graph Convolutional Neural Network (CGCNN) model.
"""

from __future__ import annotations
import os
import tempfile
import csv
import json
import shutil
from typing import Any, Dict, List, Tuple, Union
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pymatgen.core import Structure
from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet

# Type aliases
ModelArgs = Dict[str, Union[int, bool]]
StateDict = Dict[str, Tensor]
CGCNNInput = Tuple[Tensor, Tensor, List[int], List[int]]

class Normalizer:
    """
    A class to normalize and denormalize tensors using mean and standard deviation.
    
    This normalizer can be used to standardize data during training and restore
    the original scale during prediction.
    """
    
    def __init__(self, tensor: Tensor) -> None:
        """
        Initialize the normalizer with sample tensor to calculate mean and std.
        
        Args:
            tensor (Tensor): Sample tensor used to compute normalization parameters
        """
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor: Tensor) -> Tensor:
        """
        Normalize a tensor using stored mean and standard deviation.
        
        Args:
            tensor (Tensor): Input tensor to normalize
            
        Returns:
            Tensor: Normalized tensor
        """
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: Tensor) -> Tensor:
        """
        Restore the original scale of a normalized tensor.
        
        Args:
            normed_tensor (Tensor): Normalized input tensor
            
        Returns:
            Tensor: Denormalized tensor in original scale
        """
        return normed_tensor * self.std + self.mean

    def state_dict(self) -> Dict[str, Tensor]:
        """
        Get the state dictionary containing mean and std.
        
        Returns:
            Dict[str, Tensor]: State dictionary with mean and std values
        """
        return {
            'mean': self.mean,
            'std': self.std
        }

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """
        Load state dictionary to restore normalizer parameters.
        
        Args:
            state_dict (Dict[str, Tensor]): State dictionary containing mean and std
        """
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def predict_property(
    structure: Structure,
    model_path: str
) -> float:
    """
    Predict a property for a given crystal structure using a pre-trained CGCNN model.
    
    This function takes a crystal structure and a path to a pre-trained CGCNN model,
    creates necessary temporary files, and runs the prediction pipeline to estimate
    the target property.
    
    Args:
        structure (pymatgen.core.Structure): The input crystal structure
        model_path (str): Path to the pre-trained model checkpoint file
        
    Returns:
        float: Predicted property value for the given structure
        
    Raises:
        FileNotFoundError: If atom_init.json is not found in the model directory
    """
    # Ensure absolute paths
    model_path = os.path.abspath(model_path)
    
    # Get the directory where the model is located
    model_dir = os.path.dirname(model_path)
    atom_init_path = os.path.join(model_dir, 'atom_init.json')
    
    # Check required files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    if not os.path.exists(atom_init_path):
        raise FileNotFoundError(f"atom_init.json must be present in {model_dir}")

    # Create a base directory for temporary files
    temp_base_dir = os.path.join(tempfile.gettempdir(), 'cgcnn_prediction')
    os.makedirs(temp_base_dir, exist_ok=True)

    # Create a unique temporary directory
    temp_dir = tempfile.mkdtemp(dir=temp_base_dir, prefix='prediction_')
    
    try:
        # Save the structure as a temporary CIF file
        tmp_cif_path = os.path.join(temp_dir, 'tmp.cif')
        structure.to(filename=tmp_cif_path)
        
        # Verify CIF file creation
        if not os.path.exists(tmp_cif_path):
            raise IOError(f"Failed to create CIF file at {tmp_cif_path}")
        
        # Create a temporary id_prop.csv file with dummy target
        with open(os.path.join(temp_dir, 'id_prop.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['tmp', '0'])  # '0' as a dummy target value
        
        # Copy the atom_init.json to the temporary directory
        shutil.copy(atom_init_path, os.path.join(temp_dir, 'atom_init.json'))
        
        # Initialize dataset and dataloader
        dataset = CIFData(temp_dir)
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pool
        )
        
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model_args = checkpoint['args']
        
        # Extract feature dimensions from the first structure
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        
        # Initialize and configure the model
        model = CrystalGraphConvNet(
            orig_atom_fea_len,
            nbr_fea_len,
            atom_fea_len=model_args['atom_fea_len'],
            n_conv=model_args['n_conv'],
            h_fea_len=model_args['h_fea_len'],
            n_h=model_args['n_h'],
            classification=False
        )
        
        # Load model state and normalizer
        model.load_state_dict(checkpoint['state_dict'])
        normalizer = Normalizer(torch.zeros(3))
        normalizer.load_state_dict(checkpoint['normalizer'])
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            for input, target, batch_cif_ids in test_loader:
                input_var = (
                    Variable(input[0]),
                    Variable(input[1]),
                    input[2],
                    input[3]
                )
                # Compute and denormalize output
                output = model(*input_var)
                predicted_value = normalizer.denorm(output.data.cpu()).item()
                return predicted_value
        
        raise RuntimeError("No prediction generated")
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            print(f"Warning: Failed to clean up temporary directory {temp_dir}: {cleanup_error}")
