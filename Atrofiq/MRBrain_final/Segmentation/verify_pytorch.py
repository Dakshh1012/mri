
import os
import sys
import numpy as np
import torch
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mri_pipeline_pytorch as pt_pipeline

def verify_pytorch():
    print("Setting up PyTorch verification...")
    
    # Paths
    synthseg_home = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(synthseg_home, 'models')
    
    path_model_segmentation_pt = os.path.join(models_dir, 'synthseg_2.0_pytorch.pth')
    path_model_parcellation_pt = os.path.join(models_dir, 'synthseg_parc_2.0_pytorch.pth')
    
    # Test both modes
    modes = [
        {'do_parcellation': False, 'name': 'Segmentation Only'},
        {'do_parcellation': True, 'name': 'Segmentation + Parcellation'}
    ]
    
    device = torch.device('cpu')
    
    for mode in modes:
        print(f"\nTesting {mode['name']}...")
        do_parcellation = mode['do_parcellation']
        
        try:
            # Build PyTorch Model
            print("Building PyTorch model...")
            pt_model = pt_pipeline.build_model_pytorch(
                path_seg=path_model_segmentation_pt,
                path_parc=path_model_parcellation_pt,
                do_parcellation=do_parcellation,
                device=device
            )
            
            # Create Input
            print("Generating input...")
            # Shape: (1, 1, 160, 160, 160)
            input_shape = (160, 160, 160)
            np.random.seed(42)
            input_data = np.random.rand(*input_shape).astype(np.float32)
            
            # PT Input: (1, 1, D, H, W)
            pt_input_np = input_data[np.newaxis, np.newaxis, ...]
            pt_input = torch.from_numpy(pt_input_np).float().to(device)
            
            # Run Inference
            print("Running PyTorch inference...")
            with torch.no_grad():
                if do_parcellation:
                    pt_seg_tensor, pt_parc_tensor = pt_model(pt_input)
                    print(f"Seg Output Shape: {pt_seg_tensor.shape}")
                    print(f"Parc Output Shape: {pt_parc_tensor.shape}")
                else:
                    pt_seg_tensor = pt_model(pt_input)
                    print(f"Seg Output Shape: {pt_seg_tensor.shape}")
                    
            print("Success!")
            
        except Exception as e:
            print(f"Failed: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    verify_pytorch()
