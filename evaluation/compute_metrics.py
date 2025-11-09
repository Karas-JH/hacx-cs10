import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import os
import glob
import torch
import pickle
import struct
import argparse


def compute_weighted_f1(predictions_file, ground_truth_file):
    # Read predictions with robustness to empty or partial files
    try:
        pred_df = pd.read_csv(predictions_file)
    except pd.errors.EmptyDataError:
        print(f"Warning: Predictions file is empty: {predictions_file}")
        return 0.0
    
    # Remove any empty rows
    pred_df = pred_df.dropna(how='all')
    
    # Validate predictions columns
    pred_required = {'file_name', 'predicted_class'}
    if not pred_required.issubset(set(pred_df.columns)):
        missing = pred_required.difference(set(pred_df.columns))
        print(f"Warning: Missing required columns in predictions file: {', '.join(sorted(missing))}")
        return 0.0
    
    # Read ground truth
    try:
        gt_df = pd.read_csv(ground_truth_file)
    except pd.errors.EmptyDataError:
        print(f"Warning: Ground truth file is empty: {ground_truth_file}")
        return 0.0
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found: {ground_truth_file}")
        return 0.0
    
    gt_df = gt_df.dropna(how='all')
    gt_required = {'file_name', 'actual_class'}
    if not gt_required.issubset(set(gt_df.columns)):
        missing = gt_required.difference(set(gt_df.columns))
        print(f"Warning: Missing required columns in ground truth file: {', '.join(sorted(missing))}")
        return 0.0
    
    # Merge on file_name (many-to-one safe). If duplicates, keep first occurrence.
    gt_df = gt_df.drop_duplicates(subset=['file_name'])
    
    # Drop actual_class from predictions if it exists (we'll use ground truth values)
    pred_df_clean = pred_df.drop(columns=['actual_class'], errors='ignore')
    
    # Merge with ground truth
    merged = pred_df_clean.merge(gt_df[['file_name', 'actual_class']], on='file_name', how='inner')
    
    # Filter invalid rows
    merged = merged.dropna(subset=['predicted_class', 'actual_class'])
    if merged.empty:
        print("Warning: No overlapping rows between predictions and ground truth to compute F1.")
        return 0.0
    
    # Ensure numeric labels
    try:
        y_true = merged['actual_class'].astype(int).values
        y_pred = merged['predicted_class'].astype(int).values
    except Exception:
        # If labels are strings like 'smoke', 'haze', 'normal', map to indices
        label_map = {'smoke': 0, 'haze': 1, 'normal': 2}
        y_true = merged['actual_class'].map(label_map).values
        y_pred = merged['predicted_class'].map(label_map).values
        # Drop rows that failed to map
        mask = (~pd.isna(y_true)) & (~pd.isna(y_pred))
        y_true = y_true[mask].astype(int)
        y_pred = y_pred[mask].astype(int)
        if len(y_true) == 0:
            print("Warning: Could not map class labels to integers for F1 computation.")
            return 0.0
    
    # Compute weighted F1 score
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    return f1_weighted


def parse_gguf_metadata(file_path):
    try:
        with open(file_path, 'rb') as f:
            # Read magic number (4 bytes)
            magic = f.read(4)
            if magic != b'GGUF':
                return None
            
            # Read version (4 bytes, uint32)
            version = struct.unpack('<I', f.read(4))[0]
            
            # Read tensor count (8 bytes, uint64 for v2+, uint32 for v1)
            if version >= 2:
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
            else:
                tensor_count = struct.unpack('<I', f.read(4))[0]
                metadata_kv_count = struct.unpack('<I', f.read(4))[0]
            
            # Try to get parameter count from file size as estimation
            # (More accurate parsing would require traversing the entire metadata)
            file_size = os.path.getsize(file_path)
            
            # Rough estimation: assuming 4 bytes per parameter for FP32
            # For quantized models, this will be less, but we'll use tensor_count
            # as a rough indicator
            return f"~{tensor_count} tensors (GGUF)"
            
    except Exception as e:
        print(f"Warning: Could not parse GGUF file: {str(e)}")
        return None


def count_model_parameters(weights_dir):
    model_params = {}
    
    # Look for common model weight file formats
    weight_patterns = ['*.pth', '*.pt', '*.pkl', '*.h5', '*.weights', '*.gguf']
    
    for pattern in weight_patterns:
        weight_files = glob.glob(os.path.join(weights_dir, pattern))
        
        for weight_file in weight_files:
            model_name = os.path.basename(weight_file)
            
            try:
                # Try loading as PyTorch model
                if weight_file.endswith(('.pth', '.pt')):
                    # Try with weights_only=True first (safer), fall back to False for custom models
                    try:
                        checkpoint = torch.load(weight_file, map_location='cpu', weights_only=True)
                    except Exception:
                        checkpoint = torch.load(weight_file, map_location='cpu', weights_only=False)
                    
                    total_params = 0
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        # Try common keys for state dict
                        state_dict = None
                        for key in ['model_state_dict', 'state_dict', 'model']:
                            if key in checkpoint:
                                state_dict = checkpoint[key]
                                break
                        
                        # If no common key found, assume the dict itself is the state dict
                        if state_dict is None:
                            state_dict = checkpoint
                        
                        # Count parameters from state dict
                        if isinstance(state_dict, dict):
                            total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
                        elif hasattr(state_dict, 'parameters'):
                            # If it's a model object stored in the dict
                            total_params = sum(p.numel() for p in state_dict.parameters())
                    
                    elif hasattr(checkpoint, 'parameters'):
                        # If the checkpoint itself is a model
                        total_params = sum(p.numel() for p in checkpoint.parameters())
                    
                    model_params[model_name] = total_params if total_params > 0 else 'N/A'
                
                # Try loading as pickle
                elif weight_file.endswith('.pkl'):
                    with open(weight_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Try to count parameters if it's a PyTorch model
                    if hasattr(model, 'parameters'):
                        total_params = sum(p.numel() for p in model.parameters())
                        model_params[model_name] = total_params
                    else:
                        model_params[model_name] = 'N/A'
                
                # Try loading as GGUF
                elif weight_file.endswith('.gguf'):
                    param_info = parse_gguf_metadata(weight_file)
                    if param_info:
                        model_params[model_name] = param_info
                    else:
                        model_params[model_name] = 'N/A (GGUF)'
                
                # Handle Keras/TensorFlow H5 models
                elif weight_file.endswith(('.h5', '.hdf5')):
                    try:
                        import tensorflow as tf

                        model = tf.keras.models.load_model(weight_file, compile=False)
                        model_params[model_name] = model.count_params()
                    except ModuleNotFoundError:
                        try:
                            import h5py

                            total_params = 0

                            def _accumulate(_, obj):
                                nonlocal total_params
                                if isinstance(obj, h5py.Dataset):
                                    total_params += int(np.prod(obj.shape))

                            with h5py.File(weight_file, 'r') as f:
                                f.visititems(_accumulate)

                            model_params[model_name] = total_params if total_params > 0 else 'N/A'
                        except Exception:
                            model_params[model_name] = 'N/A (h5)'
                    except Exception:
                        model_params[model_name] = 'Error'

            except Exception as e:
                print(f"Warning: Could not load {model_name}: {str(e)}")
                model_params[model_name] = 'Error'
    
    return model_params


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Compute evaluation metrics from model predictions.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-p', '--predictions',
        type=str,
        default='./output/predictions.csv',
        help='Path to predictions CSV file with columns: file_name, predicted_class, actual_class'
    )
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights',
        help='Directory containing model weight files'
    )
    parser.add_argument(
        '-g', '--ground-truth',
        type=str,
        default='./ground_truth.csv',
        help='Path to ground truth CSV with columns: file_name, actual_class (file_path optional)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./output/eval_result.csv',
        help='Path to output CSV file for evaluation results'
    )
    
    args = parser.parse_args()
    
    # Paths
    predictions_file = args.predictions
    weights_dir = args.weights
    output_file = args.output
    ground_truth_file = args.ground_truth
    
    # Compute weighted F1 score
    print("Computing weighted F1 score (using ground_truth.csv)...")
    f1_weighted = compute_weighted_f1(predictions_file, ground_truth_file)
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    
    # Count model parameters
    print(f"\nCounting model parameters from {weights_dir}...")
    model_params = count_model_parameters(weights_dir)
    
    # Prepare results
    results = []
    
    if model_params:
        for model_name, param_count in model_params.items():
            results.append({
                'model_name': model_name,
                'weighted_f1_score': f1_weighted,
                'num_parameters': param_count
            })
        print(f"\nFound {len(model_params)} model(s)")
    else:
        # If no models found, still output the F1 score
        results.append({
            'model_name': 'N/A',
            'weighted_f1_score': f1_weighted,
            'num_parameters': 'N/A'
        })
        print("\nNo model weight files found in weights directory")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Display results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)


if __name__ == '__main__':
    main()

