#!/usr/bin/env python3
"""
Script to run Docker-based model evaluation and measure inference time.
This script:
1. Starts a Docker container with the model
2. Runs predictions on test data (.tif files)
3. Measures the time to process all files and specifically the last file
4. Outputs timing statistics
"""

import subprocess
import time
import os
import glob
import argparse
import sys
import json
from pathlib import Path
import threading
import csv


def get_test_files(test_data_dir):
    """Get all image files (tif/png/jpg) from test_data directory recursively."""
    patterns = [
        os.path.join(test_data_dir, '**', '*.tif'),
        os.path.join(test_data_dir, '**', '*.png'),
        os.path.join(test_data_dir, '**', '*.jpg'),
        os.path.join(test_data_dir, '**', '*.jpeg'),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(files)
    if not files:
        print(f"Error: No image files found in {test_data_dir}")
        sys.exit(1)
    return files


def check_docker_installed():
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Docker found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Docker is not installed or not in PATH")
        return False


def check_docker_image_exists(image_name):
    """Check if the Docker image exists."""
    try:
        result = subprocess.run(
            ['docker', 'images', '-q', image_name],
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout.strip():
            print(f"Docker image '{image_name}' found")
            return True
        else:
            print(f"Docker image '{image_name}' not found")
            return False
    except subprocess.CalledProcessError:
        return False


def monitor_predictions_file(predictions_file, expected_file_count, result_dict, start_time):
    """
    Monitor predictions.csv file and detect when all predictions have been written.
    
    Args:
        predictions_file: Path to predictions CSV file
        expected_file_count: Expected number of test files to process
        result_dict: Dictionary to store the result timestamp
        start_time: Start time of the evaluation
    """
    print(f"Monitoring predictions file: {predictions_file}")
    print(f"Waiting for {expected_file_count} predictions to be written")
    
    last_prediction_found = False
    poll_interval = 0.1  # Check every 100ms
    last_count = 0
    
    while not last_prediction_found:
        try:
            if os.path.exists(predictions_file):
                with open(predictions_file, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    current_count = len(rows)
                    
                    # Show progress when count changes
                    if current_count > last_count:
                        print(f"  Progress: {current_count}/{expected_file_count} predictions written", end='\r')
                        last_count = current_count
                    
                    # Check if we've reached the expected count
                    if current_count >= expected_file_count:
                        # Record the timestamp when last prediction is written
                        result_dict['last_prediction_time'] = time.time()
                        result_dict['time_to_last_prediction'] = time.time() - start_time
                        result_dict['total_predictions'] = current_count
                        last_prediction_found = True
                        print(f"\nAll {current_count} predictions detected at {result_dict['time_to_last_prediction']:.4f}s")
                        break
            
            if not last_prediction_found:
                time.sleep(poll_interval)
                
        except Exception as e:
            # File might be being written, try again
            time.sleep(poll_interval)

def run_docker_inference(image_name, container_name, test_data_dir, output_dir, expected_file_count, weights_dir=None, volume_mounts=None, predictions_filename='predictions.csv'):
    """
    Run inference using Docker container.
    
    Args:
        image_name: Name of the Docker image
        container_name: Name for the container
        test_data_dir: Directory containing test data
        output_dir: Directory for output predictions
        expected_file_count: Number of test files expected to be processed
        weights_dir: Directory containing model weights (optional)
        volume_mounts: Additional volume mounts as list of tuples [(host_path, container_path), ...]
        predictions_filename: Name of the predictions CSV file to monitor
    
    Returns:
        Dictionary with timing information
    """
    # Prepare volume mounts
    volumes = [
        f"{os.path.abspath(test_data_dir)}:/data/test:ro",
        f"{os.path.abspath(output_dir)}:/data/output"
    ]
    
    # Add weights directory if provided
    if weights_dir and os.path.exists(weights_dir):
        volumes.append(f"{os.path.abspath(weights_dir)}:/app/weights:ro")
        print(f"Mounting weights directory: {weights_dir}")
    
    if volume_mounts:
        for host_path, container_path in volume_mounts:
            volumes.append(f"{os.path.abspath(host_path)}:{container_path}")
    
    # Build docker run command
    docker_cmd = [
        'docker', 'run',
        '--name', container_name,
        '--rm',  # Remove container after completion
        '--gpus', 'all',  # Enable GPU support
        '--shm-size', '2g',  # Increase shared memory to avoid bus errors
    ]
    
    # Add volume mounts
    for volume in volumes:
        docker_cmd.extend(['-v', volume])
    
    # Add image name
    docker_cmd.append(image_name)
    
    print(f"\nStarting Docker container '{container_name}'...")
    print(f"Command: {' '.join(docker_cmd)}\n")
    
    # Prepare predictions file path
    predictions_file = os.path.join(output_dir, predictions_filename)
    
    # Clear old predictions file to ensure accurate monitoring
    if os.path.exists(predictions_file):
        os.remove(predictions_file)
        print(f"Cleared old predictions file: {predictions_file}")
    
    # Dictionary to store monitoring results
    monitor_results = {}
    
    # Start timing
    start_time = time.time()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(
        target=monitor_predictions_file,
        args=(predictions_file, expected_file_count, monitor_results, start_time),
        daemon=True
    )
    monitor_thread.start()
    
    try:
        # Run docker container
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Wait for monitoring thread to finish (with timeout)
        monitor_thread.join(timeout=10)
        
        print("\nDocker container completed successfully")
        print(f"\nContainer output:\n{result.stdout}")
        
        if result.stderr:
            print(f"\nContainer stderr:\n{result.stderr}")
        
        return {
            'success': True,
            'total_time': total_time,
            'time_to_last_prediction': monitor_results.get('time_to_last_prediction', None),
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        total_time = end_time - start_time
        
        # Wait for monitoring thread
        monitor_thread.join(timeout=5)
        
        print(f"Error running Docker container: {e}")
        print(f"\nStdout:\n{e.stdout}")
        print(f"\nStderr:\n{e.stderr}")
        
        return {
            'success': False,
            'total_time': total_time,
            'time_to_last_prediction': monitor_results.get('time_to_last_prediction', None),
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }


def main():
    parser = argparse.ArgumentParser(
        description='Run Docker-based model evaluation and measure inference time.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--image',
        type=str,
        required=True,
        help='Docker image name (e.g., mymodel:latest)'
    )
    parser.add_argument(
        '-t', '--test-data',
        type=str,
        default='./test_data',
        help='Directory containing test .tif files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./output',
        help='Directory for output predictions and results'
    )
    parser.add_argument(
        '-n', '--container-name',
        type=str,
        default=f'eval_container_{int(time.time())}',
        help='Name for the Docker container'
    )
    parser.add_argument(
        '--skip-full-eval',
        action='store_true',
        help='Skip full evaluation'
    )
    parser.add_argument(
        '--predictions-file',
        type=str,
        default='predictions.csv',
        help='Name of the predictions CSV file to monitor in output directory'
    )
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights',
        help='Directory containing model weights to mount'
    )
    
    args = parser.parse_args()
    
    # Check prerequisites
    print("=" * 70)
    print("DOCKER EVALUATION SETUP")
    print("=" * 70)
    
    if not check_docker_installed():
        sys.exit(1)
    
    if not check_docker_image_exists(args.image):
        print(f"\nTo build the Docker image, run:")
        print(f"  docker build -t {args.image} .")
        sys.exit(1)
    
    # Get test files
    test_files = get_test_files(args.test_data)
    print(f"\nFound {len(test_files)} test files")
    print(f"  First file: {os.path.basename(test_files[0])}")
    print(f"  Last file:  {os.path.basename(test_files[-1])}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    results = {
        'image_name': args.image,
        'num_test_files': len(test_files),
        'first_file': os.path.basename(test_files[0]),
        'last_file': os.path.basename(test_files[-1])
    }
    
    # Run full evaluation
    if not args.skip_full_eval:
        print("\n" + "=" * 70)
        print("RUNNING FULL EVALUATION")
        print("=" * 70)
        
        eval_result = run_docker_inference(
            args.image,
            args.container_name,
            args.test_data,
            args.output,
            len(test_files),  # Pass expected number of files
            weights_dir=args.weights,
            predictions_filename=args.predictions_file
        )
        
        results['full_evaluation'] = {
            'success': eval_result['success'],
            'total_time': eval_result['total_time'],
            'time_to_last_prediction': eval_result.get('time_to_last_prediction'),
            'avg_time_per_file': eval_result['total_time'] / len(test_files)
        }
        
        print(f"\nTotal container runtime: {eval_result['total_time']:.4f} seconds")
        
        if eval_result.get('time_to_last_prediction'):
            print(f"Time to last prediction: {eval_result['time_to_last_prediction']:.4f} seconds")
            print(f"Average time per file: {eval_result['time_to_last_prediction'] / len(test_files):.4f} seconds")
            print(f"Throughput: {len(test_files) / eval_result['time_to_last_prediction']:.2f} files/second")
        else:
            print(f"Warning: Could not detect last file prediction in CSV")
            print(f"Average time per file (based on total): {results['full_evaluation']['avg_time_per_file']:.4f} seconds")
            print(f"Throughput: {len(test_files) / eval_result['total_time']:.2f} files/second")
    
    # Save timing results
    results_file = os.path.join(args.output, 'timing_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Compute additional metrics (F1 score and model parameters)
    print("\n" + "=" * 70)
    print("COMPUTING ADDITIONAL METRICS")
    print("=" * 70)
    
    try:
        import pandas as pd
        from sklearn.metrics import f1_score
        
        # Compute F1 score
        predictions_csv = os.path.join(args.output, 'predictions.csv')
        if os.path.exists(predictions_csv):
            df = pd.read_csv(predictions_csv)
            df = df.dropna()
            y_true = df['actual_class'].values
            y_pred = df['predicted_class'].values
            weighted_f1 = f1_score(y_true, y_pred, average='weighted')
            print(f"Weighted F1 Score: {weighted_f1:.4f}")
            results['weighted_f1_score'] = weighted_f1
        else:
            print(f"Warning: Predictions file not found: {predictions_csv}")
            weighted_f1 = None
        
        # Count model parameters
        if os.path.exists(args.weights):
            weight_files = []
            for pattern in ['*.pth', '*.pt', '*.h5', '*.hdf5', '*.gguf']:
                weight_files.extend(glob.glob(os.path.join(args.weights, pattern)))

            if weight_files:
                # Get model file
                model_file = weight_files[0]
                model_name = os.path.basename(model_file)
                results['model_name'] = model_name

                total_params = None

                try:
                    if model_file.endswith(('.pth', '.pt')):
                        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)

                        if isinstance(checkpoint, dict):
                            if 'model_state_dict' in checkpoint:
                                state_dict = checkpoint['model_state_dict']
                            elif 'state_dict' in checkpoint:
                                state_dict = checkpoint['state_dict']
                            else:
                                state_dict = checkpoint
                        else:
                            state_dict = checkpoint

                        if isinstance(state_dict, dict):
                            total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
                        else:
                            total_params = sum(p.numel() for p in state_dict.parameters())

                    elif model_file.endswith(('.h5', '.hdf5')):
                        try:
                            import tensorflow as tf

                            keras_model = tf.keras.models.load_model(model_file, compile=False)
                            total_params = keras_model.count_params()
                        except ModuleNotFoundError:
                            print("Warning: TensorFlow not available to count parameters for .h5 model")
                        except Exception as e:
                            print(f"Warning: Could not load Keras model: {e}")

                    else:
                        print(f"Warning: Parameter counting not implemented for {model_name}")

                    if total_params is not None:
                        print(f"Model: {model_name}")
                        print(f"Model Parameters: {total_params:,}")
                        results['model_parameters'] = total_params
                    else:
                        results['model_parameters'] = 'N/A'
                except Exception as e:
                    print(f"Warning: Could not compute parameters for {model_name}: {e}")
                    results['model_parameters'] = 'Error'
            else:
                print(f"Warning: No model files found in {args.weights}")
                total_params = None
        else:
            print(f"Warning: Weights directory not found: {args.weights}")
            total_params = None
            
    except Exception as e:
        print(f"Warning: Could not compute additional metrics: {e}")
        weighted_f1 = None
        total_params = None
    
    # Save updated results with all metrics
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print consolidated summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    if eval_result and eval_result.get('time_to_last_prediction'):
        throughput = len(test_files) / eval_result['time_to_last_prediction']
    else:
        throughput = len(test_files) / eval_result['total_time'] if eval_result else None

    model_name = results.get('model_name', 'N/A')
    print(f"Model Name: {model_name}")
    print(f"Number of Test Images: {len(test_files)}")
    if throughput is not None:
        print(f"Throughput: {throughput:.2f} files/second")
    if weighted_f1 is not None:
        print(f"Weighted F1 Score: {weighted_f1:.4f}")

    model_params_value = results.get('model_parameters')
    if isinstance(model_params_value, (int, float)):
        print(f"Model Size: {int(model_params_value):,} parameters")
    elif isinstance(model_params_value, str):
        print(f"Model Size: {model_params_value}")

    print("=" * 70)
    print(f"Results saved to: {results_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()

