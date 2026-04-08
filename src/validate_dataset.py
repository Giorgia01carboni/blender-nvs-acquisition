"""
Dataset Validator for Gaussian Splatting.
Verifies image integrity and validates 4x4 extrinsic matrices.
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def is_valid_rotation_matrix(matrix: np.ndarray, tolerance: float = 1e-4) -> bool:
    """
    Checks if the top-left 3x3 matrix is a valid rotation matrix (SO(3)).
    Condition 1: R^T * R = I
    Condition 2: det(R) = 1
    """
    R = matrix[:3, :3]
    
    # Check orthogonality
    should_be_identity = np.dot(R.T, R)
    identity = np.eye(3)
    if not np.allclose(should_be_identity, identity, atol=tolerance):
        return False
        
    # Check determinant
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=tolerance):
        return False
        
    return True

def validate_image(image_path: Path) -> bool:
    """
    Checks if an image exists and contains actual visual data 
    (not just a solid color or pure transparency).
    """
    if not image_path.exists():
        logger.error(f"Missing file: {image_path.name}")
        return False

    try:
        with Image.open(image_path) as img:
            img_array = np.array(img)
            
            # Check if image is purely one color (variance == 0)
            # We check the RGB channels; ignoring alpha for the variance check
            rgb_variance = np.var(img_array[:, :, :3])
            
            if rgb_variance < 1.0:
                logger.error(f"Degenerate image (solid color detected): {image_path.name}")
                return False
                
    except Exception as e:
        logger.error(f"Corrupted image file {image_path.name}: {str(e)}")
        return False
        
    return True

def validate_dataset(dataset_dir: Path) -> bool:
    """Runs QA checks on a single dataset directory."""
    json_path = dataset_dir / "transforms.json"
    
    if not json_path.exists():
        logger.error(f"Missing transforms.json in {dataset_dir.name}")
        return False

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Corrupted transforms.json in {dataset_dir.name}")
        return False

    if "frames" not in data or len(data["frames"]) == 0:
        logger.error(f"No frames found in JSON for {dataset_dir.name}")
        return False

    valid_frames = 0
    for frame in data["frames"]:
        # 1. Image Check
        img_path = dataset_dir / frame["file_path"]
        if not validate_image(img_path):
            return False

        # 2. Matrix Check
        try:
            matrix = np.array(frame["transform_matrix"])
            if matrix.shape != (4, 4):
                logger.error(f"Invalid matrix shape {matrix.shape} in {dataset_dir.name}")
                return False
                
            if not is_valid_rotation_matrix(matrix):
                logger.error(f"Non-orthogonal rotation matrix in {dataset_dir.name}")
                return False
        except KeyError:
            logger.error(f"Missing transform_matrix in {dataset_dir.name}")
            return False

        valid_frames += 1

    logger.debug(f"{dataset_dir.name}: {valid_frames} frames validated.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate GSplat synthetic datasets.")
    parser.add_argument("--datasets_dir", type=Path, required=True, help="Directory containing the rendered dataset folders.")
    parser.add_argument("--quarantine_dir", type=Path, default=None, help="Optional: Move invalid datasets here.")
    
    args = parser.parse_args()

    if not args.datasets_dir.exists():
        logger.error(f"Directory not found: {args.datasets_dir}")
        return

    dataset_folders = [d for d in args.datasets_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(dataset_folders)} datasets to validate.")

    if args.quarantine_dir and not args.quarantine_dir.exists():
        args.quarantine_dir.mkdir(parents=True)

    passed = 0
    failed = 0

    for dataset in dataset_folders:
        is_valid = validate_dataset(dataset)
        
        if is_valid:
            passed += 1
        else:
            failed += 1
            if args.quarantine_dir:
                logger.warning(f"Quarantining {dataset.name} -> {args.quarantine_dir}")
                dataset.rename(args.quarantine_dir / dataset.name)

    logger.info("--- Validation Complete ---")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")

if __name__ == "__main__":
    main()