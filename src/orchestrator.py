"""
Orchestrator for Synthetic Dataset Generation.
Spawns headless Blender processes to render 3D meshes into image datasets.

How to call the pipeline:
conda activate obj-render
python src/orchestrator.py \
  --input_dir ./raw_meshes \
  --output_dir ./datasets_video \
  --blender_bin ./blender \
  --worker_script ./src/blender_worker.py \
  --max_workers 4 \
  --mode video
  
If video mode is selected: 
ffmpeg -framerate 30 -i images/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4
"""

import argparse
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def process_mesh(mesh_path: Path, output_root: Path, blender_bin: Path, worker_script: Path,
                  mode: str, texture: str = None, num_pillars: int = 8, pillar_height: float = 3.0, cameras_per_pillar: int = 3) -> bool:
    """
    Executes a headless Blender instance to process a single mesh.
    """
    # Create a dedicated output directory for this specific mesh
    mesh_id = mesh_path.stem
    dataset_dir = output_root / mesh_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Blender headless command
    # Everything after '--' is passed to sys.argv in the Blender Python environment
    command = [
        str(blender_bin),
        "-b",                       # Run in background (headless)
        "-P", str(worker_script),   # Execute specific Python script
        "--",                       # Argument separator for script
        "--input", str(mesh_path),
        "--output", str(dataset_dir),
        "--mode", mode,
    ]

    if mode == 'pillar':
        command.extend(["--num_pillars", str(num_pillars), "--pillar_height", str(pillar_height), "--cameras_per_pillar", str(cameras_per_pillar)])
        if texture:
            command.extend(["--texture", str(texture)])

    try:
        logger.info(f"Starting rendering for: {mesh_id}")
        # capture_output prevents stdout/stderr from polluting the host terminal
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info(f"Successfully processed: {mesh_id}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to process {mesh_id}. Exit code: {e.returncode}")
        logger.error(f"Blender Error Output:\n{e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch process meshes for GSplat training.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing .glb/.obj files.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save generated datasets.")
    parser.add_argument("--blender_bin", type=Path, required=True, help="Path to the Blender executable.")
    parser.add_argument("--worker_script", type=Path, required=True, help="Path to blender_worker.py.")
    parser.add_argument("--max_workers", type=int, default=2, help="Number of concurrent Blender instances.")
    parser.add_argument("--mode", type=str, choices=['static', 'video', 'pillar'], default='static',
                    help="Acquisition mode: 'static' (Fibonacci), 'video' (Spiral), or 'pillar' (pillar-mounted cameras).")
    parser.add_argument("--texture", type=str, default=None, help="Path to floor texture image for pillar mode.")
    parser.add_argument("--num_pillars", type=int, default=8, help="Number of pillars (pillar mode only).")
    parser.add_argument("--pillar_height", type=float, default=3.0, help="Height of pillars (pillar mode only).")
    parser.add_argument("--cameras_per_pillar", type=int, default=3, help="Cameras mounted on each pillar.")
    
    args = parser.parse_args()

    # Path validation
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    if not args.blender_bin.exists():
        logger.error(f"Blender binary not found: {args.blender_bin}")
        return
    if not args.worker_script.exists():
        logger.error(f"Worker script not found: {args.worker_script}")
        return

    # Gather target meshes
    supported_extensions = {".glb", ".gltf", ".obj", ".blend"}
    meshes: List[Path] = [
        p for p in args.input_dir.iterdir() 
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]

    logger.info(f"Found {len(meshes)} meshes to process. Using {args.max_workers} workers.")

    # Concurrency execution

    successful_renders = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_mesh, mesh, args.output_dir, args.blender_bin, args.worker_script,
                            args.mode, args.texture, args.num_pillars, args.pillar_height, args.cameras_per_pillar): mesh
            for mesh in meshes
        }

        for future in as_completed(futures):
            success = future.result()
            if success:
                successful_renders += 1

    logger.info(f"Pipeline complete. Successfully rendered {successful_renders}/{len(meshes)} datasets.")

if __name__ == "__main__":
    main()