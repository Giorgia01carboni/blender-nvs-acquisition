import bpy
import math
import sys
import argparse
import json
import os
from pathlib import Path
from mathutils import Vector, Matrix

def parse_args():
    """Extracts arguments passed after the '--' delimiter in the Blender CLI."""
    if "--" not in sys.argv:
        return None
    argv = sys.argv[sys.argv.index("--") + 1:]
    
    parser = argparse.ArgumentParser(description="Blender headless rendering worker.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input mesh.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output dataset directory.")
    parser.add_argument("--num_cameras", type=int, default=100, help="Number of views to render.")
    parser.add_argument("--resolution", type=int, default=800, help="Resolution of the rendered images.")
    parser.add_argument("--radius", type=float, default=2.5, help="Radius of the camera hemisphere.")
    parser.add_argument("--mode", type=str, choices=['static', 'video'], default='static', help="Acquisition mode.")
    
    return parser.parse_args(argv)

def clean_scene():
    """Removes all default objects, meshes, and cameras from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.images]:
        for item in collection:
            collection.remove(item)

def import_mesh(filepath: str):
    """Imports a mesh based on its file extension."""
    ext = Path(filepath).suffix.lower()
    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == '.blend':
        bpy.ops.wm.append(filepath=filepath, directory=str(Path(filepath).parent))
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def normalize_geometry():
    """
    Centers and scales meshes using true evaluated vertex bounds.
    This guarantees mathematically perfect bounding boxes even for rigged/posed humans.
    """
    # 1. Get the current evaluation state (accounts for rigs and modifiers)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    if not meshes:
        return

    min_coords = Vector((float("inf"), float("inf"), float("inf")))
    max_coords = Vector((float("-inf"), float("-inf"), float("-inf")))

    valid_bounds = False

    for obj in meshes:
        # 2. Get the evaluated object with all armatures/deformations applied
        eval_obj = obj.evaluated_get(depsgraph)
        
        try:
            # 3. Bake the evaluated object into a temporary raw mesh
            eval_mesh = eval_obj.to_mesh()
        except RuntimeError:
            continue
            
        if not eval_mesh.vertices:
            eval_obj.to_mesh_clear()
            continue
            
        valid_bounds = True
        world_matrix = obj.matrix_world
        
        # 4. Calculate bounds directly from true world-space vertex positions
        for v in eval_mesh.vertices:
            world_co = world_matrix @ v.co
            for i in range(3):
                if world_co[i] < min_coords[i]: min_coords[i] = world_co[i]
                if world_co[i] > max_coords[i]: max_coords[i] = world_co[i]
                
        # Clear memory immediately to prevent RAM leaks during batch processing
        eval_obj.to_mesh_clear()

    if not valid_bounds:
        return

    # 5. Compute Centroid and Scale Factor
    centroid = (min_coords + max_coords) / 2.0
    dimensions = max_coords - min_coords
    max_dimension = max(dimensions)

    if max_dimension == 0:
        return

    scale_factor = 1.0 / max_dimension

    # 6. Apply matrix transformation
    T = Matrix.Translation(-centroid)
    S = Matrix.Scale(scale_factor, 4)
    
    for mesh in meshes:
        mesh.matrix_world = S @ T @ mesh.matrix_world
        
    # Force view layer update so the camera math in the next step sees the new positions
    bpy.context.view_layer.update()
    
def setup_rendering(resolution: int):
    """Configures Cycles for headless GPU rendering and disables denoising."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.use_denoising = False
    
    # 1. Hardware-Agnostic GPU Initialization
    prefs = bpy.context.preferences.addons['cycles'].preferences
    
    # Define preference order: OPTIX (RTX), CUDA (NVIDIA), HIP (AMD)
    preferred_devices = ['OPTIX', 'CUDA', 'HIP']
    device_set = False
    
    for dev_type in preferred_devices:
        try:
            prefs.compute_device_type = dev_type
            device_set = True
            break
        except TypeError:
            pass # Device type not available in this build/hardware
            
    if not device_set:
        print("WARNING: No compatible GPU compute API found. Defaulting to CPU.")
        prefs.compute_device_type = 'NONE'

    # 2. Refresh and Enable Devices
    prefs.get_devices()
    valid_devices = [d for d in prefs.devices if d.type != 'CPU']
    
    for device in valid_devices:
        device.use = True
        print(f"Enabled Render Device: {device.name}")
            
    scene.cycles.device = 'GPU' if valid_devices else 'CPU'
    scene.cycles.samples = 128  
    
    # 3. Output Configuration
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True

    # 4. Uniform Lighting Setup
    world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg_node.inputs[1].default_value = 1.5

def get_fibonacci_hemisphere(num_points: int, radius: float):
    """Calculates uniform spherical coordinates."""
    coords = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(num_points):
        z = 1 - (i / float(num_points - 1)) 
        radius_at_z = math.sqrt(1 - z * z)
        theta = phi * i
        x = math.cos(theta) * radius_at_z
        y = math.sin(theta) * radius_at_z
        coords.append(Vector((x * radius, y * radius, z * radius)))
    return coords

def get_look_at_matrix(camera_location: Vector, target_location: Vector) -> Matrix:
    """Calculates exact camera rotation to look at target, bypassing constraints."""
    direction = target_location - camera_location
    # Blender cameras look down the -Z axis, with Y pointing up
    quat = direction.to_track_quat('-Z', 'Y')
    mat = quat.to_matrix().to_4x4()
    mat.translation = camera_location
    return mat

# Video option
def get_spiral_trajectory(num_frames: int, radius: float, loops: int = 5):
    """Calculates continuous spiral coordinates for video rendering."""
    coords = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        z = t  # Hemisphere: 0 to 1
        r_at_z = math.sqrt(1 - z**2)
        theta = 2 * math.pi * loops * t
        
        x = math.cos(theta) * r_at_z
        y = math.sin(theta) * r_at_z
        coords.append(Vector((x * radius, y * radius, z * radius)))
    return coords

def render_video_dataset(output_dir: str, num_frames: int, radius: float):
    """Renders a continuous spiral trajectory and exports frame metadata."""
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    cam_data = bpy.data.cameras.new("VideoCam")
    cam_obj = bpy.data.objects.new("VideoCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # 5 loops provides significant parallax for gsplat
    positions = get_spiral_trajectory(num_frames, radius, loops=5)
    frames_json = []
    origin = Vector((0.0, 0.0, 0.0))

    for i, pos in enumerate(positions):
        cam_obj.matrix_world = get_look_at_matrix(pos, origin)
        bpy.context.view_layer.update()
        
        img_filename = f"frame_{i:04d}.png" # Padding for ffmpeg
        bpy.context.scene.render.filepath = os.path.join(images_dir, img_filename)
        
        # Render frame
        bpy.ops.render.render(write_still=True)

        # Record extrinsic
        matrix = cam_obj.matrix_world
        frames_json.append({
            "file_path": f"images/{img_filename}",
            "transform_matrix": [list(matrix[row]) for row in range(4)]
        })

    # Export JSON so gsplat can still use these frames
    with open(os.path.join(output_dir, "transforms.json"), "w") as f:
        json.dump({"camera_angle_x": cam_data.angle_x, "frames": frames_json}, f, indent=4)

def render_dataset(output_dir: str, num_cameras: int, radius: float):
    """Iterates through camera positions, renders, and extracts extrinsics."""
    os.makedirs(output_dir, exist_ok=True)
    
    cam_data = bpy.data.cameras.new("RenderCam")
    cam_data.lens = 35
    cam_data.sensor_width = 36
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    positions = get_fibonacci_hemisphere(num_cameras, radius)
    frames_json = []
    origin = Vector((0.0, 0.0, 0.0))

    for i, pos in enumerate(positions):
        # 1. Apply mathematically calculated matrix
        cam_obj.matrix_world = get_look_at_matrix(pos, origin)
        bpy.context.view_layer.update()
        
        img_filename = f"r_{i}.png"
        bpy.context.scene.render.filepath = os.path.join(output_dir, "images", img_filename)
        
        # 2. Render silently
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                bpy.ops.render.render(write_still=True)
            finally:
                sys.stdout = old_stdout

        # 3. Extract Extrinsic Matrix directly
        matrix = cam_obj.matrix_world
        frames_json.append({
            "file_path": f"images/{img_filename}",
            "transform_matrix": [list(matrix[row]) for row in range(4)]
        })

    dataset_metadata = {
        "camera_angle_x": cam_data.angle_x,
        "frames": frames_json
    }

    with open(os.path.join(output_dir, "transforms.json"), "w") as f:
        json.dump(dataset_metadata, f, indent=4)

def main():
    args = parse_args()
    if not args:
        return

    try:
        clean_scene()
        import_mesh(args.input)
        normalize_geometry()
        setup_rendering(args.resolution)
        
        # Create images subdirectory within the output folder
        images_dir = os.path.join(args.output, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        if args.mode == 'static':
            render_dataset(args.output, args.num_cameras, args.radius)
        else:
            render_video_dataset(args.output, args.num_cameras, args.radius)
        
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()