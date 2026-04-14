import bpy
import math
import sys
import argparse
import json
import os
from pathlib import Path
from mathutils import Vector, Matrix

def parse_args():
    if "--" not in sys.argv:
        return None
    argv = sys.argv[sys.argv.index("--") + 1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_cameras", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=800)
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--mode", type=str, choices=['static', 'video', 'pillar'], default='static')
    parser.add_argument("--texture", type=str, default=None)
    parser.add_argument("--num_pillars", type=int, default=8)
    parser.add_argument("--pillar_height", type=float, default=3.0)
    parser.add_argument("--cameras_per_pillar", type=int, default=3)

    return parser.parse_args(argv)

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.images]:
        for item in collection:
            collection.remove(item)

def import_mesh(filepath: str):
    ext = Path(filepath).suffix.lower()
    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == '.blend':
        bpy.ops.wm.append(filepath=filepath, directory=str(Path(filepath).parent))
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def normalize_geometry(target_size: float = 1.0, floor_z: float = None):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    if not meshes: return

    min_coords = Vector((float("inf"), float("inf"), float("inf")))
    max_coords = Vector((float("-inf"), float("-inf"), float("-inf")))
    valid_bounds = False

    for obj in meshes:
        eval_obj = obj.evaluated_get(depsgraph)
        try: eval_mesh = eval_obj.to_mesh()
        except RuntimeError: continue
            
        if not eval_mesh.vertices:
            eval_obj.to_mesh_clear()
            continue
            
        valid_bounds = True
        world_matrix = obj.matrix_world
        
        for v in eval_mesh.vertices:
            world_co = world_matrix @ v.co
            for i in range(3):
                if world_co[i] < min_coords[i]: min_coords[i] = world_co[i]
                if world_co[i] > max_coords[i]: max_coords[i] = world_co[i]
        eval_obj.to_mesh_clear()

    if not valid_bounds: return

    centroid = (min_coords + max_coords) / 2.0
    dimensions = max_coords - min_coords
    max_dimension = max(dimensions)
    if max_dimension == 0: return

    scale_factor = target_size / max_dimension
    T = Matrix.Translation(-centroid)
    S = Matrix.Scale(scale_factor, 4)

    for mesh in meshes:
        mesh.matrix_world = S @ T @ mesh.matrix_world

    if floor_z is not None:
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        min_z = float("inf")
        for obj in meshes:
            eval_obj = obj.evaluated_get(depsgraph)
            try: eval_mesh = eval_obj.to_mesh()
            except RuntimeError: continue
            for v in eval_mesh.vertices:
                wz = (obj.matrix_world @ v.co).z
                if wz < min_z: min_z = wz
            eval_obj.to_mesh_clear()

        if min_z != float("inf"):
            lift = Matrix.Translation(Vector((0, 0, floor_z - min_z)))
            for mesh in meshes:
                mesh.matrix_world = lift @ mesh.matrix_world

    bpy.context.view_layer.update()
    
def setup_rendering(resolution: int, transparent: bool = True):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.use_denoising = False

    prefs = bpy.context.preferences.addons['cycles'].preferences
    for dev_type in ['OPTIX', 'CUDA', 'HIP']:
        try:
            prefs.compute_device_type = dev_type
            break
        except TypeError: pass

    prefs.get_devices()
    valid_devices = [d for d in prefs.devices if d.type != 'CPU']
    for device in valid_devices:
        device.use = True

    scene.cycles.device = 'GPU' if valid_devices else 'CPU'
    scene.cycles.samples = 128
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = transparent

    world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg_node.inputs[1].default_value = 1.5

def get_fibonacci_hemisphere(num_points: int, radius: float):
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
    direction = target_location - camera_location
    quat = direction.to_track_quat('-Z', 'Y')
    mat = quat.to_matrix().to_4x4()
    mat.translation = camera_location
    return mat

def create_floor_with_texture(texture_path: str, size: float = 10.0):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, -0.5))
    floor = bpy.context.active_object
    floor.name = "Floor"

    mat = bpy.data.materials.new("FloorMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output_node = nodes.new('ShaderNodeOutputMaterial')
    bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    tex_node = nodes.new('ShaderNodeTexImage')
    mapping_node = nodes.new('ShaderNodeMapping')
    coord_node = nodes.new('ShaderNodeTexCoord')

    tex_node.image = bpy.data.images.load(texture_path)
    mapping_node.inputs['Scale'].default_value = (4.0, 4.0, 1.0)

    links.new(coord_node.outputs['UV'], mapping_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
    links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
    floor.data.materials.append(mat)

def create_uniform_pillars(num_pillars: int, radius: float, pillar_height: float, pillar_radius: float = 0.08):
    floor_z = -0.5
    pillars = []
    for i in range(num_pillars):
        angle = 2 * math.pi * i / num_pillars
        x = math.cos(angle) * radius
        y = math.sin(angle) * radius
        z = floor_z + (pillar_height / 2.0)

        bpy.ops.mesh.primitive_cylinder_add(radius=pillar_radius, depth=pillar_height, location=(x, y, z))
        pillar = bpy.context.active_object
        pillar.name = f"Pillar_{i:02d}"

        mat = bpy.data.materials.new(f"PillarMat_{i:02d}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf: bsdf.inputs['Base Color'].default_value = (0.6, 0.6, 0.6, 1.0)
        pillar.data.materials.append(mat)
        pillars.append(pillar)
    return pillars

def get_multi_camera_positions(num_pillars: int, radius: float, pillar_height: float, cameras_per_pillar: int, pillar_radius: float = 0.08):
    floor_z = -0.5
    positions = []
    
    # Dynamically calculate elevation fractions
    if cameras_per_pillar == 1:
        elevation_fractions = [0.5] # Middle of the pillar
    else:
        max_frac = 0.9
        min_frac = 0.15
        step = (max_frac - min_frac) / (cameras_per_pillar - 1)
        elevation_fractions = [max_frac - (i * step) for i in range(cameras_per_pillar)]
        
    for i in range(num_pillars):
        angle = 2 * math.pi * i / num_pillars
        cam_radius = radius - (pillar_radius * 1.5)
        x = math.cos(angle) * cam_radius
        y = math.sin(angle) * cam_radius
        for frac in elevation_fractions:
            z = floor_z + (pillar_height * frac)
            positions.append(Vector((x, y, z)))
            
    return positions

def render_pillar_dataset(output_dir: str, num_pillars: int, radius: float, cameras_per_pillar: int, pillar_height: float, texture_path: str = None):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    if texture_path: create_floor_with_texture(texture_path, size=radius * 3)
    create_uniform_pillars(num_pillars, radius, pillar_height)

    cam_data = bpy.data.cameras.new("PillarCam")
    cam_data.lens = 35
    cam_data.sensor_width = 36
    cam_obj = bpy.data.objects.new("PillarCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    positions = get_multi_camera_positions(num_pillars, radius, pillar_height, cameras_per_pillar)
    frames_json = []
    origin = Vector((0.0, 0.0, 0.0))

    for i, pos in enumerate(positions):
        cam_obj.matrix_world = get_look_at_matrix(pos, origin)
        bpy.context.view_layer.update()

        img_filename = f"r_{i:04d}.png"
        bpy.context.scene.render.filepath = os.path.join(images_dir, img_filename)

        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try: bpy.ops.render.render(write_still=True)
            finally: sys.stdout = old_stdout

        matrix = cam_obj.matrix_world
        frames_json.append({
            "file_path": f"images/{img_filename}",
            "transform_matrix": [list(matrix[row]) for row in range(4)]
        })

    with open(os.path.join(output_dir, "transforms.json"), "w") as f:
        json.dump({"camera_angle_x": cam_data.angle_x, "frames": frames_json}, f, indent=4)

def get_spiral_trajectory(num_frames: int, radius: float, loops: int = 5):
    coords = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        z = t
        r_at_z = math.sqrt(1 - z**2)
        theta = 2 * math.pi * loops * t
        coords.append(Vector((math.cos(theta) * r_at_z * radius, math.sin(theta) * r_at_z * radius, z * radius)))
    return coords

def render_video_dataset(output_dir: str, num_frames: int, radius: float):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    cam_data = bpy.data.cameras.new("VideoCam")
    cam_obj = bpy.data.objects.new("VideoCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    positions = get_spiral_trajectory(num_frames, radius, loops=5)
    frames_json = []
    origin = Vector((0.0, 0.0, 0.0))

    for i, pos in enumerate(positions):
        cam_obj.matrix_world = get_look_at_matrix(pos, origin)
        bpy.context.view_layer.update()
        img_filename = f"frame_{i:04d}.png"
        bpy.context.scene.render.filepath = os.path.join(images_dir, img_filename)
        bpy.ops.render.render(write_still=True)

        matrix = cam_obj.matrix_world
        frames_json.append({
            "file_path": f"images/{img_filename}",
            "transform_matrix": [list(matrix[row]) for row in range(4)]
        })

    with open(os.path.join(output_dir, "transforms.json"), "w") as f:
        json.dump({"camera_angle_x": cam_data.angle_x, "frames": frames_json}, f, indent=4)

def render_dataset(output_dir: str, num_cameras: int, radius: float):
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
        cam_obj.matrix_world = get_look_at_matrix(pos, origin)
        bpy.context.view_layer.update()
        img_filename = f"r_{i}.png"
        bpy.context.scene.render.filepath = os.path.join(output_dir, "images", img_filename)
        
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try: bpy.ops.render.render(write_still=True)
            finally: sys.stdout = old_stdout

        matrix = cam_obj.matrix_world
        frames_json.append({
            "file_path": f"images/{img_filename}",
            "transform_matrix": [list(matrix[row]) for row in range(4)]
        })

    with open(os.path.join(output_dir, "transforms.json"), "w") as f:
        json.dump({"camera_angle_x": cam_data.angle_x, "frames": frames_json}, f, indent=4)

def main():
    args = parse_args()
    if not args: return

    try:
        clean_scene()
        import_mesh(args.input)

        # -------------------------------------------------------------
        # USER CONFIGURATION: SCALE MULTIPLIER
        # Adjust this variable to change the object's normalized size 
        # as a percentage of the camera array's radius.
        scale_multiplier = 0.6 
        # -------------------------------------------------------------

        if args.mode == 'pillar':
            normalize_geometry(target_size=args.radius * scale_multiplier, floor_z=-0.5)
        else:
            normalize_geometry(target_size=1.0) 

        transparent = args.mode != 'pillar'
        setup_rendering(args.resolution, transparent=transparent)
        os.makedirs(os.path.join(args.output, "images"), exist_ok=True)

        if args.mode == 'static':
            render_dataset(args.output, args.num_cameras, args.radius)
        elif args.mode == 'video':
            render_video_dataset(args.output, args.num_cameras, args.radius)
        elif args.mode == 'pillar':
            render_pillar_dataset(args.output, args.num_pillars, args.radius, args.cameras_per_pillar, args.pillar_height, args.texture)
        
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()