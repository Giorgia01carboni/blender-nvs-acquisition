import objaverse
import os
import shutil

TARGET_DIR = "/home/xrdev/Desktop/GPU-tests/gaussian-splatting/synthetic-acquisition-blender/raw_meshes"
os.makedirs(TARGET_DIR, exist_ok=True)

def fetch_humanoid_base(limit=4):
    print("Loading Objaverse LVIS annotations...")
    lvis_annotations = objaverse.load_lvis_annotations()
    
    person_uids = lvis_annotations.get('person', [])
    print(f"Identified {len(person_uids)} humanoid objects.")
    
    if not person_uids:
        return {}
        
    target_uids = person_uids[:limit]
    
    print(f"Downloading {len(target_uids)} objects into Hugging Face cache...")
    # Call without download_dir
    cached_objects = objaverse.load_objects(uids=target_uids)
    
    # Isolate files into the target directory
    final_objects = {}
    for uid, cached_path in cached_objects.items():
        # Preserve original file extension (typically .glb)
        _, ext = os.path.splitext(cached_path)
        target_path = os.path.join(TARGET_DIR, f"{uid}{ext}")
        
        # Copy from cache to pipeline directory
        shutil.copy2(cached_path, target_path)
        final_objects[uid] = target_path
        
    return final_objects

if __name__ == "__main__":
    downloaded_files = fetch_humanoid_base(limit=4)
    print(f"\nTotal copied to target directory: {len(downloaded_files)}")
    for uid, path in downloaded_files.items():
        print(f"Path: {path}")