#!/usr/bin/env python
import os
import sys
import trimesh
import numpy as np
import argparse
import json

def create_colored_obb(mesh, color):
    """Create an OBB with specified color"""
    obb = mesh.bounding_box_oriented
    obb.visual.face_colors = color
    obb.visual.vertex_colors = color
    return obb

def extract_part_meshes(mesh_path, results_dir, save_dir, scale="0.0"):
    """
    Extract individual part meshes from segmentation results and compute their oriented bounding boxes
    
    Args:
        mesh_path: Path to the original mesh
        results_dir: Directory containing mesh_{scale}.npy segmentation results
        save_dir: Directory to save individual part meshes
        scale: Scale value to use (default "0.0")
    """
    # Load original mesh
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()  # Updated from deprecated dump() method
    
    # Load part labels
    labels = np.load(os.path.join(results_dir, f"mesh_{scale}.npy"))
    
    # Create scale-specific output directories
    scale_dir = os.path.join(save_dir, f"scale_{scale}")
    parts_dir = os.path.join(scale_dir, "parts")
    obb_dir = os.path.join(scale_dir, "obb")
    
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(obb_dir, exist_ok=True)
    
    # Dictionary to store OBB information for all parts
    part_obb_info = {}
    
    # List to store all part meshes for combined OBB
    part_meshes = []
    # List to store all OBBs for combined visualization
    all_obbs = []
    
    # Color scheme
    part_colors = [
        [255, 0, 0, 255],    # Red
        [0, 255, 0, 255],    # Green
        [0, 0, 255, 255],    # Blue
        [255, 255, 0, 255],  # Yellow
        [255, 0, 255, 255],  # Magenta
        [0, 255, 255, 255],  # Cyan
    ]
    combined_color = [128, 128, 128, 128]  # Semi-transparent gray
    
    # Extract each part as separate mesh
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        if label == -1:  # Skip unlabeled faces
            continue
            
        # Get faces for this part
        part_faces = mesh.faces[labels == label]
        
        # Create new mesh for this part
        part_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=part_faces,
            process=False  # Avoid processing that might alter the mesh
        )
        
        # Remove unused vertices
        part_mesh = part_mesh.copy()
        part_mesh.remove_unreferenced_vertices()
        
        # Add to list of part meshes
        part_meshes.append(part_mesh)
        
        # Compute colored oriented bounding box
        color = part_colors[i % len(part_colors)]  # Cycle through colors if more parts than colors
        obb = create_colored_obb(part_mesh, color)
        all_obbs.append(obb)
        
        # Extract OBB information
        obb_info = {
            "centroid": obb.centroid.tolist(),  # Changed from center to centroid
            "extents": obb.extents.tolist(),  # Length in each axis
            "transform": obb.transform.tolist(),  # 4x4 transformation matrix
            "bounds": obb.bounds.tolist(),  # Min and max bounds
            "volume": float(obb.volume),  # Convert to float for JSON serialization
            "area": float(obb.area)  # Convert to float for JSON serialization
        }
        
        # Save part mesh in scale-specific parts directory
        mesh_save_path = os.path.join(parts_dir, f"part_{label}.ply")
        part_mesh.export(mesh_save_path)
        
        # Save colored OBB mesh for visualization in scale-specific obb directory
        obb_save_path = os.path.join(obb_dir, f"part_{label}_obb.ply")
        obb.export(obb_save_path)
        
        # Store OBB info
        part_obb_info[str(label)] = obb_info
        
        print(f"Saved part {label} with {len(part_faces)} faces to {mesh_save_path}")
        print(f"Saved OBB to {obb_save_path}")
    
    # Create combined mesh from all parts
    if part_meshes:
        combined_mesh = trimesh.util.concatenate(part_meshes)
        combined_obb = create_colored_obb(combined_mesh, combined_color)
        all_obbs.append(combined_obb)
        
        # Save combined OBB mesh in scale-specific obb directory
        combined_obb_path = os.path.join(obb_dir, "combined_obb.ply")
        combined_obb.export(combined_obb_path)
        
        # Save all OBBs together in one scene in scale-specific obb directory
        scene = trimesh.Scene(all_obbs)
        scene_path = os.path.join(obb_dir, "all_obbs.ply")
        scene.export(scene_path)
        
        # Add combined OBB info to JSON
        part_obb_info["combined"] = {
            "centroid": combined_obb.centroid.tolist(),
            "extents": combined_obb.extents.tolist(),
            "transform": combined_obb.transform.tolist(),
            "bounds": combined_obb.bounds.tolist(),
            "volume": float(combined_obb.volume),
            "area": float(combined_obb.area)
        }
        print(f"Saved combined OBB to {combined_obb_path}")
        print(f"Saved combined visualization to {scene_path}")
    
    # Save OBB information to JSON in scale-specific directory
    obb_json_path = os.path.join(scale_dir, "part_obb_info.json")
    with open(obb_json_path, "w") as f:
        json.dump(part_obb_info, f, indent=2)
    print(f"Saved OBB information to {obb_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract individual part meshes from segmentation results")
    parser.add_argument("--mesh-path", required=True, help="Path to the original mesh file")
    parser.add_argument("--results-dir", required=True, help="Directory containing mesh_{scale}.npy segmentation results")
    parser.add_argument("--save-dir", required=True, help="Directory to save individual part meshes")
    parser.add_argument("--scale", default="0.0", help="Scale value to use (default: 0.0)")
    
    args = parser.parse_args()
    
    extract_part_meshes(
        mesh_path=args.mesh_path,
        results_dir=args.results_dir,
        save_dir=args.save_dir,
        scale=args.scale
    )

if __name__ == "__main__":
    main() 