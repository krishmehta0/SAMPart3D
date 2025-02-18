#!/usr/bin/env python
import os
import sys
import trimesh
import numpy as np
import argparse

def extract_part_meshes(mesh_path, results_dir, save_dir, scale="0.0"):
    """
    Extract individual part meshes from segmentation results
    
    Args:
        mesh_path: Path to the original mesh
        results_dir: Directory containing mesh_{scale}.npy segmentation results
        save_dir: Directory to save individual part meshes
        scale: Scale value to use (default "0.0")
    """
    # Load original mesh
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    # Load part labels
    labels = np.load(os.path.join(results_dir, f"mesh_{scale}.npy"))
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract each part as separate mesh
    unique_labels = np.unique(labels)
    for label in unique_labels:
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
        
        # Save part mesh
        save_path = os.path.join(save_dir, f"part_{label}.ply")
        part_mesh.export(save_path)
        
        print(f"Saved part {label} with {len(part_faces)} faces to {save_path}")

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