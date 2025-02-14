import vtk
import os
import numpy as np
import argparse
import trimesh

def save_mesh_views(ply_path, output_dir):
    print(f"Loading mesh from: {ply_path}")
    
    # Create PLY reader
    reader = vtk.vtkPLYReader()
    reader.SetFileName(ply_path)
    reader.Update()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create renderer and window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    
    # Add the mesh to the scene
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer.AddActor(actor)
    
    # Set background color
    renderer.SetBackground(1, 1, 1)  # white background
    
    # Window to image filter
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    
    # PNG writer
    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(w2i.GetOutputPort())
    
    # Define camera views
    views = {
        "front": {"position": (0, 0, 2), "up": (0, 1, 0)},
        "side": {"position": (2, 0, 0), "up": (0, 1, 0)},
        "top": {"position": (0, 2, 0), "up": (0, 0, -1)},
        "iso": {"position": (1.5, 1.5, 1.5), "up": (0, 1, 0)}
    }
    
    # Set window size
    render_window.SetSize(1280, 720)
    
    # Reset camera to fit the data
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    
    # Render and save each view
    for view_name, params in views.items():
        # Set camera position
        camera.SetPosition(*params["position"])
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(*params["up"])
        
        # Render
        render_window.Render()
        
        # Update the window to image filter
        w2i.Modified()
        
        # Save image
        output_path = os.path.join(output_dir, f"{view_name}.png")
        writer.SetFileName(output_path)
        writer.Write()
        print(f"Saved {view_name} view to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View SAMPart3D segmentation results")
    parser.add_argument("--scale", type=str, default="0.0", 
                      help="Scale to view (0.0, 0.5, 1.0, 1.5, or 2.0)")
    args = parser.parse_args()
    
    # Construct paths
    ply_path = f"exp/sampart3d/knight/vis_pcd/last/mesh_{args.scale}.ply"
    output_dir = f"exp/sampart3d/knight/3d_views/scale_{args.scale}"
    
    if not os.path.exists(ply_path):
        print(f"Error: Could not find PLY file at {ply_path}")
        print("Available scales: 0.0, 0.5, 1.0, 1.5, 2.0")
        exit(1)
    
    save_mesh_views(ply_path, output_dir) 