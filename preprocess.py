import numpy as np
import json
import cv2
import os
from pathlib import Path
import trimesh
import torch
from PIL import Image
import argparse
import glob
import Imath
import OpenEXR
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    DirectionalLights,
    BlendParams
)

def save_exr(filename, depth_arr):
    """Save depth array as EXR file"""
    # Ensure depth array is 2D and float32
    if len(depth_arr.shape) != 2:
        raise ValueError("Depth array must be 2D")
    depth_arr = depth_arr.astype(np.float32)
    
    # Prepare header with correct dimensions
    header = OpenEXR.Header(depth_arr.shape[1], depth_arr.shape[0])
    header['channels'] = {'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    
    # Ensure the array is in Fortran order (column-major) for OpenEXR
    depth_arr = np.asfortranarray(depth_arr)
    
    # Create file and write data
    exr = OpenEXR.OutputFile(filename, header)
    exr.writePixels({'Y': depth_arr.tobytes()})
    exr.close()

class MultiViewRenderer:
    def __init__(self, num_views=16, image_size=(800, 800)):
        self.num_views = num_views
        self.image_width, self.image_height = image_size
        self.camera_angle_x = np.pi / 3  # 60 degrees FOV
        # Force CPU usage
        self.device = torch.device("cpu")
        
    def setup_renderer(self):
        """Setup PyTorch3D renderer"""
        # Camera settings with forced CPU
        cameras = FoVPerspectiveCameras(
            fov=np.degrees(self.camera_angle_x),
            aspect_ratio=self.image_width/self.image_height,
            device=self.device
        )
        
        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=(self.image_height, self.image_width),
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        # Lighting with forced CPU
        lights = DirectionalLights(
            device=self.device,
            direction=[[0.0, 0.0, -1.0]],
            ambient_color=((1.0, 1.0, 1.0),),
            diffuse_color=((0.0, 0.0, 0.0),),
            specular_color=((0.0, 0.0, 0.0),),
        )
        
        # Create renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(background_color=(1.0, 1.0, 1.0))
            )
        )
        
        return renderer, cameras

    def setup_camera_poses(self):
        """Generate camera positions in a circle around the object"""
        # Generate evenly spaced viewing angles
        angles = torch.linspace(0, 2 * np.pi, self.num_views)
        
        # Camera distance and elevation
        distance = 2.0
        elevation = 15.0  # degrees
        
        # Generate camera positions using pytorch3d's look_at_view_transform
        R, T = look_at_view_transform(
            dist=distance,
            elev=elevation * torch.ones(self.num_views),
            azim=angles * 180 / np.pi
        )
        
        return R, T

    def render_views(self, mesh_path, output_dir):
        """Render multiple views of the mesh"""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load mesh with trimesh first
        print(f"Loading mesh file: {mesh_path}")
        tri_mesh = trimesh.load(mesh_path)
        if isinstance(tri_mesh, trimesh.Scene):
            print("Converting Scene to Mesh...")
            geometries = list(tri_mesh.geometry.values())
            if len(geometries) == 0:
                raise ValueError("No geometries found in the scene")
            if len(geometries) > 1:
                tri_mesh = trimesh.util.concatenate(geometries)
            else:
                tri_mesh = geometries[0]
        
        print(f"Mesh stats - Vertices: {len(tri_mesh.vertices)}, Faces: {len(tri_mesh.faces)}")
        
        # Calculate scaling factor and center offset before normalization
        mesh_center_offset = tri_mesh.vertices.mean(axis=0)
        scale = np.max(np.abs(tri_mesh.vertices - mesh_center_offset))
        
        # Center and scale the mesh
        tri_mesh.vertices -= mesh_center_offset
        tri_mesh.vertices /= scale
        
        # Convert to PyTorch3D format
        verts = torch.tensor(tri_mesh.vertices, device=self.device, dtype=torch.float32)
        faces = torch.tensor(tri_mesh.faces, device=self.device, dtype=torch.int64)
        
        # Create a white texture for all vertices
        verts_rgb = torch.ones_like(verts)[None]
        
        # Setup renderer and camera poses
        renderer, cameras = self.setup_renderer()
        R, T = self.setup_camera_poses()
        R, T = R.to(self.device), T.to(self.device)
        
        # Create a Meshes object
        textures = TexturesVertex(verts_features=verts_rgb)
        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )
        
        # Prepare metadata
        transforms = []  # Initialize the transforms list
        metadata = {
            'camera_angle_x': self.camera_angle_x,
            'transforms': transforms,
            'scaling_factor': float(scale),
            'mesh_offset': mesh_center_offset.tolist()
        }
        
        # Save camera transforms for metadata
        for i in range(self.num_views):
            transform = torch.eye(4)
            transform[:3, :3] = R[i]
            transform[:3, 3] = T[i]
            transforms.append(transform.tolist())
        
        # Save metadata
        with open(os.path.join(output_dir, "meta.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Render views
        for i in range(self.num_views):
            print(f"Rendering view {i+1}/{self.num_views}")
            
            # Render
            images = renderer(mesh, cameras=cameras, R=R[i:i+1], T=T[i:i+1])
            
            # Get color image
            image = images[0, ..., :3].cpu().numpy()  # RGB
            # Ensure image is in 0-255 range
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            # Save as RGB
            Image.fromarray(image).save(os.path.join(output_dir, f"render_{i:04d}.webp"), quality=100)
            
            # Get depth image and ensure it's 2D
            depth = images[0, ..., 3].cpu().numpy()  # Depth
            depth = depth.reshape(self.image_height, self.image_width)  # Ensure 2D shape
            
            # Save depth as EXR
            header = OpenEXR.Header(self.image_width, self.image_height)
            header['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
            
            # Ensure depth is in correct format and range
            depth = depth.astype(np.float32)
            depth = np.clip(depth, 0, np.inf)  # Ensure no negative values
            
            # Create file and write data
            exr = OpenEXR.OutputFile(os.path.join(output_dir, f"depth_{i:04d}.exr"), header)
            exr.writePixels({'Z': depth.tobytes()})
            exr.close()
            
            # For debugging
            self.logger.info(f"Frame {i} - Depth range: {depth.min():.3f} to {depth.max():.3f}")
            self.logger.info(f"Frame {i} - Image range: {image.min()} to {image.max()}")
            
            # Clean up any range files if they exist from previous runs
            range_file = os.path.join(output_dir, f"depth_{i:04d}_range.json")
            if os.path.exists(range_file):
                os.remove(range_file)

def preprocess_mesh(mesh_path, output_root, object_uid):
    """
    Preprocess a single mesh file
    Args:
        mesh_path: Path to input mesh file
        output_root: Root directory for outputs
        object_uid: Unique identifier for the object
    """
    # Create output directory for this object
    output_dir = os.path.join(output_root, object_uid)
    
    # Initialize renderer
    renderer = MultiViewRenderer()
    
    # Render views
    renderer.render_views(mesh_path, output_dir)
    
    print(f"Processed {object_uid}: Output saved to {output_dir}")

def list_available_meshes(mesh_root):
    """List all available mesh files in the mesh root directory"""
    mesh_files = glob.glob(os.path.join(mesh_root, "*.glb"))
    mesh_files.extend(glob.glob(os.path.join(mesh_root, "*.obj")))
    mesh_files.extend(glob.glob(os.path.join(mesh_root, "*.ply")))
    
    if not mesh_files:
        print(f"No mesh files found in {mesh_root}")
        return []
    
    print("\nAvailable mesh files:")
    for i, mesh_file in enumerate(mesh_files):
        filename = os.path.basename(mesh_file)
        print(f"{i+1}. {filename}")
    
    return mesh_files

def main():
    parser = argparse.ArgumentParser(description='Render multiple views of 3D meshes for SAMPart3D training')
    parser.add_argument('--mesh_root', type=str, default='mesh_root',
                        help='Directory containing mesh files')
    parser.add_argument('--data_root', type=str, default='data_root',
                        help='Directory where rendered views will be saved')
    parser.add_argument('--object', type=str, default=None,
                        help='Specific object to process (filename without extension). If not provided, will list available objects.')
    parser.add_argument('--list', action='store_true',
                        help='List available mesh files and exit')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.mesh_root, exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)
    
    # Get list of available meshes
    mesh_files = list_available_meshes(args.mesh_root)
    
    if args.list:
        return
        
    if not mesh_files:
        print("No mesh files found. Please add some mesh files to the mesh_root directory.")
        return
        
    if args.object:
        # Process specific object
        matching_files = [f for f in mesh_files if os.path.splitext(os.path.basename(f))[0] == args.object]
        if not matching_files:
            print(f"No mesh file found for object '{args.object}'")
            print("Available objects:")
            for f in mesh_files:
                print(f"  {os.path.splitext(os.path.basename(f))[0]}")
            return
            
        mesh_path = matching_files[0]
        object_uid = args.object
        preprocess_mesh(mesh_path, args.data_root, object_uid)
    else:
        # Interactive object selection
        while True:
            try:
                choice = input("\nEnter the number of the mesh to process (or 'q' to quit): ")
                if choice.lower() == 'q':
                    break
                    
                idx = int(choice) - 1
                if 0 <= idx < len(mesh_files):
                    mesh_path = mesh_files[idx]
                    object_uid = os.path.splitext(os.path.basename(mesh_path))[0]
                    preprocess_mesh(mesh_path, args.data_root, object_uid)
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number or 'q' to quit.")

if __name__ == "__main__":
    main()