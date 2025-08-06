import torch
import trimesh
import numpy as np
from gaussian_splatting.datasets.colmap import Dataset, Parser
from gaussian_splatting.utils import set_random_seed
import argparse
import os
from tqdm import tqdm
import torch.nn.functional as F
from typing import Tuple, Dict, Any

# ModernGL imports
import moderngl
import pyrr
from PIL import Image


class MeshDistiller:
    """Engine for distilling 2D features onto 3D mesh triangles."""

    def __init__(self, args) -> None:
        set_random_seed(42)
        
        self.args = args
        self.device = f"cuda"
        
        # Debug flag for normal visualization
        self.debug_normals = False  # Set to True to render normals instead of face IDs
        
        # Load mesh
        mesh_path = args.mesh_path
        self._load_mesh(mesh_path)

        print(f"Loaded mesh: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        
        # Initialize ModernGL context
        self.ctx = moderngl.create_context(standalone=True)
        self._setup_shaders()
        self._prepare_mesh_data()
        
        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=args.data_dir,
            factor=args.data_factor,
            test_every=args.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            load_features=True,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1
        print("Scene scale:", self.scene_scale)

    def _load_mesh(self, mesh_path):
        loaded_mesh = trimesh.load(mesh_path, process=False)
        if isinstance(loaded_mesh, trimesh.Scene):
            # Extract the first mesh from the scene
            if len(loaded_mesh.geometry) == 0:
                raise ValueError("No geometry found in the scene")
            # Get the first mesh (or combine all meshes)
            meshes = [geom for geom in loaded_mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            if len(meshes) == 0:
                raise ValueError("No triangle meshes found in the scene")
            elif len(meshes) == 1:
                self.mesh = meshes[0]
            else:
                # Combine multiple meshes into one
                print(f"Found {len(meshes)} meshes in scene, combining them...")
                self.mesh = trimesh.util.concatenate(meshes)
        elif isinstance(loaded_mesh, trimesh.Trimesh):
            self.mesh = loaded_mesh
        else:
            raise ValueError(f"Unsupported mesh type: {type(loaded_mesh)}")

    def _setup_shaders(self):
        """Setup vertex and fragment shaders for both face ID and normal rendering."""
        # Face ID rendering shaders
        face_id_vertex_shader = '''
        #version 330 core
        
        in vec3 in_position;
        //flat out int face_id;
        
        uniform mat4 mvp;
        
        
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
            //face_id = base_face_id + (gl_VertexID / 3);
        }
        '''
        
        face_id_fragment_shader = '''
        #version 330 core        
        //flat in int face_id;
        out vec4 color;
        uniform int base_face_id;
        
        void main() {
            // Encode face_id into RGB channels for more range
            // This allows up to 16M faces (24-bit encoding)
            int face_id = gl_PrimitiveID + base_face_id;
            int r = (face_id >> 16) & 0xFF;
            int g = (face_id >> 8) & 0xFF;
            int b = face_id & 0xFF;
            color = vec4(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1.0);
        }
        '''
        
        # Normal visualization shaders
        normal_vertex_shader = '''
        #version 330 core
        
        in vec3 in_position;
        out vec3 world_pos;
        
        uniform mat4 mvp;
        
        void main() {
            vec4 world_position = vec4(in_position, 1.0);
            world_pos = world_position.xyz;
            gl_Position = mvp * vec4(in_position, 1.0);
        }
        '''
        
        normal_fragment_shader = '''
        #version 330 core
        
        in vec3 world_pos;
        out vec4 color;
        
        void main() {
            // Calculate surface normal using derivatives
            vec3 normal = normalize(cross(dFdx(world_pos), dFdy(world_pos)));
            
            // Convert normal from [-1,1] to [0,1] range for color visualization
            // Red = X component, Green = Y component, Blue = Z component
            vec3 normal_color = normal * 0.5 + 0.5;
            
            // Output normal as RGB color
            color = vec4(normal_color, 1.0);
        }
        '''
        
        # Create both programs
        self.face_id_program = self.ctx.program(
            vertex_shader=face_id_vertex_shader,
            fragment_shader=face_id_fragment_shader
        )
        
        self.normal_program = self.ctx.program(
            vertex_shader=normal_vertex_shader,
            fragment_shader=normal_fragment_shader
        )
        
        # Set the active program based on debug flag
        self.program = self.normal_program if self.debug_normals else self.face_id_program

    def _prepare_mesh_data(self):
        """Prepare mesh data for ModernGL rendering."""
        vertices = self.mesh.vertices.astype(np.float32)
        faces = self.mesh.faces.flatten().astype(np.uint32)
        
        # Create vertex buffer
        self.vbo = self.ctx.buffer(vertices.tobytes())
        
        # Create index buffer
        self.ibo = self.ctx.buffer(faces.tobytes())
        
        # Create vertex array objects for both programs
        self.face_id_vao = self.ctx.vertex_array(self.face_id_program, [(self.vbo, '3f', 'in_position')], self.ibo)
        self.normal_vao = self.ctx.vertex_array(self.normal_program, [(self.vbo, '3f', 'in_position')], self.ibo)
        
        # Set the active VAO based on debug flag
        self.vao = self.normal_vao if self.debug_normals else self.face_id_vao


    def render_face_ids(self, viewmat: torch.Tensor, K: torch.Tensor, 
                       height: int, width: int) -> torch.Tensor:
        """
        Render face IDs using ModernGL.
        Returns a tensor of shape [height, width] with face IDs.
        """
        # Create framebuffer
        color_texture = self.ctx.texture((width, height), 4)
        depth_texture = self.ctx.depth_texture((width, height))
        fbo = self.ctx.framebuffer(
            color_attachments=[color_texture],
            depth_attachment=depth_texture
        )
        
        # Convert matrices to numpy
        viewmat_np = viewmat.cpu().numpy().astype(np.float32)
        K_np = K.cpu().numpy().astype(np.float32)
        
        # Apply COLMAP to OpenGL coordinate system conversion
        # Based on working solution from interactive_mesh_viewer.py
        # COLMAP: +X right, +Y down, +Z forward
        # OpenGL: +X right, +Y up, +Z backward
        transform = np.array([
            [1,  0,  0,  0],  # Keep X axis
            [0,  1,  0,  0],  # Keep Y axis (no flip needed with corrected projection)
            [0,  0, -1,  0],  # Flip Z axis (COLMAP->OpenGL)
            [0,  0,  0,  1]
        ], dtype=np.float32)
        
        # Convert viewmat (w2c) to c2w, apply transform, then back to w2c
        c2w = np.linalg.inv(viewmat_np)
        opengl_c2w = c2w @ transform
        viewmat_gl = np.linalg.inv(opengl_c2w)
        
        # Get mesh bounds for appropriate near/far planes
        mesh_bounds = self.mesh.bounds
        mesh_center = self.mesh.vertices.mean(axis=0)
        mesh_size = np.linalg.norm(mesh_bounds[1] - mesh_bounds[0])
        
        # Use simple hardcoded near/far planes
        near = 1.0
        far = 1000.0
        
        # Create projection matrix from camera intrinsics (COLMAP coordinate system)
        fx, fy = K_np[0, 0], K_np[1, 1]
        cx, cy = K_np[0, 2], K_np[1, 2]
        
        # OpenGL projection matrix from COLMAP intrinsics
        # Note: COLMAP image coordinates are (0,0) at top-left, OpenGL at bottom-left
        # So we need to flip the Y coordinate in the projection
        proj_matrix = np.array([
            [2.0 * fx / width, 0, (2.0 * cx - width) / width, 0],
            [0, -2.0 * fy / height, (height - 2.0 * cy) / height, 0],  # Negative fy flips Y axis, adjust cy for Y-flip
            [0, 0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        
        # Combine matrices
        mvp_matrix = proj_matrix @ viewmat_gl
        
        # Debug: Check transformation
        camera_pos_world = np.linalg.inv(viewmat_np)[:3, 3]
        print(f"Camera position (world): {camera_pos_world}")
        print(f"Mesh center (world): {mesh_center}")
        
        # Transform mesh center to camera space for debugging
        mesh_center_homo = np.append(mesh_center, 1.0)
        mesh_center_cam = viewmat_gl @ mesh_center_homo
        print(f"Mesh center in camera space: {mesh_center_cam[:3]}")
        print(f"Near/Far: {near}/{far}")
        
        # Set uniforms and render - transpose matrix for OpenGL column-major format
        if self.debug_normals:
            # Use normal rendering program
            self.normal_program['mvp'].write(mvp_matrix.T.tobytes())
            self.vao = self.normal_vao
            debug_filename = "debug_normals.png"
        else:
            # Use face ID rendering program
            self.face_id_program['mvp'].write(mvp_matrix.T.tobytes())
            self.face_id_program['base_face_id'].value = 1  # Start face IDs from 1 (0 = background)
            self.vao = self.face_id_vao
            debug_filename = "debug_face_ids.png"
        
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        
        self.vao.render()
        
        # Read framebuffer and save debug image
        data = fbo.color_attachments[0].read()
        img_array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
        img_array = np.flip(img_array, axis=0)
        
        # Save debug image to check if anything renders
        debug_img = Image.fromarray(img_array[:, :, :3])
        debug_img.save(debug_filename)
        print(f"Saved {debug_filename}")
        
        if self.debug_normals:
            print("DEBUG MODE: Rendering normals as colors (R=X, G=Y, B=Z)")
            # Return dummy face IDs for debug mode
            face_ids = np.zeros((height, width), dtype=np.int32)
            face_ids_tensor = torch.from_numpy(face_ids).to(self.device)
            
            # Clean up ModernGL resources
            fbo.release()
            color_texture.release()
            depth_texture.release()
            
            return face_ids_tensor
        
        # Check if anything was rendered
        non_black_pixels = np.sum(img_array[:, :, :3].any(axis=2))
        print(f"Non-black pixels: {non_black_pixels} / {height*width}")
        
        # Decode face IDs from RGB channels (24-bit encoding)
        r = img_array[:, :, 0].astype(np.int32)
        g = img_array[:, :, 1].astype(np.int32)
        b = img_array[:, :, 2].astype(np.int32)
        face_ids = (r << 16) | (g << 8) | b
        
        face_ids_tensor = torch.from_numpy(face_ids).to(self.device)
        
        # Clean up ModernGL resources
        fbo.release()
        color_texture.release()
        depth_texture.release()
        
        return face_ids_tensor

    def aggregate_features_vectorized(self, face_ids: torch.Tensor, features: torch.Tensor, 
                                    face_features: torch.Tensor, face_weights: torch.Tensor) -> None:
        """
        Vectorized feature aggregation directly from face IDs without explicit pixel mapping.
        Much faster than the original get_face_pixel_mapping + loop approach.
        """
        height, width = face_ids.shape
        C, H, W = features.shape
        
        # Flatten face_ids and features for vectorized operations
        face_ids_flat = face_ids.flatten()  # [H*W]
        features_flat = features.view(C, -1).T  # [H*W, C]
        
        # Remove background pixels (face_id = 0)
        valid_mask = face_ids_flat > 0
        if not valid_mask.any():
            return  # No valid faces in this image
        
        valid_face_ids = face_ids_flat[valid_mask]  # [N_valid]
        valid_features = features_flat[valid_mask]  # [N_valid, C]
        
        # Convert to 0-based indexing (subtract 1 since face IDs start from 1)
        face_indices = (valid_face_ids - 1).long()  # [N_valid] - convert to int64 for scatter_add
        
        # Normalize features
        valid_features = F.normalize(valid_features, p=2, dim=-1)
        
        # Use scatter_add for efficient aggregation
        # This accumulates features for each face index
        face_features.scatter_add_(0, face_indices.unsqueeze(1).expand(-1, C), valid_features)
        
        # Count pixels per face using scatter_add with ones
        pixel_counts = torch.ones(valid_face_ids.shape[0], device=self.device, dtype=face_weights.dtype)
        face_weights.scatter_add_(0, face_indices, pixel_counts)
        
        # Clean up all intermediate tensors
        del pixel_counts, valid_face_ids, face_indices, valid_features, face_ids_flat, features_flat, valid_mask

    def sample_features_at_pixels(self, features: torch.Tensor, pixel_coords: torch.Tensor) -> torch.Tensor:
        """Sample features at specific pixel coordinates. (Legacy method - kept for compatibility)"""
        # features: [C, H, W]
        # pixel_coords: [N, 2] (x, y coordinates)
        
        C, H, W = features.shape
        N = pixel_coords.shape[0]
        
        # Clamp coordinates to valid range
        x_coords = torch.clamp(pixel_coords[:, 0].long(), 0, W - 1)
        y_coords = torch.clamp(pixel_coords[:, 1].long(), 0, H - 1)
        
        # Sample features
        sampled_features = features[:, y_coords, x_coords].T  # [N, C]
        
        return sampled_features

    # def get_face_pixel_mapping(self, face_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
    #     \"\"\"\n        Convert face ID image to face -> pixel coordinates mapping.\n        NOTE: This is the old slow method, kept for debugging. Use aggregate_features_vectorized() instead.\n        \"\"\"\n        face_pixel_map = {}\n        height, width = face_ids.shape\n        \n        # Create coordinate grids\n        y_coords, x_coords = torch.meshgrid(\n            torch.arange(height, device=self.device, dtype=torch.float32),\n            torch.arange(width, device=self.device, dtype=torch.float32),\n            indexing='ij'\n        )\n        \n        # Get unique face IDs (excluding background = 0)\n        unique_faces = torch.unique(face_ids)\n        unique_faces = unique_faces[unique_faces > 0]  # Remove background\n        \n        for face_id in unique_faces:\n            # Find pixels belonging to this face\n            mask = face_ids == face_id\n            if mask.any():\n                face_pixels = torch.stack([x_coords[mask], y_coords[mask]], dim=1)\n                # Convert face ID back to 0-based indexing (subtract 1 since we start from 1)\n                actual_face_id = face_id.item() - 1\n                face_pixel_map[actual_face_id] = face_pixels\n        \n        return face_pixel_map

    @torch.no_grad()
    def distill(self) -> None:
        """Entry for mesh feature distillation."""
        print("Running mesh feature distillation...")
        device = self.device
        
        # Get mesh data
        num_faces = len(self.mesh.faces)
        
        # Prepare data structures for features
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=1, shuffle=False
        )
        feature_dim = self.trainset[0]["features"].shape[-1]
        
        # Use weighted averaging for feature accumulation
        face_features = torch.zeros(
            (num_faces, feature_dim), dtype=torch.float32, device=device
        )
        face_weights = torch.zeros(
            (num_faces,), dtype=torch.float32, device=device
        )
        
        for i, data in tqdm(enumerate(trainloader), desc="Distilling features", total=len(trainloader)):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            features = data["features"].to(device)
            
            # Process features
            features = features.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            features = torch.nn.functional.interpolate(
                features,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            features = features.squeeze(0)  # Remove batch dimension: [C, H, W]
            
            # Get viewmat (inverse of camtoworld) - DO NOT modify the rotation!
            # The camtoworld matrix contains the correct camera pose from COLMAP
            viewmat = torch.linalg.inv(camtoworlds.squeeze(0))
            K = Ks.squeeze(0)
            
            # Render face IDs using ModernGL
            face_ids = self.render_face_ids(viewmat, K, height, width)
            
            # Vectorized feature aggregation (much faster than the old approach)
            self.aggregate_features_vectorized(face_ids, features, face_features, face_weights)
            
            # Explicit cleanup of GPU tensors
            del features, face_ids, camtoworlds, Ks, pixels
            torch.cuda.empty_cache()  # Force GPU memory cleanup
        
        # Normalize accumulated features
        face_weights = torch.clamp(face_weights, min=1e-8)
        face_features = face_features / face_weights.unsqueeze(-1)
        face_features = torch.nan_to_num(face_features, nan=0.0)
        
        # Save results
        basename, _ = os.path.splitext(self.args.mesh_path)
        output_path = f"{basename}_face_features.pt"
        
        torch.save({
            'features': face_features.cpu(),
            'weights': face_weights.cpu(),
            'mesh_path': self.args.mesh_path,
            'feature_dim': feature_dim,
            'num_faces': num_faces
        }, output_path)
        
        print(f"Saved mesh face features to: {output_path}")
        print(f"Features shape: {face_features.shape}")
        print(f"Faces with features: {(face_weights > 0).sum().item()}/{num_faces}")
        print(f"Average pixels per face: {face_weights[face_weights > 0].mean().item():.2f}")

    def cleanup(self):
        """Clean up ModernGL resources."""
        if hasattr(self, 'face_id_vao'):
            self.face_id_vao.release()
        if hasattr(self, 'normal_vao'):
            self.normal_vao.release()
        if hasattr(self, 'vbo'):
            self.vbo.release()
        if hasattr(self, 'ibo'):
            self.ibo.release()
        if hasattr(self, 'face_id_program'):
            self.face_id_program.release()
        if hasattr(self, 'normal_program'):
            self.normal_program.release()
        if hasattr(self, 'ctx'):
            self.ctx.release()


def main():
    parser = argparse.ArgumentParser(description="Distill 2D features onto 3D mesh faces")
    parser.add_argument(
        "--mesh-path",
        type=str,
        required=True,
        help="Path to the 3D mesh file (.obj)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the dataset"
    )
    parser.add_argument(
        "--data-factor",
        type=int,
        default=1,
        help="Downsample factor for the dataset"
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=8,
        help="Every N images there is a test image"
    )
    
    args = parser.parse_args()
    
    distiller = MeshDistiller(args)
    try:
        distiller.distill()
    finally:
        distiller.cleanup()


if __name__ == "__main__":
    main()