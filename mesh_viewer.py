import argparse
import os
import time
import moderngl
import trimesh
import numpy as np
from pathlib import Path
from typing import Literal, Callable, Tuple

import torch
import torch.nn.functional as F
import viser

from nerfview import Viewer, CameraState, RenderTabState, apply_float_colormap
from sklearn.decomposition import PCA
from featup.featurizers.maskclip.clip import tokenize


class MeshRenderTabState(RenderTabState):
    """Render tab state for mesh viewer."""
    
    # Mesh-specific parameters
    background_color: tuple[float, float, float] = (0.1, 0.1, 0.1)
    render_mode: Literal["normal", "texture", "feature", "relevance"] = "normal"
    
    # Feature visualization parameters
    enable_features: bool = False
    query_text: str = ""
    text_change: bool = False
    relevance: torch.Tensor = None
    colormap: str = "turbo"


def get_proj_mat(
    K: torch.Tensor,
    img_wh: Tuple[int, int],
    znear: float = 0.001,
    zfar: float = 1000.0,
) -> torch.Tensor:
    """Get the OpenGL-style projection matrix.

    Args:
        K (torch.Tensor): (3, 3) camera intrinsic matrix.
        img_wh (Tuple[int, int]): Image width and height.
        znear (float): Near plane.
        zfar (float): Far plane.

    Returns:
        proj_mat (torch.Tensor): (4, 4) projection matrix.
    """
    W, H = img_wh
    # Assume a camera model without distortion.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fovx = 2.0 * torch.arctan(W / (2.0 * fx)).item()
    fovy = 2.0 * torch.arctan(H / (2.0 * fy)).item()
    t = znear * np.tan(0.5 * fovy).item()
    b = -t
    r = znear * np.tan(0.5 * fovx).item()
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (cx - W / 2) / W * 2, 0.0],
            [0.0, 2 * n / (t - b), (cy - H / 2) / H * 2, 0.0],
            [0.0, 0.0, (f + n) / (f - n), -f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=K.device,
    )




class MeshRenderer:
    """GPU-accelerated mesh renderer using ModernGL for normal visualization and feature rendering."""
    
    def __init__(self, mesh_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store mesh path for feature loading
        self.mesh_path = mesh_path
        
        # Load mesh
        self._load_mesh(mesh_path)
        
        # Initialize ModernGL context (delay until first render to avoid conflicts)
        self.ctx = None
        self.program = None
        self.vao = None
        
        # Feature data
        self.face_features = None
        self.face_features_pca = None
        self.clip_model = None
        
        # Load features and initialize CLIP model
        self._load_features()
        self._init_clip_model()
        

    def _load_mesh(self, mesh_path: str):
        """Load mesh from file."""
        loaded_mesh = trimesh.load(mesh_path)
        
        if isinstance(loaded_mesh, trimesh.Scene):
            if len(loaded_mesh.geometry) == 0:
                raise ValueError("No geometry found in the scene")
            
            # Handle multi-material scenes properly
            meshes = [(name, geom) for name, geom in loaded_mesh.geometry.items() if isinstance(geom, trimesh.Trimesh)]
            if len(meshes) == 0:
                raise ValueError("No triangle meshes found in the scene")
            elif len(meshes) == 1:
                self.mesh = meshes[0][1]
                self.scene = None
            else:
                # Keep the scene to preserve material information
                self.scene = loaded_mesh
                self.mesh = self._combine_scene_with_materials(loaded_mesh)
        elif isinstance(loaded_mesh, trimesh.Trimesh):
            self.mesh = loaded_mesh
            self.scene = None
        else:
            raise ValueError(f"Unsupported mesh type: {type(loaded_mesh)}")
        
        # Set vertex colors and texture coordinates only for single mesh case
        # Multi-material meshes already have these set by _combine_scene_with_materials
        if self.scene is None:  # Single mesh case
            # Always use default gray vertex colors
            self.vertex_colors = np.ones((len(self.mesh.vertices), 3), dtype=np.float32) * 0.7
            
            # Check for texture coordinates
            if hasattr(self.mesh.visual, 'uv') and self.mesh.visual.uv is not None:
                self.texture_coords = self.mesh.visual.uv.astype(np.float32)
            else:
                # Default UV coordinates (0,0) for all vertices
                self.texture_coords = np.zeros((len(self.mesh.vertices), 2), dtype=np.float32)
        
        # Center the mesh at origin (keep original scale)
        self._center_mesh()
        
        print(f"Loaded mesh: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")

    def _combine_scene_with_materials(self, scene):
        """Combine scene meshes while preserving material groups for efficient rendering."""
        all_vertices = []
        all_faces = []
        all_texture_coords = []
        all_vertex_colors = []
        material_groups = []  # List of (face_indices, texture, material_name)
        vertex_offset = 0
        face_offset = 0
        
        # Extract materials from scene first
        scene_materials = {}
        if hasattr(scene, 'materials') and scene.materials:
            # Sort materials by name to ensure consistent ordering
            sorted_materials = sorted(scene.materials.items())
            for mat_name, material in sorted_materials:
                scene_materials[mat_name] = material
        
        # Process geometry in original order (don't sort - this was causing the texture mismatch!)
        for name, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                # Add vertices with offset tracking
                all_vertices.append(geom.vertices)
                
                # Add faces with vertex offset
                faces_with_offset = geom.faces + vertex_offset
                all_faces.append(faces_with_offset)
                
                material = None
                material_name = f"default_material_{len(material_groups)}"
                
                # Try to get material from geometry visual
                if hasattr(geom.visual, 'material') and geom.visual.material is not None:
                    material = geom.visual.material
                    material_name = getattr(material, 'name', material_name)
                elif hasattr(geom.visual, 'material_name') and geom.visual.material_name in scene_materials:
                    material = scene_materials[geom.visual.material_name]
                    material_name = geom.visual.material_name
                
                # Create face indices for this material group
                num_faces = len(geom.faces)
                face_indices = np.arange(face_offset, face_offset + num_faces, dtype=np.uint32)
                
                # Load texture for this material
                texture_image = None
                if material is not None and hasattr(material, 'image') and material.image is not None:
                    texture_image = material.image
                
                # Add to material groups
                material_groups.append({
                    'face_indices': face_indices,
                    'texture_image': texture_image,
                    'material_name': material_name,
                    'num_faces': num_faces
                })
                
                # Handle texture coordinates
                if hasattr(geom.visual, 'uv') and geom.visual.uv is not None:
                    uv_coords = geom.visual.uv
                    
                    # Ensure UV coordinates match vertex count
                    if len(uv_coords) == len(geom.vertices):
                        all_texture_coords.append(uv_coords.astype(np.float32))
                    else:
                        default_uv = np.zeros((len(geom.vertices), 2), dtype=np.float32)
                        all_texture_coords.append(default_uv)
                else:
                    # Default UV coordinates
                    default_uv = np.zeros((len(geom.vertices), 2), dtype=np.float32)
                    all_texture_coords.append(default_uv)
                
                vertex_colors = np.ones((len(geom.vertices), 3), dtype=np.float32) * 0.7
                all_vertex_colors.append(vertex_colors)
                
                vertex_offset += len(geom.vertices)
                face_offset += num_faces
        
        # Combine all data manually
        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)
        combined_texture_coords = np.vstack(all_texture_coords)
        combined_vertex_colors = np.vstack(all_vertex_colors)
        
        # Create combined mesh
        combined_mesh = trimesh.Trimesh(
            vertices=combined_vertices,
            faces=combined_faces,
            process=False
        )
        
        # Store material groups for efficient rendering
        self.material_groups = material_groups
        
        # Override texture coordinates and vertex colors
        self.texture_coords = combined_texture_coords
        self.vertex_colors = combined_vertex_colors
        self.has_texture_coords = True
        self.has_vertex_colors = True
        
        return combined_mesh

    def _center_mesh(self):
        """Center mesh at origin and scale."""
        bounds = self.mesh.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        self.mesh.vertices -= center
        self.mesh.vertices *= 0.2
    
    def _load_features(self):
        """Load pre-computed mesh face features from mesh_distill.py output."""
        # Try to find features file based on mesh path
        mesh_basename = os.path.splitext(self.mesh_path)[0]
        feature_path = f"{mesh_basename}_face_features.pt"
        
        if not os.path.exists(feature_path):
            print(f"No features found at {feature_path}")
            return
        
        try:
            feature_data = torch.load(feature_path, map_location=self.device)
            self.face_features = feature_data['features'].to(self.device)  # [num_faces, feature_dim]
            print(f"Loaded face features: {self.face_features.shape}")
            
            if self.face_features is not None:
                features_np = self.face_features.cpu().numpy()
                pca = PCA(n_components=3)
                features_pca = pca.fit_transform(features_np)
                self.face_features_pca = torch.from_numpy(features_pca).float().to(self.device)
                print(f"PCA features computed: {self.face_features_pca.shape}")
                
        except Exception as e:
            print(f"Failed to load features: {e}")
    
    def _init_clip_model(self):
        """Initialize CLIP model for text-based relevance queries."""
        if self.face_features is None:
            return
            
        try:
            self.clip_model = (
                torch.hub.load("mhamilton723/FeatUp", "maskclip", use_norm=False)
                .to(self.device)
                .eval()
                .model.model
            )
            print("CLIP model loaded")
        except Exception as e:
            print(f"Failed to load CLIP model: {e}")
    
    @torch.no_grad()
    def compute_relevance(self, query_text: str) -> torch.Tensor:
        """Compute text relevance scores for mesh faces using CLIP."""
        if self.face_features is None or self.clip_model is None:
            return None
            
        # Encode text query
        text_features = self.clip_model.encode_text(
            tokenize(query_text).to(self.device)
        ).float()
        text_features = F.normalize(text_features, dim=0)
        
        # Normalize face features
        face_features_norm = F.normalize(self.face_features, dim=-1)
        
        sim = torch.sum(face_features_norm * text_features, dim=-1, keepdim=True)
        sim = sim.clamp(min=sim.mean())
        sim = (sim - sim.min()) / (sim.max() - sim.min())
        return sim.squeeze(-1)
        

    def _setup_shaders(self):
        """Setup OpenGL shaders for normal, texture, and feature rendering."""
        vertex_shader = '''
        #version 430 core
        in vec3 in_position;
        in vec3 in_color;
        in vec2 in_texcoord;
        out vec3 world_pos;
        out vec3 vertex_color;
        out vec2 texcoord;
        uniform mat4 mvp;
        
        void main() {
            world_pos = in_position;
            vertex_color = in_color;
            texcoord = in_texcoord;
            gl_Position = mvp * vec4(in_position, 1.0);
        }
        '''
        
        fragment_shader = '''
        #version 430 core
        in vec3 world_pos;
        in vec3 vertex_color;
        in vec2 texcoord;
        out vec4 color;
        
        uniform int render_mode;
        uniform sampler2D tex_diffuse;
        uniform bool has_texture;
        uniform int num_faces;
        
        layout(std430, binding = 0) readonly buffer FaceColorBuffer {
            vec4 face_colors[];
        };
    
        void main() {
            if (render_mode == 0) {
                vec3 normal = normalize(cross(dFdx(world_pos), dFdy(world_pos)));
                color = vec4(normal * 0.5 + 0.5, 1.0);
            } else if (render_mode == 2 || render_mode == 3) {
                int face_id = gl_PrimitiveID;
                color = (face_id >= 0 && face_id < num_faces) ? 
                       vec4(face_colors[face_id].rgb, 1.0) : vec4(1,0,0,1);
            } else {
                color = has_texture ? 
                       texture(tex_diffuse, vec2(texcoord.x, 1.0 - texcoord.y)) : 
                       vec4(vertex_color, 1.0);
            }
        }
        '''
        
        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
        self.face_color_buffer = None

    def _init_opengl(self):
        """Initialize OpenGL context and resources on first use."""
        if self.ctx is not None:
            return
        
        try:
            self.ctx = moderngl.create_context()
        except:
            self.ctx = moderngl.create_context(standalone=True)
        
        self._setup_shaders()
        self._prepare_mesh_data()

    def _prepare_mesh_data(self):
        """Prepare mesh data for OpenGL rendering."""
        vertices = self.mesh.vertices.astype(np.float32)
        faces = self.mesh.faces.astype(np.uint32)
        
        vertex_data = np.column_stack([
            vertices, self.vertex_colors, self.texture_coords
        ]).astype(np.float32)
        
        self.vbo = self.ctx.buffer(vertex_data.tobytes())
        self.ibo_full = self.ctx.buffer(faces.flatten().tobytes())
        
        self.material_ibos = []
        self.material_textures = []
        
        if hasattr(self, 'material_groups'):
            for group in self.material_groups:
                face_indices = group['face_indices']
                material_faces = faces[face_indices]
                ibo = self.ctx.buffer(material_faces.tobytes())
                self.material_ibos.append(ibo)
                texture = self._load_single_texture(group['texture_image'])
                self.material_textures.append(texture)
        
        self.vao_full = self.ctx.vertex_array(self.program, [
            (self.vbo, '3f 3f 2f', 'in_position', 'in_color', 'in_texcoord')
        ], self.ibo_full)
        
        self.material_vaos = []
        for ibo in self.material_ibos:
            vao = self.ctx.vertex_array(self.program, [
                (self.vbo, '3f 3f 2f', 'in_position', 'in_color', 'in_texcoord')
            ], ibo)
            self.material_vaos.append(vao)
        
        self._load_texture()
        self._create_face_color_buffer()

    def _load_single_texture(self, texture_image):
        """Load a single texture from PIL image."""
        if texture_image is None:
            return None
            
        try:
            # Convert PIL image to numpy array
            texture_array = np.array(texture_image)
            if len(texture_array.shape) == 3 and texture_array.shape[2] >= 3:
                # Create OpenGL texture
                height, width = texture_array.shape[:2]
                texture = self.ctx.texture((width, height), 3, texture_array.tobytes())
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                return texture
        except Exception as e:
            print(f"Failed to load texture: {e}")
        
        return None

    def _load_texture(self):
        """Load default texture for single material meshes (fallback)."""
        self.default_texture = None
        self.has_loaded_texture = False
        
        if hasattr(self.mesh.visual, 'material') and hasattr(self.mesh.visual.material, 'image'):
            self.default_texture = self._load_single_texture(self.mesh.visual.material.image)
            if self.default_texture is not None:
                self.has_loaded_texture = True
        
        if not self.has_loaded_texture:
            default_texture_data = np.array([[[255, 255, 255]]], dtype=np.uint8)
            self.default_texture = self.ctx.texture((1, 1), 3, default_texture_data.tobytes())
    
    def _create_face_color_buffer(self):
        """Create SSBO for per-face colors with vec4 alignment."""
        num_faces = len(self.mesh.faces)
        if self.face_color_buffer is not None:
            self.face_color_buffer.release()
        
        face_colors = np.ones((num_faces, 4), dtype=np.float32)
        face_colors[:, :3] = 0.7  # Default gray
        
        try:
            self.face_color_buffer = self.ctx.buffer(face_colors.tobytes())
        except Exception as e:
            print(f"Failed to create face color SSBO: {e}")
            fallback_data = np.array([[0.7, 0.7, 0.7, 1.0]], dtype=np.float32)
            self.face_color_buffer = self.ctx.buffer(fallback_data.tobytes())


    def update_face_colors(self, face_colors: np.ndarray):
        """Update per-face colors for feature visualization."""
        if self.face_color_buffer is None:
            return
        
        num_faces = face_colors.shape[0]
        face_colors_vec4 = np.ones((num_faces, 4), dtype=np.float32)
        face_colors_vec4[:, :3] = face_colors.astype(np.float32)
        self.face_color_buffer.write(face_colors_vec4.tobytes())

    def _prepare_feature_colors(self, render_tab_state: MeshRenderTabState) -> np.ndarray:
        """Prepare per-face colors for feature visualization."""
        num_faces = len(self.mesh.faces)
        
        if render_tab_state.render_mode == "feature" and self.face_features_pca is not None:
            face_colors = self.face_features_pca.cpu().numpy()
            face_colors = (face_colors - face_colors.min()) / (face_colors.max() - face_colors.min())
            return face_colors
        elif render_tab_state.render_mode == "relevance" and render_tab_state.relevance is not None:
            relevance_scores = render_tab_state.relevance.cpu().numpy()
            face_colors = apply_float_colormap(
                torch.from_numpy(relevance_scores).unsqueeze(-1), 
                render_tab_state.colormap
            ).squeeze(-1).cpu().numpy()
            return face_colors
        else:
            return np.full((num_faces, 3), 0.7, dtype=np.float32)
    
    def render(self, camera_state: CameraState, render_tab_state: MeshRenderTabState) -> np.ndarray:
        """Render mesh with current settings."""
        # Initialize OpenGL context on first use
        self._init_opengl()
        
        width = render_tab_state.viewer_width
        height = render_tab_state.viewer_height
        img_wh = [width, height]

        # Create framebuffer
        color_attachment = self.ctx.texture((width, height), 4)
        depth_attachment = self.ctx.depth_texture((width, height))
        fbo = self.ctx.framebuffer(
            color_attachments=[color_attachment],
            depth_attachment=depth_attachment
        )

        # Setup rendering state
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        bg_color = render_tab_state.background_color
        self.ctx.clear(bg_color[0], bg_color[1], bg_color[2], 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)  # Show both sides

        # Calculate MVP matrix
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        K = torch.as_tensor(K, dtype=torch.float32)
        
        view_matrix = np.linalg.inv(c2w).astype(np.float32)
        
        proj_matrix = get_proj_mat(K, img_wh).numpy()
        
        mvp_matrix = proj_matrix @ view_matrix
        
        # Set common uniforms
        self.program['mvp'].write(mvp_matrix.T.tobytes())
        self.program['num_faces'] = len(self.mesh.faces)
        
        if render_tab_state.render_mode in ["feature", "relevance"]:
            face_colors = self._prepare_feature_colors(render_tab_state)
            self.update_face_colors(face_colors)
            if self.face_color_buffer is not None:
                self.face_color_buffer.bind_to_storage_buffer(0)
        
        mode_map = {"normal": 0, "texture": 1, "feature": 2, "relevance": 3}
        self.program['render_mode'] = mode_map.get(render_tab_state.render_mode, 0)
        
        if render_tab_state.render_mode == "texture":
            if hasattr(self, 'material_vaos') and self.material_vaos:
                for vao, texture in zip(self.material_vaos, self.material_textures):
                    self.program['has_texture'] = texture is not None
                    if texture is not None:
                        texture.use(location=0)
                        self.program['tex_diffuse'] = 0
                    vao.render()
            else:
                self.program['has_texture'] = self.has_loaded_texture
                if self.default_texture is not None:
                    self.default_texture.use(location=0)
                    self.program['tex_diffuse'] = 0
                self.vao_full.render()
        else:
            self.vao_full.render()
        
        # Read back framebuffer
        data = color_attachment.read()
        img_array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
        
        # Cleanup
        fbo.release()
        color_attachment.release()
        depth_attachment.release()
        
        return img_array[:, :, :3]

    def cleanup(self):
        """Clean up OpenGL resources."""
        if hasattr(self, 'vao_full'):
            self.vao_full.release()
        if hasattr(self, 'material_vaos'):
            for vao in self.material_vaos:
                vao.release()
        if hasattr(self, 'vbo'):
            self.vbo.release()
        if hasattr(self, 'ibo_full'):
            self.ibo_full.release()
        if hasattr(self, 'material_ibos'):
            for ibo in self.material_ibos:
                ibo.release()
        if hasattr(self, 'material_textures'):
            for texture in self.material_textures:
                if texture is not None:
                    texture.release()
        if hasattr(self, 'default_texture'):
            if self.default_texture is not None:
                self.default_texture.release()
        if hasattr(self, 'face_color_buffer'):
            if self.face_color_buffer is not None:
                self.face_color_buffer.release()
        if hasattr(self, 'program'):
            self.program.release()
        if hasattr(self, 'ctx'):
            self.ctx.release()


class MeshViewer(Viewer):
    """Mesh viewer using nerfview framework."""
    
    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mesh_renderer: MeshRenderer = None,
        mode: Literal["rendering", "training"] = "rendering",
    ):
        # Set mesh_renderer before calling parent constructor
        # because parent may call _populate_rendering_tab during init
        self.mesh_renderer = mesh_renderer
        
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("Mesh Viewer")

    def _init_rendering_tab(self):
        """Initialize rendering tab state."""
        self.render_tab_state = MeshRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        """Populate rendering tab with mesh-specific controls."""
        server = self.server
        with self._rendering_folder:
            with server.gui.add_folder("Mesh Rendering"):
                
                # Check if features are available (with safety check)
                has_features = (getattr(self, 'mesh_renderer', None) is not None and 
                               hasattr(self.mesh_renderer, 'face_features') and 
                               self.mesh_renderer.face_features is not None)
                
                # Render mode dropdown - add feature modes if available
                if has_features:
                    render_options = ("normal", "texture", "feature", "relevance")
                else:
                    render_options = ("normal", "texture")
                
                render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    options=render_options,
                    initial_value=self.render_tab_state.render_mode,
                    hint="Choose rendering mode",
                )

                @render_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.render_mode = render_mode_dropdown.value
                    self.rerender(_)

                # Background color
                background_color_rgb = server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.background_color,
                    hint="Background color for rendering",
                )

                @background_color_rgb.on_update
                def _(_) -> None:
                    self.render_tab_state.background_color = background_color_rgb.value
                    self.rerender(_)
                
                # Feature controls (only show if features are available)
                if has_features:
                    with server.gui.add_folder("Feature Visualization"):
                        # Query text for relevance computation
                        query_text_input = server.gui.add_text(
                            "Prompt",
                            initial_value=self.render_tab_state.query_text,
                            hint="Enter text query for relevance visualization",
                        )
                        
                        # Query button to trigger relevance computation
                        query_submit_button = server.gui.add_button(
                            "Query",
                            hint="Click to compute relevance for the entered text",
                        )

                        @query_submit_button.on_click
                        def _(_) -> None:
                            self.render_tab_state.query_text = query_text_input.value
                            self.render_tab_state.text_change = True
                            self.rerender(_)
                        
                        # Colormap for relevance visualization
                        colormap_dropdown = server.gui.add_dropdown(
                            "Colormap",
                            options=("turbo", "viridis", "plasma", "inferno", "jet"),
                            initial_value=self.render_tab_state.colormap,
                            hint="Colormap for relevance visualization",
                        )

                        @colormap_dropdown.on_update
                        def _(_) -> None:
                            self.render_tab_state.colormap = colormap_dropdown.value
                            if self.render_tab_state.render_mode == "relevance":
                                self.rerender(_)
                else:
                    # Show disabled placeholder if no features
                    enable_features_checkbox = server.gui.add_checkbox(
                        "Enable Features",
                        initial_value=False,
                        disabled=True,
                        hint="No features found - run mesh_distill.py first",
                    )

                    query_text_input = server.gui.add_text(
                        "Query Text",
                        initial_value="",
                        disabled=True,
                        hint="Feature support requires running mesh_distill.py first",
                    )

        # Store handles (different based on feature availability)
        handles = {
            "render_mode_dropdown": render_mode_dropdown,
            "background_color_rgb": background_color_rgb,
        }
        
        if has_features:
            handles.update({
                "query_text_input": query_text_input,
                "query_submit_button": query_submit_button,
                "colormap_dropdown": colormap_dropdown,
            })
        else:
            handles.update({
                "enable_features_checkbox": enable_features_checkbox,
                "query_text_input": query_text_input,
            })
        
        self._rendering_tab_handles.update(handles)
        
        # Call parent to add common controls
        super()._populate_rendering_tab()

    def _after_render(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="Mesh viewer with normal visualization")
    parser.add_argument("--mesh-path", type=str, required=True, help="Path to mesh file (.obj)")
    parser.add_argument("--port", type=int, default=8080, help="Port for viewer server")
    parser.add_argument("--output-dir", type=str, default="results/", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize mesh renderer
    mesh_renderer = MeshRenderer(args.mesh_path)
    
    # Create render function
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, MeshRenderTabState)
        
        # Compute relevance if text query changed
        if render_tab_state.text_change and render_tab_state.query_text.strip():
            render_tab_state.relevance = mesh_renderer.compute_relevance(render_tab_state.query_text)
            render_tab_state.text_change = False
        
        return mesh_renderer.render(camera_state, render_tab_state)
    
    # Start viser server
    server = viser.ViserServer(port=args.port, verbose=False)
    # Create mesh viewer
    viewer = MeshViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mesh_renderer=mesh_renderer,
        mode="rendering"
    )
    print(f"Mesh viewer running on http://localhost:{args.port}")
    
    try:
        time.sleep(100000)
    except KeyboardInterrupt:
        pass
    finally:
        mesh_renderer.cleanup()


if __name__ == "__main__":
    main()