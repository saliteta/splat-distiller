import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
import numpy as np
import trimesh
import argparse
from pathlib import Path
from colmap_loader import SceneManager
import pyrr

class MeshViewer(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Interactive Mesh Viewer - SPACE: toggle camera, L/R arrows: switch COLMAP cameras"
    window_size = (1200, 800)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Test flag - set to True to use a test sphere instead of loading mesh
        self.use_test_sphere = False
        
        # Get arguments from class attribute
        self.mesh_path = getattr(self, 'mesh_path', None)
        self.colmap_path = getattr(self, 'colmap_path', None)
        
        if not self.use_test_sphere and (not self.mesh_path or not self.colmap_path):
            raise ValueError("mesh_path and colmap_path must be provided")
        
        # Load mesh or create test sphere
        if self.use_test_sphere:
            self._create_test_sphere()
        else:
            self._load_mesh()
        
        # Load COLMAP data (only if not using test sphere)
        if not self.use_test_sphere:
            self._load_colmap_data()
        else:
            # Create dummy COLMAP data for test sphere
            self.colmap_cameras = {}
            self.colmap_images = {}
            self.image_list = []
        
        # Setup shaders
        self._setup_shaders()
        
        # Setup mesh rendering
        self._setup_mesh_rendering()
        
        # Camera control state
        self.camera_pos = np.array([0.0, 0.0, 20.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.camera_up = np.array([0.0, 1.0, 0.0])
        
        # Mouse control state
        self.mouse_last_pos = (0, 0)
        self.mouse_sensitivity = 0.01
        self.zoom_sensitivity = 0.1
        self.mouse_buttons = [False, False, False]  # Track mouse button states
        
        # Current COLMAP camera
        self.current_camera_idx = 0
        self.use_colmap_camera = True  # Start in manual mode
        
        # Set default camera position based on mesh bounds
        self._set_default_camera()

    def _load_mesh(self):
        """Load the mesh from file."""
        loaded_mesh = trimesh.load(self.mesh_path)
        if isinstance(loaded_mesh, trimesh.Scene):
            if len(loaded_mesh.geometry) == 0:
                raise ValueError("No geometry found in the scene")
            meshes = [geom for geom in loaded_mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            if len(meshes) == 0:
                raise ValueError("No triangle meshes found in the scene")
            elif len(meshes) == 1:
                self.mesh = meshes[0]
            else:
                print(f"Found {len(meshes)} meshes in scene, combining them...")
                self.mesh = trimesh.util.concatenate(meshes)
        elif isinstance(loaded_mesh, trimesh.Trimesh):
            self.mesh = loaded_mesh
        else:
            raise ValueError(f"Unsupported mesh type: {type(loaded_mesh)}")
        
        print(f"Loaded mesh: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        print(f"Mesh bounds: {self.mesh.bounds}")
        print(f"Mesh center: {self.mesh.vertices.mean(axis=0)}")

    def _create_test_sphere(self):
        """Create a test sphere mesh at the origin."""
        # Create a sphere with radius 5.0 centered at origin
        self.mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        #self.mesh.export("test_sphere.obj")
        
        print(f"Created test sphere: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        print(f"Sphere bounds: {self.mesh.bounds}")
        print(f"Sphere center: {self.mesh.vertices.mean(axis=0)}")

    def _load_colmap_data(self):
        """Load COLMAP camera data."""
        manager = SceneManager(self.colmap_path)
        manager.load_cameras()
        manager.load_images()
        
        # Extract camera data
        self.colmap_cameras = {}
        self.colmap_images = {}
        
        # Process cameras
        for cam_id, cam in manager.cameras.items():
            self.colmap_cameras[cam_id] = {
                'fx': cam.fx, 'fy': cam.fy, 'cx': cam.cx, 'cy': cam.cy,
                'width': cam.width, 'height': cam.height
            }
        
        # Process images (camera poses)
        for img_id, img in manager.images.items():
            # Get world-to-camera matrix
            rot = img.R()  # 3x3 rotation matrix
            trans = img.tvec.reshape(3, 1)  # 3x1 translation vector
            w2c = np.concatenate([np.concatenate([rot, trans], 1), 
                                 np.array([[0, 0, 0, 1]])], axis=0)
            
            # Convert to camera-to-world
            c2w = np.linalg.inv(w2c)
            
            self.colmap_images[img_id] = {
                'name': img.name,
                'camera_id': img.camera_id,
                'c2w': c2w,
                'w2c': w2c,
                'position': c2w[:3, 3],
                'rotation': c2w[:3, :3]
            }
        
        self.image_list = list(self.colmap_images.keys())
        print(f"Loaded {len(self.colmap_cameras)} cameras and {len(self.colmap_images)} images")

    def _set_default_camera(self):
        """Set a good default camera position based on mesh bounds."""
        # Get mesh center and size
        mesh_center = self.mesh.vertices.mean(axis=0)
        mesh_bounds = self.mesh.bounds
        mesh_size = np.linalg.norm(mesh_bounds[1] - mesh_bounds[0])
        
        # Position camera at a reasonable distance from the mesh
        camera_distance = mesh_size * 1.5
        
        # Set camera position (looking down the negative Z axis towards mesh center)
        self.camera_target = mesh_center.copy()
        #self.camera_pos = mesh_center + np.array([camera_distance * 0.5, camera_distance * 0.5, camera_distance])
        self.camera_pos = np.array([11.0, 3.0, 284.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.camera_up = np.array([0.0, 1.0, 0.0])

        mat = self._get_view_matrix()
        proj = self._get_projection_matrix()
        
        print(f"Set default camera:")
        print(f"  Position: {self.camera_pos}")
        print(f"  Target: {self.camera_target}")
        print(f"  Distance: {camera_distance:.2f}")
        print(f"  Mesh center: {mesh_center}")
        print(f"  Mesh size: {mesh_size:.2f}")

    def _setup_shaders(self):
        """Setup vertex and fragment shaders."""
        vertex_shader = '''
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
        
        fragment_shader = '''
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
        
        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

    def _setup_mesh_rendering(self):
        """Setup mesh buffers for rendering."""
        vertices = self.mesh.vertices.astype(np.float32)
        faces = self.mesh.faces.flatten().astype(np.uint32)
        
        # Create buffers
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(faces.tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '3f', 'in_position')], self.ibo)

    def _set_colmap_camera(self, image_idx):
        """Set camera to match COLMAP image."""
        if image_idx >= len(self.image_list):
            return
        
        img_id = self.image_list[image_idx]
        img_data = self.colmap_images[img_id]
        
        # Set camera position and orientation from COLMAP
        c2w = img_data['c2w']
        self.camera_pos = c2w[:3, 3].copy()
        
        # Calculate target point (camera looking forward)
        forward = -c2w[:3, 2]  # Negative Z is forward in camera space
        self.camera_target = self.camera_pos + forward * 10.0  # Look 10 units forward
        self.camera_up = c2w[:3, 1]  # Y axis

    def _get_view_matrix(self):
        """Calculate view matrix using pyrr library."""
        if self.use_colmap_camera and self.current_camera_idx < len(self.image_list):
            # Use COLMAP camera with coordinate system conversion
            img_id = self.image_list[self.current_camera_idx]
            colmap_w2c = self.colmap_images[img_id]['w2c'].astype(np.float32)

            # Use c2w approach for easier debugging
            colmap_c2w = self.colmap_images[img_id]['c2w'].astype(np.float32)

            # Coordinate transformation: COLMAP -> OpenGL + 180° rotation + horizontal flip
            # COLMAP: +X right, +Y down, +Z forward
            # OpenGL: +X right, +Y up, +Z backward
            # Additional: 180° rotation + horizontal flip = flip both X and Y
            transform = np.array([
                [1,  0,  0,  0],  # Flip X axis (horizontal flip)
                [0,  1,  0,  0],   # Keep Y axis (180° rotation cancels the Y flip)
                [0,  0, -1,  0],   # Flip Z axis (COLMAP->OpenGL)
                [0,  0,  0,  1]
            ], dtype=np.float32)

            # Apply transformation to c2w, then invert to get w2c
            opengl_c2w = colmap_c2w @ transform
            opengl_w2c = np.linalg.inv(opengl_c2w)
            return opengl_w2c
        else:
            # Use pyrr's robust lookAt matrix calculation
            view_matrix = pyrr.matrix44.create_look_at(
                eye=self.camera_pos,
                target=self.camera_target,
                up=self.camera_up,
                dtype=np.float32
            )
            # Transpose to column-major format for OpenGL
            return view_matrix.T

    def _get_projection_matrix(self):
        """Calculate projection matrix using pyrr library."""
        if self.use_colmap_camera and self.current_camera_idx < len(self.image_list):
            # Use COLMAP camera intrinsics
            img_id = self.image_list[self.current_camera_idx]
            cam_id = self.colmap_images[img_id]['camera_id']
            cam = self.colmap_cameras[cam_id]
            
            width, height = self.window_size
            fx, fy = cam['fx'], cam['fy']
            cx, cy = cam['cx'], cam['cy']
            
            # Scale intrinsics to current window size
            scale_x = width / cam['width']
            scale_y = height / cam['height']
            fx *= scale_x
            fy *= scale_y
            cx *= scale_x
            cy *= scale_y
            
            near, far = 1.0, 1000.0
            
            # Standard OpenGL projection matrix from COLMAP intrinsics
            # Note: COLMAP image coordinates are (0,0) at top-left, OpenGL at bottom-left
            # So we need to flip the Y coordinate in the projection
            proj_matrix = np.array([
                [2.0 * fx / width, 0, (2.0 * cx - width) / width, 0],
                [0, -2.0 * fy / height, (height - 2.0 * cy) / height, 0],  # Negative fy flips Y axis, adjust cy for Y-flip
                [0, 0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
                [0, 0, -1, 0]
            ], dtype=np.float32)
        else:
            # Use pyrr's robust perspective projection matrix
            fov = 60  # pyrr expects degrees, not radians
            # Use actual current viewport size instead of initial window_size
            viewport = self.ctx.viewport
            actual_width, actual_height = viewport[2], viewport[3]
            aspect = actual_width / actual_height
            near, far = 1.0, 1000.0
            
            proj_matrix = pyrr.matrix44.create_perspective_projection_matrix(
                fovy=fov,
                aspect=aspect,
                near=near,
                far=far,
                dtype=np.float32
            )
            # Transpose to column-major format for OpenGL
            proj_matrix = proj_matrix.T
        
        return proj_matrix

    def on_render(self, time, frame_time):
        """Main render function."""
        self.ctx.clear(0.1, 0.1, 0.2)
        self.ctx.enable(moderngl.DEPTH_TEST)
        # Disable face culling for debugging
        self.ctx.disable(moderngl.CULL_FACE)
        
        # Calculate matrices
        #model_matrix = np.eye(4, dtype=np.float32)
        view_matrix = self._get_view_matrix()
        proj_matrix = self._get_projection_matrix()
        #mvp_matrix = proj_matrix @ view_matrix #@ model_matrix
        mvp_matrix = proj_matrix @ view_matrix
        
        # Debug: print matrices occasionally
        if hasattr(self, '_debug_counter') and self._debug_counter % 120 == 0:  # Every 2 seconds
            print("=== MAIN VIEWER MATRIX DEBUG ===")
            print(f"Camera pos: {self.camera_pos}")
            print(f"Camera target: {self.camera_target}")
            
            # Test if a simple sphere vertex transforms correctly
            test_vertex = np.array([0.0, 0.0, 0.0, 1.0])  # Origin
            test_vertex_view = view_matrix @ test_vertex
            test_vertex_clip = mvp_matrix @ test_vertex
            print(f"Origin in view space: {test_vertex_view}")
            print(f"Origin in clip space: {test_vertex_clip}")
            
            # Test a vertex at sphere surface
            sphere_vertex = np.array([1.0, 0.0, 0.0, 1.0])  # Edge of unit sphere
            sphere_vertex_clip = mvp_matrix @ sphere_vertex
            print(f"Sphere edge in clip space: {sphere_vertex_clip}")
            
            print(f"View matrix:\n{view_matrix}")
            print(f"Proj matrix:\n{proj_matrix}")
            print("===================================")
        
        # Update uniforms - matrices already in column-major format
        self.program['mvp'].write(mvp_matrix.T.tobytes())
        
        # Render mesh
        self.vao.render()

    def _print_debug_info(self):

        mode_str = "COLMAP" if self.use_colmap_camera else "MANUAL"
        print(f"=== MODE: {mode_str} ===")

        if self.use_colmap_camera and self.current_camera_idx < len(self.image_list):
            img_id = self.image_list[self.current_camera_idx]
            img_data = self.colmap_images[img_id]
            print(f"COLMAP Camera {self.current_camera_idx}/{len(self.image_list)-1}: {img_data['name']}")
            print(f"Position: {img_data['position']}")
            print(f"Camera ID: {img_data['camera_id']}")
        else:
            print(f"Manual Camera - Pos: {self.camera_pos}, Target: {self.camera_target}")

        mesh_center = self.mesh.vertices.mean(axis=0)
        print(f"Mesh Center: {mesh_center}")
        print("Controls: SPACE=toggle mode, L/R arrows=switch cameras, Mouse=rotate/pan/zoom")
        print("---")

    def on_mouse_press_event(self, x, y, button):
        """Handle mouse press events."""
        if button < len(self.mouse_buttons):
            self.mouse_buttons[button] = True
            print(f"Mouse button {button} pressed")

    def on_mouse_release_event(self, x, y, button):
        """Handle mouse release events."""
        if button < len(self.mouse_buttons):
            self.mouse_buttons[button] = False
            print(f"Mouse button {button} released")

    def on_mouse_drag_event(self, x, y, dx, dy):
        """Handle mouse drag for camera control."""
        if not self.use_colmap_camera:
            # Check which button is being dragged (moderngl-window provides this info)
            # Left button (button 1) - rotate around target
            if self.mouse_buttons[1]:
                self._rotate_camera(dx, dy)
                print("Camera rotated (fallback)")
            elif self.mouse_buttons[2]:
                self._pan_camera(dx, dy)
                print("Camera panned (fallback)")

    def _rotate_camera(self, dx, dy):
        """Rotate camera around target using spherical coordinates."""
        sensitivity = 0.01
        
        # Calculate current camera direction
        direction = self.camera_pos - self.camera_target
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            # Spherical coordinates rotation
            theta = np.arctan2(direction[0], direction[2])
            phi = np.arccos(np.clip(direction[1] / distance, -1, 1))
            
            theta -= dx * sensitivity
            phi = np.clip(phi - dy * sensitivity, 0.1, np.pi - 0.1)
            
            # Convert back to cartesian
            self.camera_pos[0] = self.camera_target[0] + distance * np.sin(phi) * np.sin(theta)
            self.camera_pos[1] = self.camera_target[1] + distance * np.cos(phi)
            self.camera_pos[2] = self.camera_target[2] + distance * np.sin(phi) * np.cos(theta)

    def _pan_camera(self, dx, dy):
        """Pan camera by translating both position and target."""
        # Calculate distance-based sensitivity for consistent panning
        distance = np.linalg.norm(self.camera_pos - self.camera_target)
        sensitivity = distance * 0.001  # Scale with distance for consistent screen-space movement
        
        # Calculate camera's local coordinate system using pyrr utilities
        forward = pyrr.vector3.normalise(self.camera_target - self.camera_pos)
        right = pyrr.vector3.normalise(pyrr.vector3.cross(forward, self.camera_up))
        up = pyrr.vector3.normalise(pyrr.vector3.cross(right, forward))
        
        # Pan in screen space directions
        translation = dx * sensitivity * right - dy * sensitivity * up
        self.camera_pos += translation
        self.camera_target += translation

    def on_mouse_scroll_event(self, x_offset, y_offset):
        """Handle mouse scroll for zoom."""
        if not self.use_colmap_camera:
            direction = self.camera_pos - self.camera_target
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                
                zoom_factor = 1.0 + y_offset * self.zoom_sensitivity
                new_distance = max(0.1, distance * zoom_factor)  # Prevent going too close
                self.camera_pos = self.camera_target + direction * new_distance
                print(f"Camera zoomed, distance: {new_distance:.2f}")

    def on_key_event(self, key, action, modifiers):
        """Handle keyboard events."""
        print(f"Key event: key={key}, action={action}")  # Debug print
        
        if action == self.wnd.keys.ACTION_PRESS:
            print(f"Key pressed: {key}")  # Debug print
            
            if key == self.wnd.keys.SPACE:
                self.use_colmap_camera = not self.use_colmap_camera
                mode = 'COLMAP' if self.use_colmap_camera else 'MANUAL'
                print(f"*** Switched to {mode} camera mode ***")

                
            elif key == self.wnd.keys.LEFT:
                if self.use_colmap_camera:
                    self.current_camera_idx = max(0, self.current_camera_idx - 1)
                    #self._set_colmap_camera(self.current_camera_idx)
                    print(f"*** Switched to camera {self.current_camera_idx} ***")
                    self._print_debug_info()
                else:
                    print("Left arrow - switch to COLMAP mode first (SPACE)")
                    
            elif key == self.wnd.keys.RIGHT:
                if self.use_colmap_camera:
                    self.current_camera_idx = min(len(self.image_list) - 1, self.current_camera_idx + 1)
                    #self._set_colmap_camera(self.current_camera_idx)
                    print(f"*** Switched to camera {self.current_camera_idx} ***")
                    self._print_debug_info()
                else:
                    print("Right arrow - switch to COLMAP mode first (SPACE)")
                    
            elif key == self.wnd.keys.W:
                if not self.use_colmap_camera:
                    forward = self.camera_target - self.camera_pos
                    if np.linalg.norm(forward) > 0:
                        forward = forward / np.linalg.norm(forward)
                        self.camera_pos += forward * 0.5
                        self.camera_target += forward * 0.5
                        print("Moved forward")
                        
            elif key == self.wnd.keys.S:
                if not self.use_colmap_camera:
                    forward = self.camera_target - self.camera_pos
                    if np.linalg.norm(forward) > 0:
                        forward = forward / np.linalg.norm(forward)
                        self.camera_pos -= forward * 0.5
                        self.camera_target -= forward * 0.5
                        print("Moved backward")

def main():
    import sys
    
    # Parse our custom arguments before moderngl-window gets them
    our_args = [sys.argv[0]]  # Start with script name
    mglw_args = [sys.argv[0]]  # Start with script name
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--mesh-path" and i + 1 < len(sys.argv):
            our_args.extend([sys.argv[i], sys.argv[i + 1]])
            i += 2
        elif sys.argv[i] == "--colmap-path" and i + 1 < len(sys.argv):
            our_args.extend([sys.argv[i], sys.argv[i + 1]])
            i += 2
        else:
            mglw_args.append(sys.argv[i])
            i += 1
    
    # Parse our arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-path", type=str, required=True, help="Path to mesh file")
    parser.add_argument("--colmap-path", type=str, required=True, help="Path to COLMAP sparse folder")
    args = parser.parse_args(our_args[1:])  # Skip script name
    
    # Set class attributes for the window config
    MeshViewer.mesh_path = args.mesh_path
    MeshViewer.colmap_path = args.colmap_path
    
    # Replace sys.argv with only moderngl-window compatible args
    sys.argv = mglw_args
    
    mglw.run_window_config(MeshViewer)

if __name__ == "__main__":
    main()