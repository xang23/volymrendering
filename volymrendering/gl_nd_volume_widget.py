import ctypes
import numpy as np

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt

from OpenGL.GL import *


VERT_SHADER = """
#version 330 core

layout(location = 0) in vec2 in_pos;
out vec2 v_uv;

void main()
{
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


FRAG_SHADER = """
#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler3D u_feature_x;
uniform sampler3D u_feature_y;
uniform sampler3D u_intensity_volume;

uniform int u_debug_mode;
uniform int u_force_red;

uniform float u_rot_x;
uniform float u_rot_y;
uniform float u_zoom;

uniform float u_opacity_scale;
uniform float u_visibility_boost;
uniform float u_sampling_rate;
uniform int u_active_steps;

// GPU widget transfer function
const int MAX_WIDGETS = 32;

uniform int u_num_widgets;

// x=center_x, y=center_y, z=size_x, w=size_y
uniform vec4 u_widget_params[MAX_WIDGETS];

// r,g,b,opacity
uniform vec4 u_widget_color[MAX_WIDGETS];

// 0=gaussian, 1=rectangular, 2=triangular, 3=ellipsoid, 4=diamond
uniform int u_widget_type[MAX_WIDGETS];

// 0=max, 1=add, 2=multiply
uniform int u_widget_blend[MAX_WIDGETS];

// 0=gaussian, 1=linear, 2=constant, 3=power2, 4=power3
uniform int u_widget_falloff[MAX_WIDGETS];

const int MAX_STEPS = 384;

mat3 rotX(float a)
{
    float c = cos(a);
    float s = sin(a);

    return mat3(
        1.0, 0.0, 0.0,
        0.0, c, -s,
        0.0, s, c
    );
}

mat3 rotY(float a)
{
    float c = cos(a);
    float s = sin(a);

    return mat3(
        c, 0.0, s,
        0.0, 1.0, 0.0,
        -s, 0.0, c
    );
}

float applyFalloff(float distance, int falloff_type)
{
    if (falloff_type == 0)
    {
        return exp(-(distance * distance) / 2.0);
    }
    else if (falloff_type == 1)
    {
        return max(0.0, 1.0 - distance);
    }
    else if (falloff_type == 2)
    {
        return distance <= 1.0 ? 1.0 : 0.0;
    }
    else if (falloff_type == 3)
    {
        return max(0.0, 1.0 - distance * distance);
    }
    else if (falloff_type == 4)
    {
        return max(0.0, 1.0 - distance * distance * distance);
    }

    return max(0.0, 1.0 - distance);
}

float widgetOpacity(float x255, float y255, int i)
{
    vec4 p = u_widget_params[i];

    float cx = p.x;
    float cy = p.y;
    float sx = max(p.z, 1.0);
    float sy = max(p.w, 1.0);

    int type = u_widget_type[i];
    int falloff = u_widget_falloff[i];

    float dx;
    float dy;
    float distance;

    if (type == 0)
    {
        // Gaussian: sx/sy are standard deviations.
        dx = (x255 - cx) / sx;
        dy = (y255 - cy) / sy;
        distance = sqrt(dx * dx + dy * dy);

        if (distance > 3.0)
            return 0.0;

        return applyFalloff(distance, falloff);
    }
    else if (type == 1)
    {
        // Rectangular: sx/sy are full width/height.
        dx = abs(x255 - cx) / max(sx * 0.5, 1.0);
        dy = abs(y255 - cy) / max(sy * 0.5, 1.0);
        distance = max(dx, dy);

        return applyFalloff(distance, falloff);
    }
    else if (type == 2)
    {
        // Symmetric triangle/pyramid.
        dx = abs(x255 - cx) / max(sx * 0.5, 1.0);
        dy = abs(y255 - cy) / max(sy * 0.5, 1.0);
        distance = dx + dy;

        if (distance > 1.0)
            return 0.0;

        return applyFalloff(distance, falloff);
    }
    else if (type == 3)
    {
        // Ellipsoid: sx/sy are radii.
        dx = (x255 - cx) / sx;
        dy = (y255 - cy) / sy;
        distance = sqrt(dx * dx + dy * dy);

        if (distance > 1.0)
            return 0.0;

        return applyFalloff(distance, falloff);
    }
    else if (type == 4)
    {
        // Diamond: sx/sy are full width/height.
        dx = abs(x255 - cx) / max(sx * 0.5, 1.0);
        dy = abs(y255 - cy) / max(sy * 0.5, 1.0);
        distance = dx + dy;

        if (distance > 1.0)
            return 0.0;

        return applyFalloff(distance, falloff);
    }

    return 0.0;
}

vec3 estimateNormal(vec3 pos)
{
    float e = 1.0 / 128.0;

    float xp = texture(u_intensity_volume, pos + vec3(e, 0.0, 0.0)).r;
    float xm = texture(u_intensity_volume, pos - vec3(e, 0.0, 0.0)).r;

    float yp = texture(u_intensity_volume, pos + vec3(0.0, e, 0.0)).r;
    float ym = texture(u_intensity_volume, pos - vec3(0.0, e, 0.0)).r;

    float zp = texture(u_intensity_volume, pos + vec3(0.0, 0.0, e)).r;
    float zm = texture(u_intensity_volume, pos - vec3(0.0, 0.0, e)).r;

    vec3 g = vec3(xp - xm, yp - ym, zp - zm);

    if (length(g) < 1e-5)
        return vec3(0.0, 0.0, 1.0);

    return normalize(g);
}

vec4 evaluateTransferFunction(float fx, float fy)
{
    float x255 = clamp(fx, 0.0, 1.0) * 255.0;
    float y255 = clamp(fy, 0.0, 1.0) * 255.0;

    vec3 out_rgb = vec3(0.0);
    float out_a = 0.0;

    bool any_multiply = false;
    float multiply_a = 1.0;
    vec3 multiply_rgb = vec3(1.0);

    for (int i = 0; i < MAX_WIDGETS; ++i)
    {
        if (i >= u_num_widgets)
            break;

        float local_a = widgetOpacity(x255, y255, i);
        local_a *= u_widget_color[i].a;
        local_a = clamp(local_a, 0.0, 1.0);

        vec3 local_rgb = u_widget_color[i].rgb;
        int blend = u_widget_blend[i];

        if (blend == 0)
        {
            // max blend
            if (local_a > out_a)
            {
                out_a = local_a;
                out_rgb = local_rgb;
            }
        }
        else if (blend == 1)
        {
            // additive alpha blend in TF space
            float new_a = clamp(out_a + local_a, 0.0, 1.0);

            if (new_a > 0.00001)
            {
                out_rgb = (out_rgb * out_a + local_rgb * local_a) / new_a;
            }

            out_a = new_a;
        }
        else if (blend == 2)
        {
            // multiply as mask-like combination
            any_multiply = true;
            multiply_a *= (1.0 - local_a);
            multiply_rgb *= mix(vec3(1.0), local_rgb, local_a);
        }
    }

    if (any_multiply)
    {
        float mask_a = 1.0 - multiply_a;

        if (mask_a > out_a)
        {
            out_a = mask_a;
            out_rgb = multiply_rgb;
        }
    }

    return vec4(out_rgb, out_a);
}

void main()
{
    vec4 accum = vec4(0.0);

    mat3 R = rotX(u_rot_x) * rotY(u_rot_y);
    vec2 screen = (v_uv - vec2(0.5)) * u_zoom;

    int steps = clamp(u_active_steps, 1, MAX_STEPS);

    // Orthographic ray direction (camera forward)
    vec3 ray_dir = normalize(R * vec3(0.0, 0.0, -1.0));

    // Ray origin per pixel (image plane mapped into volume space)
    vec3 ray_origin = vec3(screen.x, screen.y, 0.0);
    ray_origin = R * ray_origin + vec3(0.5);

    // Ray spans through volume center
    float tmin = -1.2;
    float tmax =  1.2;

    for (int i = 0; i < MAX_STEPS; ++i)
    {
        if (i >= steps)
            break;

        float t = mix(tmin, tmax, float(i) / float(max(steps - 1, 1)));
        vec3 pos = ray_origin + t * ray_dir;

        // Skip samples outside volume (avoids fake scaling!)
        if (pos.x < 0.0 || pos.x > 1.0 ||
            pos.y < 0.0 || pos.y > 1.0 ||
            pos.z < 0.0 || pos.z > 1.0)
        {
            continue;
        }

        float fx = texture(u_feature_x, pos).r;
        float fy = texture(u_feature_y, pos).r;

        vec4 sample_color;

        // Debug modes
        if (u_debug_mode == 1)
        {
            fragColor = vec4(fx, fx, fx, 1.0);
            return;
        }
        else if (u_debug_mode == 2)
        {
            fragColor = vec4(fy, fy, fy, 1.0);
            return;
        }
        else if (u_debug_mode == 3)
        {
            fragColor = vec4(fx, fy, 0.0, 1.0);
            return;
        }
        else if (u_force_red == 1)
        {
            sample_color = vec4(1.0, 0.0, 0.0, 0.035);
        }
        else
        {
            // Transfer function
            sample_color = evaluateTransferFunction(fx, fy);

            // Skip empty space (performance + clarity)
            if (sample_color.a < 0.01)
                continue;

            // Normal + lighting
            vec3 N = estimateNormal(pos);
            if (dot(N, ray_dir) > 0.0)
                N = -N;

            vec3 view_dir = -ray_dir;
            vec3 L = view_dir;

            float diffuse = max(dot(N, L), 0.0);

            float ambient = 0.45;
            float diffuse_weight = 0.55;
            float lighting = ambient + diffuse_weight * diffuse;

            sample_color.rgb *= lighting;
            sample_color.rgb = clamp(sample_color.rgb, 0.0, 1.0);

            // Opacity correction
            float correction = 2.0 / max(u_sampling_rate, 0.01);
            correction *= u_opacity_scale;

            sample_color.a = 1.0 - pow(1.0 - sample_color.a, correction);
        }

        // Compositing
        accum.rgb += (1.0 - accum.a) * sample_color.rgb * sample_color.a;
        accum.a   += (1.0 - accum.a) * sample_color.a;

        // Early exit
        if (accum.a > 0.95)
            break;
    }

    // Background blend
    vec3 bg = vec3(0.08, 0.08, 0.08);
    vec3 out_rgb = mix(bg, accum.rgb, accum.a);

    fragColor = vec4(out_rgb, 1.0);
}
"""

OVERLAY_VERT_SHADER = """
#version 330 core

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec3 in_color;

out vec3 v_color;

void main()
{
    v_color = in_color;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

OVERLAY_FRAG_SHADER = """
#version 330 core

in vec3 v_color;
out vec4 fragColor;

void main()
{
    fragColor = vec4(v_color, 1.0);
}
"""


class GLNDVolumeWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.program = None
        self.vao = None
        self.vbo = None

        self.overlay_program = None
        self.overlay_vao = None
        self.overlay_vbo = None

        self.tex_x = None
        self.tex_y = None
        self.tex_tf = None
        self.tex_intensity = None

        self.dims = None
        self.all_features = None
        self.feature_x = None
        self.feature_y = None
        self.widgets = []

        self.debug_mode = 0
        self.force_red = False
        self.pending_upload = False

        self.tf_texture_size = 256
        self.interactive_mode = False
        self.max_steps = 192

        self.setFocusPolicy(Qt.StrongFocus)
        self.opacity_scale = 1.0
        self.visibility_boost = 1.0
        self.sampling_rate = 2.0
        self.active_steps = 384

        self.rot_x = 0.0
        self.rot_y = 0.0
        self.zoom = 1.0
        self.last_mouse_pos = None
        self.show_bounding_box = True
        self.show_axes = True

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is None:
            return

        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()

        ROTATION_SPEED = 0.003

        if event.modifiers() & Qt.ShiftModifier:
            # Shift + drag = horizontal rotation only
            self.rot_y += dx * ROTATION_SPEED

        elif event.modifiers() & Qt.ControlModifier:
            # Ctrl + drag = vertical rotation only
            self.rot_x += dy * ROTATION_SPEED

        else:
            # Normal drag = free rotation
            self.rot_x += dy * ROTATION_SPEED
            self.rot_y += dx * ROTATION_SPEED

        self.last_mouse_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()

        if delta > 0:
            self.zoom *= 0.9
        else:
            self.zoom *= 1.1

        self.zoom = max(0.3, min(3.0, self.zoom))

        #print(f"[GL-ND] zoom={self.zoom:.2f}")
        self.update()

    def set_volume_data(self, dims, all_features, feature_x, feature_y, widgets):
        self.dims = dims
        self.all_features = all_features
        self.feature_x = feature_x
        self.feature_y = feature_y
        self.widgets = widgets
        self.pending_upload = True

        print("\n[GL-ND] set_volume_data")
        print(f"[GL-ND] dims={dims}")
        print(f"[GL-ND] feature_x={feature_x}")
        print(f"[GL-ND] feature_y={feature_y}")
        print(f"[GL-ND] widgets={len(widgets)}")
        print(f"[GL-ND] available features={list(all_features.keys())}")

        self.update()

    def set_feature_pair(self, feature_x, feature_y, widgets, verbose=False, rebuild_tf=False):
        same_features = (
            self.feature_x == feature_x and
            self.feature_y == feature_y and
            self.tex_x is not None and
            self.tex_y is not None
        )

        self.feature_x = feature_x
        self.feature_y = feature_y
        self.widgets = widgets
        print(f"[GL AXES] shader fx = {feature_x}, shader fy = {feature_y}")

        if same_features:
            # GPU widget path: no TF texture rebuild needed.
            self.update()
        else:
            print("[GL-ND] Feature pair changed: uploading feature volumes")
            self.pending_upload = True
            self.update()


    def initializeGL(self):
        print("\n[GL-ND] initializeGL")
        print(f"[GL-ND] Vendor:   {glGetString(GL_VENDOR).decode()}")
        print(f"[GL-ND] Renderer: {glGetString(GL_RENDERER).decode()}")
        print(f"[GL-ND] Version:  {glGetString(GL_VERSION).decode()}")

        self.program = self._create_program(VERT_SHADER, FRAG_SHADER)
        self._create_quad()
        self.overlay_program = self._create_program(OVERLAY_VERT_SHADER, OVERLAY_FRAG_SHADER)
        self._create_overlay_buffers()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.pending_upload = True

    def resizeGL(self, w, h):
        glViewport(0, 0, max(1, w), max(1, h))
        print(f"[GL-ND] resizeGL {w}x{h}")

    def paintGL(self):
        if self.pending_upload:
            self._upload_current_data()
            self.pending_upload = False

        glClearColor(0.04, 0.04, 0.04, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        if not self.tex_x or not self.tex_y or not self.tex_intensity:
            return

        glUseProgram(self.program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.tex_x)
        glUniform1i(glGetUniformLocation(self.program, "u_feature_x"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_3D, self.tex_y)
        glUniform1i(glGetUniformLocation(self.program, "u_feature_y"), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_3D, self.tex_intensity)
        glUniform1i(glGetUniformLocation(self.program, "u_intensity_volume"), 2)

        glUniform1i(glGetUniformLocation(self.program, "u_debug_mode"), self.debug_mode)
        glUniform1i(glGetUniformLocation(self.program, "u_force_red"), 1 if self.force_red else 0)
        #Add own zoom
        glUniform1f(glGetUniformLocation(self.program, "u_zoom"), self.zoom)
        glUniform1f(glGetUniformLocation(self.program, "u_rot_x"), self.rot_x)
        glUniform1f(glGetUniformLocation(self.program, "u_rot_y"), self.rot_y)
        glUniform1f(glGetUniformLocation(self.program, "u_opacity_scale"), self.opacity_scale)
        glUniform1f(glGetUniformLocation(self.program, "u_visibility_boost"), self.visibility_boost)
        glUniform1f(glGetUniformLocation(self.program, "u_sampling_rate"), self.sampling_rate)
        glUniform1i(glGetUniformLocation(self.program, "u_active_steps"),int(self.active_steps))

        self._upload_widgets_to_shader()
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

        glUseProgram(0)
        if self.show_bounding_box or self.show_axes:
            self._draw_orientation_overlay()

    def _create_overlay_buffers(self):
        self.overlay_vao = glGenVertexArrays(1)
        self.overlay_vbo = glGenBuffers(1)

        glBindVertexArray(self.overlay_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.overlay_vbo)

        # Reserve space. Dynamic data uploaded every frame.
        glBufferData(GL_ARRAY_BUFFER, 1024 * 6 * 4, None, GL_DYNAMIC_DRAW)

        stride = 5 * 4  # x, y, r, g, b

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _draw_orientation_overlay(self):
        vertices = []

        def add_line(a, b, color):
            ax, ay = self._volume_to_screen(a)
            bx, by = self._volume_to_screen(b)

            r, g, bl = color

            vertices.extend([ax, ay, r, g, bl])
            vertices.extend([bx, by, r, g, bl])

        if self.show_bounding_box:
            corners = [
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            ]

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]

            for a, b in edges:
                add_line(corners[a], corners[b], (1.0, 1.0, 1.0))

        if self.show_axes:
            origin = (0.5, 0.5, 0.5)

            add_line(origin, (1.0, 0.5, 0.5), (1.0, 0.1, 0.1))  # X red
            add_line(origin, (0.5, 1.0, 0.5), (0.1, 1.0, 0.1))  # Y green
            add_line(origin, (0.5, 0.5, 1.0), (0.2, 0.4, 1.0))  # Z blue

        if not vertices:
            return

        data = np.asarray(vertices, dtype=np.float32)

        glDisable(GL_DEPTH_TEST)

        glUseProgram(self.overlay_program)

        glBindVertexArray(self.overlay_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.overlay_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        glLineWidth(1.0)
        glDrawArrays(GL_LINES, 0, len(vertices) // 5)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glUseProgram(0)

    def set_interactive_mode(self, enabled):
        self.interactive_mode = enabled

        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_0:
            self.debug_mode = 0
            self.force_red = False
            print("[GL-ND] Mode 0: true 2D TF")

        elif event.key() == Qt.Key_1:
            self.debug_mode = 1
            self.force_red = False
            print("[GL-ND] Mode 1: feature X")

        elif event.key() == Qt.Key_2:
            self.debug_mode = 2
            self.force_red = False
            print("[GL-ND] Mode 2: feature Y")

        elif event.key() == Qt.Key_3:
            self.debug_mode = 3
            self.force_red = False
            print("[GL-ND] Mode 3: joint feature color")

        elif event.key() == Qt.Key_R:
            self.force_red = not self.force_red
            print(f"[GL-ND] force_red={self.force_red}")

        self.update()

    def _upload_current_data(self):
        if self.all_features is None or self.dims is None:
            return

        if self.feature_x not in self.all_features:
            print(f"[GL-ND] ERROR: feature_x missing: {self.feature_x}")
            return

        if self.feature_y not in self.all_features:
            print(f"[GL-ND] ERROR: feature_y missing: {self.feature_y}")
            return

        print("\n[GL-ND] Uploading real feature volumes")

        x = self._normalize(self.all_features[self.feature_x], self.feature_x)
        y = self._normalize(self.all_features[self.feature_y], self.feature_y)

        x_vol = self._reshape_to_volume(x)
        y_vol = self._reshape_to_volume(y)

        self._delete_textures()

        self.tex_x = self._upload_3d_texture(x_vol, self.feature_x)
        self.tex_y = self._upload_3d_texture(y_vol, self.feature_y)

        intensity = self._normalize(self.all_features["Intensity"], "Intensity")
        intensity_vol = self._reshape_to_volume(intensity)
        self.tex_intensity = self._upload_3d_texture(intensity_vol, "Intensity lighting")

        print("[GL-ND] Upload complete")
        print("[GL-ND] Keys: 0=true 2D TF, 1=X feature, 2=Y feature, 3=joint color, R=red debug")

    def _normalize(self, data, name):
        arr = np.asarray(data, dtype=np.float32)
        mn = float(np.min(arr))
        mx = float(np.max(arr))

        print(f"[GL-ND] {name} raw range: {mn:.4f} / {mx:.4f}")

        if mx <= mn:
            print(f"[GL-ND] WARNING: {name} has zero range")
            return np.zeros_like(arr, dtype=np.float32)

        out = (arr - mn) / (mx - mn)

        print(f"[GL-ND] {name} normalized range: {out.min():.4f} / {out.max():.4f}")
        return out.astype(np.float32)

    def _reshape_to_volume(self, flat):
        w, h, d = self.dims
        expected = w * h * d

        if flat.size != expected:
            raise RuntimeError(
                f"[GL-ND] Feature size mismatch: expected {expected}, got {flat.size}"
            )

        # VTK/numpy scalar order is usually x-fastest, so reshape to z,y,x for OpenGL upload.
        vol = flat.reshape((d, h, w))
        return np.ascontiguousarray(vol.astype(np.float32))

    def _upload_3d_texture(self, vol, name):
        d, h, w = vol.shape

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, tex)

        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glTexImage3D(
            GL_TEXTURE_3D,
            0,
            GL_R32F,
            w,
            h,
            d,
            0,
            GL_RED,
            GL_FLOAT,
            vol,
        )

        glBindTexture(GL_TEXTURE_3D, 0)

        print(f"[GL-ND] Uploaded 3D texture '{name}': id={tex}, shape={vol.shape}")
        return tex

    def _upload_2d_texture(self, tf):
        h, w, _ = tf.shape

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA32F,
            w,
            h,
            0,
            GL_RGBA,
            GL_FLOAT,
            tf,
        )

        glBindTexture(GL_TEXTURE_2D, 0)

        print(f"[GL-ND] Uploaded TRUE 2D TF texture: id={tex}, shape={tf.shape}")
        return tex

    def _delete_textures(self):
        for tex in [self.tex_x, self.tex_y, self.tex_tf, self.tex_intensity]:
            if tex:
                glDeleteTextures([tex])

        self.tex_x = None
        self.tex_y = None
        self.tex_tf = None
        self.tex_intensity = None

    def _create_quad(self):
        quad = np.array(
            [
                -1.0, -1.0,
                 1.0, -1.0,
                -1.0,  1.0,
                 1.0,  1.0,
            ],
            dtype=np.float32,
        )

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _compile_shader(self, src, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, src)
        glCompileShader(shader)

        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(shader).decode())

        return shader

    def _create_program(self, vs_src, fs_src):
        vs = self._compile_shader(vs_src, GL_VERTEX_SHADER)
        fs = self._compile_shader(fs_src, GL_FRAGMENT_SHADER)

        program = glCreateProgram()
        glAttachShader(program, vs)
        glAttachShader(program, fs)
        glLinkProgram(program)

        if not glGetProgramiv(program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(program).decode())

        glDeleteShader(vs)
        glDeleteShader(fs)

        print("[GL-ND] Shader program compiled")
        return program

    def _widget_type_id(self, widget):
        name = widget.widget_type.value

        if name == "gaussian":
            return 0
        if name == "rectangular":
            return 1
        if name == "triangular":
            return 2
        if name == "ellipsoid":
            return 3
        if name == "diamond":
            return 4

        return 0


    def _blend_id(self, widget):
        blend = getattr(widget, "blend_mode", "max")

        if blend == "max":
            return 0
        if blend == "add":
            return 1
        if blend == "multiply":
            return 2

        return 0


    def _falloff_id(self, widget):
        falloff = getattr(widget, "falloff_type", "linear")

        if falloff == "gaussian":
            return 0
        if falloff == "linear":
            return 1
        if falloff == "constant":
            return 2
        if falloff == "power2":
            return 3
        if falloff == "power3":
            return 4

        return 1


    def _widget_size_params(self, widget):
        name = widget.widget_type.value

        if name == "gaussian":
            sx = getattr(widget, "intensity_std", 20.0)
            sy = getattr(widget, "gradient_std", 20.0)

        elif name == "rectangular":
            sx = getattr(widget, "intensity_width", 50.0)
            sy = getattr(widget, "gradient_height", 50.0)

        elif name == "triangular":
            sx = getattr(widget, "intensity_width", 50.0)
            sy = getattr(widget, "gradient_height", 50.0)

        elif name == "ellipsoid":
            sx = getattr(widget, "intensity_radius", 25.0)
            sy = getattr(widget, "gradient_radius", 25.0)

        elif name == "diamond":
            sx = getattr(widget, "intensity_width", 50.0)
            sy = getattr(widget, "gradient_height", 50.0)

        else:
            sx = 30.0
            sy = 30.0

        return float(sx), float(sy)


    def _upload_widgets_to_shader(self):
        max_widgets = 32
        widgets = self.widgets[:max_widgets]

        glUniform1i(
            glGetUniformLocation(self.program, "u_num_widgets"),
            len(widgets)
        )

        params = np.zeros((max_widgets, 4), dtype=np.float32)
        colors = np.zeros((max_widgets, 4), dtype=np.float32)
        types = np.zeros((max_widgets,), dtype=np.int32)
        blends = np.zeros((max_widgets,), dtype=np.int32)
        falloffs = np.zeros((max_widgets,), dtype=np.int32)

        for i, widget in enumerate(widgets):
            sx, sy = self._widget_size_params(widget)

            params[i] = [
                float(widget.center_intensity),
                float(widget.center_gradient),
                sx,
                sy,
            ]

            r, g, b = widget.color
            colors[i] = [
                float(r),
                float(g),
                float(b),
                float(widget.opacity),
            ]

            types[i] = self._widget_type_id(widget)
            blends[i] = self._blend_id(widget)
            falloffs[i] = self._falloff_id(widget)

        glUniform4fv(
            glGetUniformLocation(self.program, "u_widget_params"),
            max_widgets,
            params
        )

        glUniform4fv(
            glGetUniformLocation(self.program, "u_widget_color"),
            max_widgets,
            colors
        )

        glUniform1iv(
            glGetUniformLocation(self.program, "u_widget_type"),
            max_widgets,
            types
        )

        glUniform1iv(
            glGetUniformLocation(self.program, "u_widget_blend"),
            max_widgets,
            blends
        )

        glUniform1iv(
            glGetUniformLocation(self.program, "u_widget_falloff"),
            max_widgets,
            falloffs
        )

    def _volume_to_screen(self, p):
        """
        Project volume-space point [0,1]^3 to screen NDC [-1,1]^2
        using the inverse of the raymarch sampling transform.

        Shader does:
            local = vec3(screen.x, screen.y, z - 0.5)
            rotated = R * local
            pos = rotated + 0.5

        Therefore for overlay:
            local = inverse(R) * (pos - 0.5)
            screen = local.xy / zoom
        """
        p = np.asarray(p, dtype=np.float32)

        centered = p - np.array([0.5, 0.5, 0.5], dtype=np.float32)

        cx = np.cos(self.rot_x)
        sx = np.sin(self.rot_x)
        cy = np.cos(self.rot_y)
        sy = np.sin(self.rot_y)

        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cx, -sx],
                [0.0, sx, cx],
            ],
            dtype=np.float32,
        )

        ry = np.array(
            [
                [cy, 0.0, sy],
                [0.0, 1.0, 0.0],
                [-sy, 0.0, cy],
            ],
            dtype=np.float32,
        )

        R = ry @ rx

        # inverse of a pure rotation is transpose
        local = R.T @ centered

        x_ndc = (local[0] / max(self.zoom, 1e-6)) * 2.0
        y_ndc = (local[1] / max(self.zoom, 1e-6)) * 2.0

        return [float(x_ndc), float(y_ndc)]
