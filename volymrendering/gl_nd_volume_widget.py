import ctypes
import numpy as np

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt

from OpenGL.GL import *

from tf_texture_builder_debug import build_tf_texture_2d_debug


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
uniform sampler2D u_tf2d;

uniform int u_debug_mode;
uniform int u_force_red;

uniform float u_rot_x;
uniform float u_rot_y;
uniform float u_zoom;
uniform float u_opacity_scale;

const int MAX_STEPS = 192; //#High 384, for better interactivness 192

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

void main()
{
    vec4 accum = vec4(0.0);

    mat3 R = rotY(u_rot_y) * rotX(u_rot_x);

    vec2 screen = (v_uv - vec2(0.5)) * u_zoom;

    for (int i = 0; i < MAX_STEPS; ++i)
    {
        float z = float(i) / float(MAX_STEPS - 1);

        vec3 local = vec3(screen.x, screen.y, z - 0.5);
        vec3 rotated = R * local;

        vec3 pos = rotated + vec3(0.5);

        if (
            pos.x < 0.0 || pos.x > 1.0 ||
            pos.y < 0.0 || pos.y > 1.0 ||
            pos.z < 0.0 || pos.z > 1.0
        )
        {
            continue;
        }

        float fx = texture(u_feature_x, pos).r;
        float fy = texture(u_feature_y, pos).r;

        vec4 sample_color;

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
            // TRUE 2D TRANSFER FUNCTION LOOKUP
            sample_color = texture(u_tf2d, vec2(fx, fy));

            // Base opacity scale for visibility
            sample_color.a *= 0.08;

            // Opacity correction for discrete ray samples
            sample_color.a = 1.0 - pow(1.0 - sample_color.a, u_opacity_scale);
        }

        accum.rgb += (1.0 - accum.a) * sample_color.rgb * sample_color.a;
        accum.a   += (1.0 - accum.a) * sample_color.a;

        if (accum.a > 0.98)
            break;
    }

    vec3 bg = vec3(0.08, 0.08, 0.08);
    vec3 out_rgb = mix(bg, accum.rgb, accum.a);

    fragColor = vec4(out_rgb, 1.0);
}
"""


class GLNDVolumeWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.program = None
        self.vao = None
        self.vbo = None

        self.tex_x = None
        self.tex_y = None
        self.tex_tf = None

        self.dims = None
        self.all_features = None
        self.feature_x = None
        self.feature_y = None
        self.widgets = []

        self.debug_mode = 0
        self.force_red = False
        self.pending_upload = False

        self.setFocusPolicy(Qt.StrongFocus)
        self.opacity_scale = 1.0

        self.rot_x = 0.0
        self.rot_y = 0.0
        self.zoom = 1.0
        self.last_mouse_pos = None

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is None:
            return

        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()

        ROTATION_SPEED = 0.005  # try 0.005–0.2 range

        self.rot_x += dy * ROTATION_SPEED
        self.rot_y += dx * ROTATION_SPEED

        self.last_mouse_pos = event.pos()

        #print(f"[GL-ND] rotation: x={self.rot_x:.1f}, y={self.rot_y:.1f}")
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

    def set_feature_pair(self, feature_x, feature_y, widgets):
        same_features = (
            self.feature_x == feature_x and
            self.feature_y == feature_y and
            self.tex_x is not None and
            self.tex_y is not None
        )

        self.feature_x = feature_x
        self.feature_y = feature_y
        self.widgets = widgets

        if same_features:
            print("[GL-ND] Updating only TRUE 2D TF texture")
            self.makeCurrent()
            tf = build_tf_texture_2d_debug(self.widgets, size=256)

            if self.tex_tf:
                glDeleteTextures([self.tex_tf])

            self.tex_tf = self._upload_2d_texture(tf)
            self.doneCurrent()
            self.update()
        else:
            print("[GL-ND] Feature pair changed: uploading feature volumes + TF")
            self.pending_upload = True
            self.update()

    def initializeGL(self):
        print("\n[GL-ND] initializeGL")
        print(f"[GL-ND] Vendor:   {glGetString(GL_VENDOR).decode()}")
        print(f"[GL-ND] Renderer: {glGetString(GL_RENDERER).decode()}")
        print(f"[GL-ND] Version:  {glGetString(GL_VERSION).decode()}")

        self.program = self._create_program(VERT_SHADER, FRAG_SHADER)
        self._create_quad()

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

        if not self.tex_x or not self.tex_y or not self.tex_tf:
            return

        glUseProgram(self.program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.tex_x)
        glUniform1i(glGetUniformLocation(self.program, "u_feature_x"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_3D, self.tex_y)
        glUniform1i(glGetUniformLocation(self.program, "u_feature_y"), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.tex_tf)
        glUniform1i(glGetUniformLocation(self.program, "u_tf2d"), 2)

        glUniform1i(glGetUniformLocation(self.program, "u_debug_mode"), self.debug_mode)
        glUniform1i(glGetUniformLocation(self.program, "u_force_red"), 1 if self.force_red else 0)
        #Add own zoom
        glUniform1f(glGetUniformLocation(self.program, "u_zoom"), self.zoom)
        glUniform1f(glGetUniformLocation(self.program, "u_rot_x"), self.rot_x)
        glUniform1f(glGetUniformLocation(self.program, "u_rot_y"), self.rot_y)
        glUniform1f(glGetUniformLocation(self.program, "u_opacity_scale"), self.opacity_scale)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

        glUseProgram(0)

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

        tf = build_tf_texture_2d_debug(self.widgets, size=256)
        self.tex_tf = self._upload_2d_texture(tf)

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
        for tex in [self.tex_x, self.tex_y, self.tex_tf]:
            if tex:
                glDeleteTextures([tex])

        self.tex_x = None
        self.tex_y = None
        self.tex_tf = None

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