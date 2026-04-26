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
uniform float u_step_scale;
uniform float u_visibility_boost;
uniform float u_opacity_scale;

const int MAX_STEPS = 256;

void main()
{
    vec4 accum = vec4(0.0);

    // Direct orthographic ray through volume.
    // No box intersection, no divide-by-zero risk.
    for (int i = 0; i < MAX_STEPS; ++i)
    {
        float z = float(i) / float(MAX_STEPS - 1);
        vec3 pos = vec3(v_uv.x, v_uv.y, z);

        float fx = texture(u_feature_x, pos).r;
        float fy = texture(u_feature_y, pos).r;

        vec4 sample_color;

        if (u_debug_mode == 1)
        {
            // Expected: horizontal left-to-right grayscale gradient.
            fragColor = vec4(fx, fx, fx, 1.0);
            return;
        }
        else if (u_debug_mode == 2)
        {
            // Expected: vertical bottom-to-top grayscale gradient.
            fragColor = vec4(fy, fy, fy, 1.0);
            return;
        }
        else if (u_debug_mode == 3)
        {
            // Expected: red varies with X, green varies with Y.
            fragColor = vec4(fx, fy, 0.0, 1.0);
            return;
        }
        else if (u_force_red == 1)
        {
            // Expected: visible red square/volume.
            sample_color = vec4(1.0, 0.0, 0.0, 0.04);
        }
        else
        {
            // TRUE 2D TF LOOKUP.
            sample_color = texture(u_tf2d, vec2(fx, fy));

            // Base per-sample opacity.
            sample_color.a *= 0.08;

            // Opacity correction for discrete ray integration.
            sample_color.a = 1.0 - pow(1.0 - sample_color.a, u_opacity_scale);

            // Optional exploration-only visibility boost.
            sample_color.a *= u_visibility_boost;
            sample_color.a = clamp(sample_color.a, 0.0, 1.0);
        }

        accum.rgb += (1.0 - accum.a) * sample_color.rgb * sample_color.a;
        accum.a   += (1.0 - accum.a) * sample_color.a;

        if (accum.a > 0.98)
            break;
    }

    // Make background visibly dark gray, not pure black.
    vec3 bg = vec3(0.08, 0.08, 0.08);
    vec3 out_rgb = mix(bg, accum.rgb, accum.a);

    fragColor = vec4(out_rgb, 1.0);
}
"""


class GLTrue2DTestWidget(QOpenGLWidget):
    def __init__(self, widgets, parent=None):
        super().__init__(parent)

        self.widgets = widgets
        self.program = None
        self.vao = None
        self.vbo = None

        self.tex_feature_x = None
        self.tex_feature_y = None
        self.tex_tf2d = None

        self.width_vox = 128
        self.height_vox = 128
        self.depth_vox = 128

        self.debug_mode = 0
        self.force_red = False
        self.step_scale = 1.0
        self.frame_counter = 0
        self.visibility_boost = 1.0

        self.setFocusPolicy(Qt.StrongFocus)

    def initializeGL(self):
        print("\n[GL] initializeGL")
        print(f"[GL] Vendor:   {glGetString(GL_VENDOR).decode()}")
        print(f"[GL] Renderer: {glGetString(GL_RENDERER).decode()}")
        print(f"[GL] Version:  {glGetString(GL_VERSION).decode()}")
        print(f"[GL] GLSL:     {glGetString(GL_SHADING_LANGUAGE_VERSION).decode()}")

        self.program = self._create_program(VERT_SHADER, FRAG_SHADER)
        self._create_fullscreen_quad()
        self._create_synthetic_feature_textures()
        self._create_tf_texture()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        print("[GL] Initialization complete")
        print("[TEST] Press keys:")
        print("       0 = true 2D TF")
        print("       1 = debug feature X")
        print("       2 = debug feature Y")
        print("       3 = debug joint feature color")
        print("       R = force red ray-marched volume")
        print("       + / - = change sampling step scale")

    def resizeGL(self, w, h):
        glViewport(0, 0, max(1, w), max(1, h))
        print(f"[GL] resizeGL: {w}x{h}")

    def paintGL(self):
        glClearColor(0.04, 0.04, 0.04, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.tex_feature_x)
        glUniform1i(glGetUniformLocation(self.program, "u_feature_x"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_3D, self.tex_feature_y)
        glUniform1i(glGetUniformLocation(self.program, "u_feature_y"), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.tex_tf2d)
        glUniform1i(glGetUniformLocation(self.program, "u_tf2d"), 2)

        glUniform1i(glGetUniformLocation(self.program, "u_debug_mode"), self.debug_mode)
        glUniform1i(glGetUniformLocation(self.program, "u_force_red"), 1 if self.force_red else 0)
        glUniform1f(glGetUniformLocation(self.program, "u_step_scale"), self.step_scale)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

        glUseProgram(0)

        glUniform1f(
            glGetUniformLocation(self.program, "u_visibility_boost"),
            self.visibility_boost
        )

        self.frame_counter += 1
        if self.frame_counter % 120 == 1:
            print(
                f"[RENDER] frame={self.frame_counter}, "
                f"mode={self.debug_mode}, "
                f"force_red={self.force_red}, "
                f"step_scale={self.step_scale:.2f}"
            )

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_0:
            self.debug_mode = 0
            self.force_red = False
            print("[MODE] True 2D TF lookup mode")

        elif key == Qt.Key_1:
            self.debug_mode = 1
            self.force_red = False
            print("[MODE] Feature X debug mode")

        elif key == Qt.Key_2:
            self.debug_mode = 2
            self.force_red = False
            print("[MODE] Feature Y debug mode")

        elif key == Qt.Key_3:
            self.debug_mode = 3
            self.force_red = False
            print("[MODE] Joint feature color debug mode")

        elif key == Qt.Key_R:
            self.force_red = not self.force_red
            print(f"[MODE] Force red volume = {self.force_red}")

        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.step_scale = max(0.25, self.step_scale * 0.8)
            print(f"[SAMPLING] step_scale={self.step_scale:.3f}")

        elif key == Qt.Key_Minus:
            self.step_scale = min(4.0, self.step_scale * 1.25)
            print(f"[SAMPLING] step_scale={self.step_scale:.3f}")

        self.update()

    def _compile_shader(self, src, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, src)
        glCompileShader(shader)

        ok = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if not ok:
            log = glGetShaderInfoLog(shader).decode()
            raise RuntimeError(f"Shader compile failed:\n{log}")

        return shader

    def _create_program(self, vert_src, frag_src):
        vs = self._compile_shader(vert_src, GL_VERTEX_SHADER)
        fs = self._compile_shader(frag_src, GL_FRAGMENT_SHADER)

        program = glCreateProgram()
        glAttachShader(program, vs)
        glAttachShader(program, fs)
        glLinkProgram(program)

        ok = glGetProgramiv(program, GL_LINK_STATUS)
        if not ok:
            log = glGetProgramInfoLog(program).decode()
            raise RuntimeError(f"Program link failed:\n{log}")

        glDeleteShader(vs)
        glDeleteShader(fs)

        print("[GL] Shader program compiled and linked")
        return program

    def _create_fullscreen_quad(self):
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
        glVertexAttribPointer(
            0,
            2,
            GL_FLOAT,
            GL_FALSE,
            0,
            ctypes.c_void_p(0),
        )

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        print(f"[GL] Fullscreen quad created: vao={self.vao}, vbo={self.vbo}")

    def _create_synthetic_feature_textures(self):
        print("\n[DATA] Creating synthetic feature volumes")
        print(f"[DATA] Dims: {self.width_vox} x {self.height_vox} x {self.depth_vox}")

        z, y, x = np.mgrid[
            0:self.depth_vox,
            0:self.height_vox,
            0:self.width_vox,
        ]

        fx = x.astype(np.float32) / float(self.width_vox - 1)
        fy = y.astype(np.float32) / float(self.height_vox - 1)

        # Add weak z modulation so you know ray marching is actually integrating volume.
        fx = np.clip(fx + 0.15 * np.sin(z / self.depth_vox * np.pi * 4.0), 0.0, 1.0)
        fy = np.clip(fy + 0.15 * np.cos(z / self.depth_vox * np.pi * 4.0), 0.0, 1.0)

        fx = np.ascontiguousarray(fx.astype(np.float32))
        fy = np.ascontiguousarray(fy.astype(np.float32))

        print(f"[DATA] Feature X range: {fx.min():.4f} / {fx.max():.4f}")
        print(f"[DATA] Feature Y range: {fy.min():.4f} / {fy.max():.4f}")
        print(f"[DATA] Center voxel fx/fy: {fx[64,64,64]:.4f}, {fy[64,64,64]:.4f}")

        self.tex_feature_x = self._upload_3d_texture(fx, "feature_x")
        self.tex_feature_y = self._upload_3d_texture(fy, "feature_y")

    def _upload_3d_texture(self, volume, label):
        depth, height, width = volume.shape

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
            width,
            height,
            depth,
            0,
            GL_RED,
            GL_FLOAT,
            volume,
        )

        glBindTexture(GL_TEXTURE_3D, 0)

        print(f"[GL] Uploaded 3D texture '{label}': id={tex}, shape={volume.shape}")
        return tex

    def _create_tf_texture(self):
        tf = build_tf_texture_2d_debug(self.widgets, size=256)

        h, w, _ = tf.shape

        self.tex_tf2d = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_tf2d)

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

        print(f"[GL] Uploaded TRUE 2D TF texture: id={self.tex_tf2d}, shape={tf.shape}")