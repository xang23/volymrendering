import numpy as np


def build_tf_texture_2d_debug(widgets, size=256, verbose=False):
    tex = np.zeros((size, size, 4), dtype=np.float32)

    if verbose:
        print("\n[TF2D] Building true 2D RGBA texture")
        print(f"[TF2D] Texture size: {size}x{size}")
        print(f"[TF2D] Widgets: {len(widgets)}")

    for wi, widget in enumerate(widgets):
        if verbose:
            print(
                f"[TF2D] Widget {wi}: "
                f"type={widget.widget_type.value}, "
                f"center=({widget.center_intensity:.1f},{widget.center_gradient:.1f}), "
                f"opacity={widget.opacity:.2f}, "
                f"color={widget.color}, "
                f"blend={widget.blend_mode}"
            )

        color = np.array(widget.color, dtype=np.float32)

        for y in range(size):
            for x in range(size):
                alpha = float(widget.calculate_opacity(x, y))
                alpha = np.clip(alpha, 0.0, 1.0)

                if widget.blend_mode == "add":
                    old_alpha = tex[y, x, 3]
                    new_alpha = min(1.0, old_alpha + alpha)

                    if new_alpha > 1e-6:
                        tex[y, x, :3] = (
                            tex[y, x, :3] * old_alpha + color * alpha
                        ) / new_alpha

                    tex[y, x, 3] = new_alpha

                elif widget.blend_mode == "multiply":
                    old_alpha = tex[y, x, 3]

                    # Multiply colors (darkening effect)
                    tex[y, x, :3] = tex[y, x, :3] * (1.0 - alpha) + (tex[y, x, :3] * color) * alpha

                    # Combine alpha conservatively
                    tex[y, x, 3] = old_alpha * (1.0 - alpha) + alpha

                else:  # MAX (default)
                    if alpha > tex[y, x, 3]:
                        tex[y, x, :3] = color
                        tex[y, x, 3] = alpha

    if verbose:
        alpha = tex[:, :, 3]

        print("[TF2D] Finished")
        print(f"[TF2D] Alpha min/max: {alpha.min():.4f} / {alpha.max():.4f}")
        print(
            f"[TF2D] Non-zero alpha texels > 0.01: "
            f"{np.sum(alpha > 0.01)} / {size * size}"
        )

        for px, py in [(64, 64), (128, 128), (192, 192), (64, 192), (192, 64)]:
            rgba = tex[py, px]
            print(f"[TF2D] Probe TF({px},{py}) = RGBA {rgba}")

    return np.ascontiguousarray(tex)