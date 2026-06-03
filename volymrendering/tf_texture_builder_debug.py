import numpy as np


def build_tf_texture_2d_debug(widgets, size=256, verbose=False):
    tex = np.zeros((size, size, 4), dtype=np.float32)

    xs = np.linspace(0.0, 255.0, size, dtype=np.float32)
    ys = np.linspace(0.0, 255.0, size, dtype=np.float32)

    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            out_rgb = np.zeros(3, dtype=np.float32)
            out_a = 0.0

            for widget in widgets:
                local_a = float(widget.calculate_opacity(x, y))
                local_a = np.clip(local_a, 0.0, 1.0)

                if local_a <= 0.0:
                    continue

                local_rgb = np.array(widget.color, dtype=np.float32)
                blend = getattr(widget, "blend_mode", "max")

                # ---- MAX ----
                if blend == "max":
                    if local_a > out_a:
                        out_rgb = local_rgb
                        out_a = local_a

                # ---- ADD ----
                elif blend == "add":
                    new_a = min(1.0, out_a + local_a)

                    if new_a > 1e-6:
                        out_rgb = (
                            out_rgb * out_a + local_rgb * local_a
                        ) / new_a

                    out_a = new_a

                # ---- MULTIPLY (match shader mask logic) ----
                elif blend == "multiply":
                    # mask accumulation (matches shader idea)
                    out_a = 1.0 - (1.0 - out_a) * (1.0 - local_a)

                    # simple mix for color
                    out_rgb = out_rgb * (1.0 - local_a) + local_rgb * local_a

                else:
                    # fallback
                    if local_a > out_a:
                        out_rgb = local_rgb
                        out_a = local_a

            tex[yi, xi, :3] = np.clip(out_rgb, 0.0, 1.0)
            tex[yi, xi, 3] = np.clip(out_a, 0.0, 1.0)

    if verbose:
        alpha = tex[:, :, 3]
        print("[TF2D] Finished")
        print(f"[TF2D] Alpha min/max: {alpha.min():.4f} / {alpha.max():.4f}")

    return np.ascontiguousarray(tex)