import math
import numpy as np
import streamlit as st
from PIL import Image
from skimage import feature
from scipy.optimize import least_squares

# ---------- constants ----------
G = 6.67430e-11
C = 299_792_458.0
M_SUN = 1.98847e30
MPC_TO_M = 3.085677581e22
MICROARCSEC_TO_RAD = 1.0 / (206265.0e6)

def mass_from_theta_distance_solar(theta_microas: float, distance_mpc: float) -> float:
    theta_rad = theta_microas * MICROARCSEC_TO_RAD
    distance_m = distance_mpc * MPC_TO_M
    mass_kg = (theta_rad * (C**2) * distance_m) / (3.0 * math.sqrt(3.0) * G)
    return mass_kg / M_SUN

def fit_circle(points_xy: np.ndarray):
    x = points_xy[:, 0]
    y = points_xy[:, 1]

    cx0, cy0 = np.mean(x), np.mean(y)
    r0 = np.mean(np.sqrt((x - cx0) ** 2 + (y - cy0) ** 2))
    p0 = np.array([cx0, cy0, r0], dtype=float)

    def residuals(p):
        cx, cy, r = p
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r

    res = least_squares(residuals, p0, method="trf")
    cx, cy, r = res.x
    return float(cx), float(cy), float(abs(r))

def measure_shadow_diameter_px(img_gray: np.ndarray,
                               canny_sigma=2.0,
                               low=0.15,
                               high=0.35,
                               crop_frac=0.70):
    # edge detection
    edges = feature.canny(img_gray, sigma=canny_sigma, low_threshold=low, high_threshold=high)

    # center crop to avoid catching borders/scale bars
    h, w = img_gray.shape
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2

    mask = np.zeros_like(edges, dtype=bool)
    mask[y0:y0 + ch, x0:x0 + cw] = True

    selected = edges & mask
    ys, xs = np.where(selected)

    if len(xs) < 30:
        return None, edges, "Not enough edge points detected. Try a clearer image or adjust thresholds."

    points = np.column_stack([xs, ys])
    _, _, r = fit_circle(points)
    diameter_px = 2.0 * r
    return diameter_px, edges, f"Edge points used: {len(xs)}"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="ShadowScale", layout="wide")
st.title("ShadowScale")
st.caption("Upload a black-hole shadow image. Get diameter (pixels). Add scale + distance to estimate mass.")

uploaded = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])

left, right = st.columns(2)

with left:
    st.subheader("Settings")
    canny_sigma = st.slider("Edge sensitivity (sigma)", 0.5, 4.0, 2.0, 0.1)
    low = st.slider("Low threshold", 0.01, 0.50, 0.15, 0.01)
    high = st.slider("High threshold", 0.05, 0.80, 0.35, 0.01)

    st.subheader("Optional: Mass estimation")
    pixel_scale = st.number_input("Microarcseconds per pixel", min_value=0.0, value=0.0, step=0.01)
    distance_mpc = st.number_input("Distance (Mpc)", min_value=0.0, value=0.0, step=0.1)

if uploaded is None:
    st.info("Upload an image to start.")
else:
    img = Image.open(uploaded).convert("L")
    arr = np.array(img, dtype=float) / 255.0

    diameter_px, edges, msg = measure_shadow_diameter_px(arr, canny_sigma, low, high)

    with right:
        st.subheader("Preview")
        st.image(np.array(img), caption="Uploaded image", use_container_width=True)
        st.image((edges * 255).astype(np.uint8), caption="Detected edges", use_container_width=True)

    if diameter_px is None:
        st.error(msg)
    else:
        st.success(msg)
        st.metric("Measured shadow diameter", f"{diameter_px:.2f} px")

        if pixel_scale > 0:
            theta_microas = diameter_px * pixel_scale
            st.metric("Angular diameter", f"{theta_microas:.3f} microarcseconds")

            if distance_mpc > 0:
                mass_solar = mass_from_theta_distance_solar(theta_microas, distance_mpc)
                st.metric("Estimated mass", f"{mass_solar:.3e} solar masses")
            else:
                st.warning("Add distance (Mpc) to compute mass.")
        else:
            st.warning("Add microarcseconds-per-pixel scale to compute angular size and mass.")
