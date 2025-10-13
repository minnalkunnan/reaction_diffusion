import numpy as np
import pandas as pd
import ast
import re
import matplotlib.pyplot as plt
from visualize import plot_one_frame

def analyze_pattern(a, dx=1.0, plot=False):
    N = len(a)
    freqs = np.fft.fftfreq(N, d=dx)
    power = np.abs(np.fft.fft(a))**2

    pos_mask = freqs > 0
    if not np.any(pos_mask):
        return np.nan, np.nan

    dominant_freq = freqs[pos_mask][np.argmax(power[pos_mask])]
    dominant_wavelength = 1 / dominant_freq if dominant_freq != 0 else np.inf

    if plot:
        plt.figure(figsize=(6, 3))
        plt.plot(freqs[pos_mask], power[pos_mask])
        plt.xlabel("Spatial frequency (1/unit length)")
        plt.ylabel("Power")
        plt.title(f"Fourier Spectrum (λ = {dominant_wavelength:.2f})")
        plt.tight_layout()
        plt.show()

    return dominant_freq, dominant_wavelength


def parse_list_string(s):
    """Safely convert a string like '[1,2,3]' or '[1 2 3]' to a list of floats."""
    if not isinstance(s, str):
        return None
    # remove brackets
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    # replace multiple spaces with a single comma
    s = re.sub(r"\s+", ",", s)
    # ensure commas between numbers
    s = s.replace(",,", ",")
    try:
        return np.array([float(x) for x in s.split(",") if x.strip() != ""])
    except Exception:
        return None


# === Load file ===
input_file = "batch_results_500-cells-V01.csv"
df = pd.read_csv(input_file)

dominant_freqs = []
dominant_wavelengths = []

for i, row in df.iterrows():
    a = parse_list_string(row["activator_final"])
    if a is None or len(a) < 3:
        print(f"Skipping row {i}: could not parse activator_final")
        dominant_freqs.append(np.nan)
        dominant_wavelengths.append(np.nan)
        continue

    if np.std(a) < 1e-6 or np.isnan(a).any():
        dominant_freqs.append(np.nan)
        dominant_wavelengths.append(np.nan)
        continue

    dom_freq, dom_lambda = analyze_pattern(a, dx=1.0, plot=False)
    dominant_freqs.append(dom_freq)
    dominant_wavelengths.append(dom_lambda)

df["dominant_freq"] = dominant_freqs
df["dominant_wavelength"] = dominant_wavelengths

df_out = df[["parameters", "dominant_freq", "dominant_wavelength"]]
df_out.to_csv("dominant_wavelengths_summary.csv", index=False)

print(df_out)
