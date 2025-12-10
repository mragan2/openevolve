import os
import numpy as np

# Adjust this to the actual filename you see in Pantheon+_Data
PANTHEON_FILE = r"C:\Users\Michal\Downloads\DataRelease-main.zip\DataRelease-main"

# Where we want to write the simple 3-column file:
OUT_FILE = r"C:\Users\Michal\Documents\GitHub\openevolve\examples\mg_mcmc\data\pantheon_plus_sn.txt"

# Load the file, letting NumPy read named columns
data = np.genfromtxt(
    PANTHEON_FILE,
    names=True,        # use first line as header
    comments='#',      # ignore comment lines
    dtype=None,
    encoding=None
)

# Inspect available column names (run once to see what they are)
print("Columns:", data.dtype.names)

# You may need to adjust these names depending on the actual header
z = data["zCMB"]
mu = data["MU"]       # or e.g. data["DLMAG"] if that's the distance modulus
sigma_mu = data["MUERR"]

# Stack into 3 columns: z, mu, sigma_mu
out = np.vstack([z, mu, sigma_mu]).T

# Save as whitespace-delimited text
np.savetxt(OUT_FILE, out, fmt="%.8f")

print(f"Written {OUT_FILE} with shape {out.shape}")
