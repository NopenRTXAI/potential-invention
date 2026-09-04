# %% [markdown]
# TinyGPT 30M on Kaggle
#
# This notebook-style Python file trains the existing NumPy model.
# It uses Kaggle CPU because the model is NumPy-based; TPU/GPU require a JAX
# or PyTorch port. Upload the repository as a Kaggle Dataset, then set REPO.

# %% 1. Configuration
import os
import sys

# Change this if your Kaggle Dataset has a different mount name.
REPO = "/kaggle/input/codespaces-react"
OUTPUT = "/kaggle/working/chat_model"
HOURS = 8
EPOCHS = 2
BATCH = 16

assert os.path.isdir(os.path.join(REPO, "engine")), (
    "REPO is wrong. Add your project as a Kaggle Dataset and update REPO."
)
sys.path.insert(0, REPO)
print("Repository:", REPO)
print("Output:", OUTPUT)

# %% 2. Optional dependencies
import subprocess

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "numpy", "pyarrow"], check=True)

# %% 3. Start resumable training
# This runs independently of the local VS Code server. It resumes from a
# checkpoint if models/chat_ckpt exists in the uploaded project.
subprocess.run([
    sys.executable, os.path.join(REPO, "scripts", "kaggle_train.py"),
    "--repo", REPO, "--output", OUTPUT,
    "--hours", str(HOURS), "--epochs", str(EPOCHS),
    "--batch", str(BATCH),
], check=True)

# %% 4. Inspect progress or final result
import json
from pathlib import Path

status = Path(OUTPUT) / "status.txt"
if status.exists():
    print(status.read_text())
else:
    print("status.txt not created yet")

# %% 5. Download the trained model
# In the Kaggle right-side Output panel, download the chat_model folder.
# It contains model.npz, metadata.json, and status.txt.
print("Download folder:", OUTPUT)
print("Files:")
for path in sorted(Path(OUTPUT).glob("*")):
    print(path.name, path.stat().st_size, "bytes")
