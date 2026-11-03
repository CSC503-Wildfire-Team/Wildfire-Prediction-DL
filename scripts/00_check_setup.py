"""Sanity check for the WildfireSpreadTS + AlphaEarth environment."""
import importlib

for mod in ["numpy", "pandas", "rasterio", "h5py", "torch", "torchvision", "ee", "tqdm"]:
    try:
        m = importlib.import_module(mod)
        print(f"{mod:12s} OK  {getattr(m, '__version__', '')}")
    except Exception as e:                      # noqa: BLE001
        print(f"{mod:12s} MISSING  ({e})")

try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception:                               # noqa: BLE001
    pass