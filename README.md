# GEMpRF Demo

This repository provides a minimal demo and utilities for running the GEMpRF analysis
pipeline on a small example dataset. It includes an example configuration, demo
stimuli and helper utilities to make running the demo easier on different systems.

This README explains the project layout, how to prepare the example configuration,
how to check GPU capacity before running heavy computations, and how to run the demo.

**Project layout**
- `run_gemprf_demo.py`: Example top-level runner that updates
	the configuration then launches GEMpRF with `gp.run(CONFIG_FILEPATH)`.
- `sample_configs/`: Example XML configuration files. The demo runner updates paths
	in the XML so they point to the local example data.
- `example_data/`: Small example BIDS-like data and stimuli used for the demo.
- `utils/auto_path.py`: Helper that updates the XML configuration file with local
	`example_data` paths (extracted from the original demo script).
- `utils/gpu_info.py`: Utility to detect NVIDIA GPUs, summarize available memory (in GB),
	and analyze whether the requested GEMpRF model (grid + refine) will fit in GPU memory.

**Why these utilities exist**
- The demo runner automatically updates config file paths so the example configuration
	can be used out-of-the-box without manual editing.
- GPU memory assessment prevents starting a long-running GPU job that would fail due
	to insufficient memory. If multiple GPUs are available the tool assumes the model
	signals are split evenly across GPUs and computes per-GPU requirements.

Quick start (Windows / PowerShell)

```

python example-001_purpose-analyse-fmriprep-data.py

```

What `utils/gpu_info.analyze_gpus` does
- Detects NVIDIA GPUs using `pynvml` (preferred) or `nvidia-smi` as fallback.
- Computes required memory for the configured model using:
	- model grid shape (default 151x151x16)
	- number of derivative sets (default 3)
	- stimulus dimensions (default 101x101 for the demo)
	- data type (default `float64`)
- Divides the total required memory evenly across detected GPUs and reports
	whether each GPU can hold its share (with a configurable safety factor).

Example: programmatic usage

```python
from utils.gpu_info import analyze_gpus

res = analyze_gpus(
		grid_shape=(151,151,16),        # model grid
		stimulus_shape=(101,101),       # demo stimulus
		dtype='float64',
		refine_multiplier=3,
		safety_factor=1.5,
)

print(res['summary'])
if res.get('warning'):
		print('WARNING:', res['warning'])
```

Notes & recommendations
- If `analyze_gpus` reports insufficient memory, either run on a machine with larger
	GPU memory, target a single capable GPU, or reduce the grid/stimulus sizes in the
	XML configuration for testing.
- The utilities assume NVIDIA GPUs; no automatic support for AMD GPUs is included.
- The GPU query requires appropriate drivers and permissions to call `nvidia-smi` or
	`pynvml`.

Contributing / extending
- Add tests
- Add other configuration XMLs and describe them the `config_library.py` to allow demo for other conditions.