"""
"@Author  :   Siddharth Mittal",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2025, Medical University of Vienna",
"@Desc    :   Utility to analyze GPU availability and memory capacity for GEMpRF.",
"""


"""GPU information and capacity analysis utilities.

This module tries to detect NVIDIA GPUs using `pynvml` if available, otherwise
falls back to parsing `nvidia-smi` output. If neither is available it reports
that GPU info cannot be obtained.

The `analyze_gpus` function summarizes available GPUs and evaluates whether
they have enough memory to hold the model signals described by the user.
"""

import os
import subprocess
import shutil
import math
from typing import Tuple, List, Dict, Any
from utils.xml_utils import update_xml_value, create_coarse_grid_config

RED = "\033[91m"
ORANGE = "\033[38;5;208m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

def _bytes_for_dtype(dtype: str) -> int:
    import numpy as _np

    return int(_np.dtype(dtype).itemsize)


def _query_with_pynvml() -> List[Dict[str, Any]]:
    try:
        import pynvml
    except Exception:
        return []

    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h).decode('utf-8') if isinstance(pynvml.nvmlDeviceGetName(h), bytes) else str(pynvml.nvmlDeviceGetName(h))
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            total = int(mem.total)
            free = int(mem.free)
            gpus.append({"index": i, "name": name, "total_bytes": total, "free_bytes": free})
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return gpus
    except Exception:
        return []


def _query_with_nvidia_smi() -> List[Dict[str, Any]]:
    smi = shutil.which('nvidia-smi')
    if not smi:
        return []

    try:
        # query total and free memory in MiB
        cmd = [smi, '--query-gpu=memory.total,memory.free,name', '--format=csv,noheader,nounits']
        out = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        gpus = []
        for idx, ln in enumerate(lines):
            parts = [p.strip() for p in ln.split(',')]
            if len(parts) < 3:
                continue
            total_mib = float(parts[0])
            free_mib = float(parts[1])
            name = parts[2]
            gpus.append({"index": idx, "name": name, "total_bytes": int(total_mib * 1024**2), "free_bytes": int(free_mib * 1024**2)})
        return gpus
    except Exception:
        return []

def _msg_no_gpu():
    return (
         f"{RED}"
        "\n\n"
        "No NVIDIA GPUs detected on this system.\n\n"
        "GEM-pRF requires a CUDA-capable GPU for full-resolution model synthesis.\n\n"
        f"{RESET}"
    )

def _msg_all_gpus_filtered():
    return (
         f"{RED}"
        "\n\n"
        "No suitable GPU found. The available GPUs do not have enough free memory for this configuration.\n\n"
        "This usually happens when:\n"
        "  • other GPU processes are using most of the memory, or\n"
        "  • the chosen pRF grid is too large for your hardware.\n\n"
        "You can continue by:\n"
        "  • using a coarse grid (e.g. 11×11×5) for testing only, or\n"
        "  • closing GPU-heavy applications, or\n"
        "  • running on a system with larger GPUs."
        "\n\n"
        f"{RESET}"
    )


def _msg_not_enough_capacity_single_gpu(grid_shape, capacity):
    return (
         f"{BLUE}"
        "\n\n"
        "This GPU does not have enough memory to compute the requested pRF model signals.\n\n"
        f"Requested pRF grid: {grid_shape}\n"
        # f"GPU capacity: {capacity:,}\n\n"
        "With current GPU, you may try to run GEM-pRF using a much smaller pRF grid only for testing.\n"
        # "With current GPU, you may try to run GEM-pRF using a much smaller pRF grid only for testing. For example:\n\n"
        # "    <search_space>\n"
        # "        <default_spatial_grid visual_field_radius=\"13.5\" num_horizontal_prfs=\"11\" num_vertical_prfs=\"11\"/>\n"
        # "        <default_sigmas num_sigmas=\"5\" min_sigma=\"0.5\" max_sigma=\"5\"/>\n"
        # "    </search_space>\n\n"
        "For full-resolution pRF grid analysis, consider running on a system with ≥ 8–16 GB of free GPU memory."
        "\n\n"
        f"{RESET}"
    )


def _msg_not_enough_capacity_multi_gpu(grid_shape, total_capacity):
    return (
        f"{RED}"
        "\n\n"
        "Not enough combined GPU memory to compute the full pRF grid.\n\n"
        f"Requested pRF grid: {grid_shape}\n"
        # f"Total GPU capacity: {total_capacity:,}\n\n"
        "Even when splitting computations across multiple GPUs, the memory is insufficient.\n\n"
        "You can continue by:\n"
        "  • switching to a coarse pRF grid (e.g. 11×11×5), or\n"
        "  • freeing GPU memory on busy devices, or\n"
        "  • running this configuration on higher-memory GPUs."
        "\n\n"
        f"{RESET}"
    )


def analyze_gpus(
    grid_shape=(151, 151, 16),
    stimulus_shape=(101, 101),
    num_frames=375,
    dtype="float64",
    safety_factor=1.5,
    min_chunk_size=5000,
    memory_ratio_threshold=0.1,
):
    def to_gb(b): return float(b) / (1024 ** 3)

    # Compute model sizes
    nx, ny, nc = grid_shape
    sx, sy = stimulus_shape
    N = int(nx) * int(ny) * int(nc)
    HW = int(sx) * int(sy)
    itemsize = _bytes_for_dtype(dtype)

    per_signal_bytes = (HW + num_frames) * itemsize
    stimulus_bytes = HW * num_frames * itemsize

    # Detect GPUs
    gpus = _query_with_pynvml() or _query_with_nvidia_smi()
    if not gpus:
        return {
            "can_run": False,
            "summary": _msg_no_gpu(),
            "default_gpu": None,
            "additional_gpus": [],
            "selected_gpus": [],
            "assignments": [],
        }

    # Compute chunk sizes
    gpu_info = []
    max_free = max(g["free_bytes"] for g in gpus)

    for g in gpus:
        free = g["free_bytes"]
        avail = free - stimulus_bytes
        chunk = int(avail // (per_signal_bytes * safety_factor)) if avail > 0 else 0
        chunk = max(chunk, 0)

        gpu_info.append({
            "index": g["index"],
            "name": g["name"],
            "free_bytes": free,
            "free_gb": to_gb(free),
            "max_chunk_size": chunk,
        })

    # Filter out weak GPUs
    filtered = [
        g for g in gpu_info
        if g["max_chunk_size"] >= min_chunk_size and
           g["free_bytes"] >= memory_ratio_threshold * max_free
    ]

    if not filtered:
        return {
            "can_run": False,
            "summary": _msg_all_gpus_filtered(),
            "default_gpu": None,
            "additional_gpus": [],
            "selected_gpus": [],
            "assignments": [],
        }

    # Capacity check
    total_capacity = sum(g["max_chunk_size"] for g in filtered)
    can_run = total_capacity >= N

    if not can_run:
        if len(filtered) == 1:
            # single GPU, insufficient
            msg = _msg_not_enough_capacity_single_gpu(grid_shape, total_capacity)
        else:
            # multi-GPU cluster, still insufficient
            msg = _msg_not_enough_capacity_multi_gpu(grid_shape, total_capacity)

        return {
            "can_run": False,
            "summary": msg,
            "default_gpu": None,
            "additional_gpus": [],
            "selected_gpus": filtered,
            "assignments": [],
        }

    # Split work across GPUs
    k = len(filtered)
    base = N // k
    remainder = N % k
    assignments = []
    for i, g in enumerate(filtered):
        n_assigned = base + (1 if i < remainder else 0)
        assignments.append({"gpu_index": g["index"], "num_signals": n_assigned})

    default_gpu = max(filtered, key=lambda g: g["free_bytes"])["index"]
    selected_gpus = [g["index"] for g in filtered]
    additional_gpus = [i for i in selected_gpus if i != default_gpu]

    # Build success summary
    lines = []
    lines.append(f"Grid: {nx}×{ny}×{nc} = {N:,} signals")
    lines.append(f"Stimulus: {sx}×{sy} × {num_frames} frames")
    lines.append("")
    lines.append("Selected GPUs:")
    for g in filtered:
        lines.append(f"  GPU{g['index']} — free={g['free_gb']:.2f} GB, max-chunk={g['max_chunk_size']:,}")
    lines.append("")
    lines.append(f"default_gpu: {default_gpu}")
    lines.append(f"additional_gpus: {additional_gpus}")
    lines.append("")
    lines.append("Assignments:")
    for a in assignments:
        lines.append(f"  GPU{a['gpu_index']}: {a['num_signals']:,}")

    return {
        "can_run": True,
        "summary": "\n".join(lines),
        "default_gpu": default_gpu,
        "additional_gpus": additional_gpus,
        "selected_gpus": selected_gpus,
        "assignments": assignments,
    }

# # def apply_gpu_selection_to_xml(config_path, default_gpu, additional_gpus):
# #     # Update default GPU
# #     update_xml_value(
# #         config_path,
# #         "//gpu/default_gpu",
# #         default_gpu
# #     )

# #     # Clear <additional_available_gpus> content
# #     from lxml import etree
# #     parser = etree.XMLParser(remove_blank_text=False)
# #     tree = etree.parse(config_path, parser)
# #     root = tree.getroot()

# #     gpu_section = root.xpath("//gpu/additional_available_gpus")[0]
# #     gpu_section.clear()

# #     # Re-add <gpu> tags for additional GPUs
# #     for idx in additional_gpus:
# #         el = etree.Element("gpu")
# #         el.text = str(idx)
# #         gpu_section.append(el)

# #     tree.write(config_path, pretty_print=True, encoding="UTF-8")

def apply_gpu_selection_to_xml(config_path, default_gpu, additional_gpus):
    """
    Update <default_gpu> and <additional_available_gpus> in the XML file.
    Ensures <additional_available_gpus> always contains ≥1 <gpu> tag.
    """

    from lxml import etree

    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(config_path, parser)
    root = tree.getroot()

    # --- Update default GPU ---
    default_node = root.xpath("//gpu/default_gpu")[0]
    default_node.text = str(default_gpu)

    # --- Update additional GPUs ---
    add_section = root.xpath("//gpu/additional_available_gpus")[0]
    add_section.clear()   # keep the node, drop its children

    # If no additional GPUs → repeat default_gpu (this prevents XML errors)
    gpu_list = additional_gpus if additional_gpus else [default_gpu]

    for idx in gpu_list:
        el = etree.Element("gpu")
        el.text = str(idx)
        add_section.append(el)

    # Save
    tree.write(config_path, pretty_print=True, encoding="UTF-8")


def handle_gpu_decision(result, config_path):
    """
    Accepts the result from analyze_gpus().
    - If can_run=True → write GPU selections and continue.
    - If can_run=False → show message and offer coarse-grid fallback.
    """

    if result["can_run"]:
        print("\033[92mGPU check passed. Updating XML...\033[0m")

        apply_gpu_selection_to_xml(
            config_path,
            result["default_gpu"],
            result["additional_gpus"]
        )

        print("\033[92mXML GPU configuration updated.\033[0m")
        return config_path

    # --- Not enough GPU memory ---
    print(result["summary"])
    print("\033[91mGEM-pRF cannot run with the current GPU configuration with the specified pRF grid.\033[0m")

    # Ask user whether they want coarse grid fallback
    if input("\nWould you like to run a coarse pRF grid (11×11×5) instead for testing? [y/n]: ").strip().lower() != "y":                        
        print("\033[91mExiting — user declined coarse grid fallback.\033[0m")
        return None

    # Create temp folder and coarse config path
    import os
    temp_dir = os.path.join(os.path.dirname(config_path), "temp_user_configs")
    os.makedirs(temp_dir, exist_ok=True)

    coarse_path = os.path.join(
        temp_dir,
        os.path.splitext(os.path.basename(config_path))[0] + "_coarse.xml"
    )

    # Create coarse config
    create_coarse_grid_config(config_path, coarse_path)

    # --- Re-run GPU analysis with coarse grid ---
    from utils.gpu_info import analyze_gpus
    coarse_result = analyze_gpus(grid_shape=(11, 11, 5))

    if not coarse_result["can_run"]:
        print("\033[91mEven the coarse grid cannot run on this GPU.\033[0m")
        return None

    # --- Apply GPU selection to coarse config ---
    apply_gpu_selection_to_xml(
        coarse_path,
        coarse_result["default_gpu"],
        coarse_result["additional_gpus"]
    )

    coarse_path = coarse_path.replace("\\", "/")
    print(
        "\033[92m\nCreated coarse-grid config:\n"
        f"  {coarse_path}\n"
        "\nRunning GEM-pRF with reduced pRF grid (11×11×5)...\n\n\033[0m"
    )

    return coarse_path


def format_info(info):
    if isinstance(info, str):
        return info

    lines = []
    for k, v in info.items():
        # Indent nested dicts/lists nicely
        if isinstance(v, (dict, list)):
            lines.append(f"{k}:")
            lines.append(f"  {v}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)

def cupy_gpu_sanity_check_verbose():
    """
    Stepwise CuPy+GPU sanity check.

    Returns:
        (ok: bool, info: dict or str)
        - ok True  => info is a dict with details (cupy_version, cuda_runtime, cuda_driver, device_count, notes)
        - ok False => info is a helpful error message (str) or dict with diagnostic fields.
    """
    # Step 1: Try import
    try:
        import cupy as cp
    except Exception as imp_err:
        msg = str(imp_err)
        # Common symptoms: missing shared libs like libnvrtc.so.* or libcudart.so.*
        if "libnvrtc" in msg or "libcudart" in msg or "cannot open shared object file" in msg:
            advice = (
                "CuPy import failed because required CUDA libraries weren't found.\n"
                "This usually means the installed CuPy wheel was built for a different CUDA version\n"
                "than the CUDA runtime libraries installed on the system.\n\n"
                "Suggestions:\n"
                "  • If you want CuPy that matches your system, install the wheel for your CUDA:\n"
                "      pip install cupy-cuda12x   # if your system CUDA is 12.x\n"
                "      pip install cupy-cuda11x   # if your system CUDA is 11.x\n"
                "  • Alternatively ensure your CUDA libs are on LD_LIBRARY_PATH (or in /usr/lib):\n"
                "      export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH\n"
            )
            return False, f"CuPy import error: {msg}\n\n{advice}"
        else:
            return False, f"Failed to import CuPy: {msg}"

    # From here on cp is available
    try:
        # Step 2: Check device count quickly
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
        except Exception as e_devcount:
            s = str(e_devcount)
            # Often indicates runtime/lib mismatch or driver problem
            if "cudaError" in s or "cannot open shared object file" in s or "libnvrtc" in s:
                return False, {
                    "error_stage": "device_count",
                    "error": s,
                    "diagnosis": "Failed when querying device count. Likely CuPy/CUDA library mismatch or driver problem.",
                    "suggestion": "Check that CuPy was installed for your CUDA version and that the CUDA toolkit libs are present."
                }
            return False, f"Error querying device count: {s}"

        if device_count == 0:
            return False, "No CUDA device detected (getDeviceCount() returned 0)."

        # Step 3: Try a trivial GPU operation to ensure kernels can be launched
        try:
            a = cp.arange(5)
            b = a * 2
            # force synchronization and actual kernel execution
            cp.cuda.Stream.null.synchronize()
        except Exception as gpu_op_err:
            s = str(gpu_op_err)
            # Look for the usual mismatch clues
            if "libnvrtc" in s or "libcudart" in s or "cannot open shared object file" in s:
                diagnosis = "Import succeeded but runtime library load failed when launching GPU code. CuPy was likely built for a different CUDA."
                suggestion = (
                    "Reinstall CuPy with the wheel matching your CUDA (e.g., pip install cupy-cuda12x) "
                    "or install the CUDA version that CuPy expects."
                )
                return False, {"error_stage": "gpu_op", "error": s, "diagnosis": diagnosis, "suggestion": suggestion}
            else:
                return False, {"error_stage": "gpu_op", "error": s, "diagnosis": "GPU op failed for an unknown reason."}

        # Step 4: Query runtime & driver versions
        try:
            runtime_ver = cp.cuda.runtime.runtimeGetVersion()
            driver_ver = cp.cuda.runtime.driverGetVersion()
            # Convert encoded int (e.g., 12020) to major.minor if possible
            def parse_ver(v):
                try:
                    v = int(v)
                    major = v // 1000
                    minor = (v % 1000) // 10
                    return f"{major}.{minor}"
                except Exception:
                    return str(v)
            runtime_str = parse_ver(runtime_ver)
            driver_str = parse_ver(driver_ver)
        except Exception as ver_err:
            # not critical — driver/runtime query failed but GPU ops already worked
            return True, {
                "cupy_version": getattr(cp, "__version__", "unknown"),
                "device_count": device_count,
                "note": "GPU operations worked, but could not query runtime/driver versions.",
                "version_query_error": str(ver_err)
            }

        # Optional quick compatibility hint:
        # CuPy wheels embed the CUDA ABI they were built for. If import worked and trivial ops passed,
        # we consider this environment usable. But if the runtime major differs from what you expect,
        # we still warn.
        # We can attempt to guess CuPy's compiled CUDA major from cp.__cuda_version__ if available.
        compiled_cuda = getattr(cp, "__cuda_version__", None)  # e.g. "12.2"
        notes = []
        if compiled_cuda:
            # compare major parts
            try:
                comp_major = int(str(compiled_cuda).split(".")[0])
                run_major = int(runtime_str.split(".")[0])
                if comp_major != run_major:
                    notes.append(
                        f"Warning: CuPy reports it was built for CUDA {compiled_cuda} but runtime is {runtime_str}. "
                        "This might be OK if libc ABI is compatible, but can also be a source of subtle errors."
                    )
            except Exception:
                pass

        info = {
            "cupy_version": getattr(cp, "__version__", "unknown"),
            "cupy_compiled_for_cuda": compiled_cuda,
            "cuda_runtime_reported": runtime_str,
            "cuda_driver_reported": driver_str,
            "device_count": device_count,
            "notes": notes or ["OK — import, device detection, and a trivial GPU op succeeded."]
        }
        return True, info

    except Exception as final_e:
        return False, f"Unexpected error during CuPy sanity check: {final_e}"

if __name__ == '__main__':
    # CuPy/CUDA matching sanity check
    ok, info = cupy_gpu_sanity_check_verbose()
    print("OK:", ok)
    print(format_info(info))
    print()

    # quick CLI test
    res = analyze_gpus()
    print(res['summary'])
    if res.get('warning'):
        print('\nWARNING: ' + res['warning'])
