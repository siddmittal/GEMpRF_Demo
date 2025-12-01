"""
"@Author  :   Siddharth Mittal",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2025, Medical University of Vienna",
"@Desc    :   Utility to choose GEMpRF configuration files interactively from a library.",
"""
import os

def get_config_library():
    """
    Returns a list of config options.
    Each entry:
      {
        "name": "...",
        "path": "...",
        "desc": "..."
      }
    """
    configs = [
        {
            "name": "Example 001 — BIDS + prfprepare (surface), individual runs",
            "file": "example-001_runtype-individual_input-bids_desc-analyse-prfprepare-surface-data.xml",
            "desc": "Use this when data is BIDS-organised and stimulus/task was prepared using prfprepare. Surface-mode."
        },
        {
            "name": "Example 002 — BIDS + fMRIPrep (surface), individual runs",
            "file": "example-002_runtype-individual_input-bids_desc-analyse-fmriprep-surface-data.xml",
            "desc": "For BIDS datasets processed via fMRIPrep. Surface-based analysis."
        },
        {
            "name": "Example 003 — BIDS + fMRIPrep (volume), individual runs",
            "file": "example-003_runtype-individual_input-bids_desc-analyse-fmriprep-volume-data.xml",
            "desc": "For volumetric fMRIPrep outputs. Use when working with MRI volumes instead of surfaces."
        },
        {
            "name": "Example 004 — BIDS + fMRIPrep (surface + volume), individual runs",
            "file": "example-004_runtype-individual_input-bids_desc-analyse-fmriprep-both-surface-volume-data.xml",
            "desc": "Hybrid: processes both surface and volumetric fMRIPrep files at once."
        },
        {
            "name": "Example 005 — Fixed paths (non-BIDS) + prfprepare (surface)",
            "file": "example-005_runtype-individual_input-fixedPath_desc-analyse-prfprepare-surface-date.xml",
            "desc": "When files are not arranged in BIDS format. Paths are manually provided. Surface-mode."
        }
    ]

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Convert file names to absolute paths inside sample_configs/
    sample_dir = os.path.join(repo_root, "sample_configs")

    for c in configs:
        c["path"] = os.path.join(sample_dir, c["file"]).replace("\\", "/")

    return configs


def choose_config():
    """Display numbered config list and return chosen config path."""

    options = get_config_library()

    # print("\nAvailable GEM-pRF configuration profiles:\n")
    # print("\033[92m\n\nHey there! You're in GEM-pRF interactive mode.\nQuestions? Compliments? Ping the author, Siddharth Mittal.\nChoose a config and let’s roll...\n\033[0m")

    print("\033[92m"
          "\n\n"
        "\n============================================================"
        "\n             🚀 GEM-pRF Demo — Interactive Mode"
        "\n============================================================\n"
        "\nHey there! You're in GEM-pRF interactive mode."
        "\nQuestions? Compliments? Ping the author, Siddharth Mittal."
        "\nChoose a config and let’s roll...\n"
        "\033[0m")


    for i, c in enumerate(options, 1):
        print(f"  {i}) {c['name']}\n     {c['desc']}\n")

    choice = None
    while choice is None:
        try:
            idx = int(input("Choose a config number: ").strip())
            if 1 <= idx <= len(options):
                choice = options[idx - 1]["path"]
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Enter a number.")

    return choice
