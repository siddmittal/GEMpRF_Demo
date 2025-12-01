"""
"@Author  :   Siddharth Mittal",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2025, Medical University of Vienna",
"@Desc    :   Utility functions for messages and prompts in GEMpRF.",
"""

class Messages:
    @classmethod
    def print_message(cls, message_id, color_code=None, custom_format=None):

        # find key starting with message_id ("001", "002", etc.)
        message_key = next(
            (k for k in cls.MESSAGES if k.startswith(message_id)),
            None
        )

        msg = cls.MESSAGES.get(message_key, "Unknown message ID.")

        if custom_format:
            msg = msg.format(**custom_format)        

        if color_code in cls.COLOR_CODES:
            color = cls.COLOR_CODES[color_code]
            reset = cls.COLOR_CODES["reset"]
            print(f"{color}{msg}{reset}")
        else:
            print(msg)

    # color dictionary (expand whenever you want)
    COLOR_CODES = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
    }

    # message dictionary outside the function
    MESSAGES = {
        "001_gpu_insufficient_memory": "Insufficient GPU memory available for the requested analysis.",
        "002_no_gpu": "No GPU detected. Please ensure a compatible GPU is installed and properly configured.",
        "003_proceed_with_cpu": "Proceeding with CPU computation may result in significantly longer processing times. Do you wish to continue? (y/n): ",
        "004_invalid_input": "Invalid input. Please enter 'y' for yes or 'n' for no.",

        "005_auto_path_info": (
            "AUTO path setting is running...\n"
            "Your config file will be updated with the correct paths for this demo.\n"
            "If you prefer to set paths manually, edit the XML configuration file and comment out this step.\n"
        ),

        "006_gpu_check_start": "Checking GPU availability...",

        "007_gpu_exit": "Exiting due to insufficient GPU memory.",

        "008_analysis_complete": "GEMpRF analysis complete.",

        "009_demo_finished": (
            "\n\nDemo finished. To understand how we set the configuration file to achieve this, compare the following XML files:"
            "\n📁 {folder}:\n"
            "\n  ├── updated   → {updated_name}\n"
            "  └── original  → {sample_name}"
        )

    }