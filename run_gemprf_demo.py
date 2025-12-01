"""
"@Author  :   Siddharth Mittal",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2025, Medical University of Vienna",
"@Desc    :   Demo script to run GEMpRF with optional interactive config selection,
               auto path setting, and GPU memory check.",     
"""

import gemprf as gp
from utils.messages import Messages

# Want to pick a config manually?
# Set CONFIG_FILEPATH below and set interactively_choose_config_file = False.
CONFIG_FILEPATH = r"path/to/your/config.xml"

if __name__ == "__main__":
    interactively_choose_config_file = True
    run_auto_path_setting = True
    run_auto_gpu_check = True

    ####################################################
    # NOTE: (OPTIONAL) Choose config file interactively
    #####################################################
    if interactively_choose_config_file:
        from utils.config_library import choose_config
        CONFIG_FILEPATH = choose_config()

    # (OPTIONAL) path settings - only to assist you    
    if run_auto_path_setting:
        Messages.print_message("005", "yellow")
        from utils.auto_path import auto_path_setting
        auto_path_setting(CONFIG_FILEPATH)    

    # (OPTIONAL) GPU memory check — only to assist you
    final_config = CONFIG_FILEPATH    
    if run_auto_gpu_check:
        from utils.gpu_info import analyze_gpus, handle_gpu_decision
        Messages.print_message("006", "yellow")

        final_config = handle_gpu_decision(analyze_gpus(), CONFIG_FILEPATH)
        if final_config is None:
            Messages.print_message("007", "red")
            exit(1)

    #####################################################
    # --------------- (THE REAL DEAL) ----------------- #
    #####################################################
    # Run GEMpRF analysis based on configuration file
    try:
        gp.run(final_config)
    except Exception as e:
        print(f"An error occurred during GEMpRF execution: {e}")
    finally:
        import os
        Messages.print_message(
            "009",
            "green",
            custom_format={"folder": os.path.dirname(final_config), "updated_name": os.path.basename(final_config), "sample_name": "sample_config.xml"}
        )