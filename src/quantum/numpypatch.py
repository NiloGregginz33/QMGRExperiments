import os
import re

def patch_qilib_numpy():
    """
    Dynamically patches the `qilib` library to replace `np.cfloat` with `np.complex128`.
    """
    try:
        # Locate the file to patch
        site_packages_path = os.path.join(os.path.dirname(__file__), "venv", "Lib", "site-packages")
        target_file = os.path.join(site_packages_path, "qilib", "utils", "python_json_structure.py")

        # Read the file and replace np.cfloat
        with open(target_file, "r") as file:
            content = file.read()

        # Replace np.cfloat with np.complex128
        patched_content = re.sub(r"np\.cfloat", "np.complex128", content)

        # Write the patched content back to the file
        with open(target_file, "w") as file:
            file.write(patched_content)

        print("Successfully patched `qilib` to use `np.complex128` instead of `np.cfloat`.")

    except Exception as e:
        print(f"Failed to patch `qilib`: {e}")
