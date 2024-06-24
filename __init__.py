from signature_core.version import __version__

from .nodes import lora

try:
    from .nodes import enhance, transform, filters, models, morphology, misc, augmentation
except:
    print(f"Error importing modules")

    import subprocess
    try:
        subprocess.check_call(['pip3', 'install', '-e', '.'])
    except subprocess.CalledProcessError:
        print("Installation failed. Please install the dependencies manually.")

    # Retry the import after attempting installation
    from .nodes import enhance, transform, filters, models, morphology, misc, augmentation

NODE_CLASS_MAPPINGS = {
    **models.NODE_CLASS_MAPPINGS,
    **transform.NODE_CLASS_MAPPINGS,
    **enhance.NODE_CLASS_MAPPINGS,
    **filters.NODE_CLASS_MAPPINGS,
    **morphology.NODE_CLASS_MAPPINGS,
    **misc.NODE_CLASS_MAPPINGS,
    **augmentation.NODE_CLASS_MAPPINGS,
    **lora.NODE_CLASS_MAPPINGS,
}

WEB_DIRECTORY = "./nodes/web"
NAME = "🔵 Signature Nodes"


__all__ = ["NODE_CLASS_MAPPINGS", "MANIFEST"]

MANIFEST = {
    "name": NAME,
    "version": __version__,
    "author": "marcojoao",
    "description": "Signature Nodes",
}