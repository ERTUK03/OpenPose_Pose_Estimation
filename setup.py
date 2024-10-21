import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def setup():
    try:
        import h5py
    except ImportError:
        install('h5py')

    try:
        from pycocotools.coco import COCO
    except ImportError:
        install('git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI')

    print("Setup complete.")
