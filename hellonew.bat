@echo off

REM Ensure pip is up-to-date
python -m pip install --upgrade pip

REM Install required Python packages
python -c "import pathlib" 2>NUL || echo pathlib is a part of standard library.
python -c "import urllib.request" 2>NUL || echo urllib.request is a part of standard library.
python -c "import tqdm" 2>NUL || pip install tqdm
python -c "import cv2" 2>NUL || pip install opencv-python-headless
python -c "import matplotlib" 2>NUL || pip install matplotlib
python -c "import numpy" 2>NUL || pip install numpy
python -m pip install openvino
python -m pip install tqdm
python -m pip install opencv-python
python -m pip install matplotlib




@echo off



echo Running the Python script...
python hellonew.py

pause


