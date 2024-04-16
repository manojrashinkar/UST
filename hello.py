from pathlib import Path
import urllib.request
from tqdm import tqdm  # Make sure to use this import
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov

def download_file(url, filename, dest_path):
    response = urllib.request.urlopen(url)
    total = response.getheader('Content-Length')
    if total:
        total = int(total)
    dest_path.mkdir(parents=True, exist_ok=True)
    file_path = dest_path / filename

    with tqdm(total=total, unit='B', unit_scale=True, desc=filename) as t:
        with open(file_path, 'wb') as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                t.update(len(chunk))

# Define paths and URLs as before
base_artifacts_dir = Path('./artifacts').expanduser()
model_name = "v3-small_224_1.0_float"
model_xml_name = f'{model_name}.xml'
model_bin_name = f'{model_name}.bin'
model_xml_path = base_artifacts_dir / model_xml_name
base_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/'

if not model_xml_path.exists():
    download_file(base_url + model_xml_name, model_xml_name, base_artifacts_dir)
    download_file(base_url + model_bin_name, model_bin_name, base_artifacts_dir)
else:
    print(f'{model_name} already downloaded to {base_artifacts_dir}')

import ipywidgets as widgets

core = ov.Core()
device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

print(device)

core = ov.Core()
model = core.read_model(model=model_xml_path)
compiled_model = core.compile_model(model=model, device_name=device.value)

output_layer = compiled_model.output(0)


# Download the image from the openvino_notebooks storage
image_filename = download_file(
    "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",dest_path=Path("data"),filename="coco.jpg"
)

# The MobileNet model expects images in RGB format.
image = cv2.cvtColor(cv2.imread(filename=str("data/coco.jpg")), code=cv2.COLOR_BGR2RGB)

# Resize to MobileNet image shape.
input_image = cv2.resize(src=image, dsize=(224, 224))

# Reshape to model input shape.
input_image = np.expand_dims(input_image, 0)
plt.imshow(image)
plt.show()



result_infer = compiled_model([input_image])[output_layer]
result_index = np.argmax(result_infer)

imagenet_filename = download_file(
    "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",dest_path=Path("data"),
  filename="imagenet_2012.txt"
)
# Specify the file path
file_path = "data/imagenet_2012.txt"

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]
    print(lines)
except FileNotFoundError:
    print("File does not exist")
except IOError:
    print("An error occurred while reading the file.")

# imagenet_classes = imagenet_filename.read_text().splitlines()

# The model description states that for this model, class 0 is a background.
# Therefore, a background must be added at the beginning of imagenet_classes.
imagenet_classes = ['background'] + lines

res = imagenet_classes[result_index]
print(res)
