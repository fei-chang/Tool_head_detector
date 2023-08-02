# Head Detector
This is a ready-to-use head detector tool adopted [yolov3](https://github.com/ultralytics/yolov3).

The head detector is trained on dataset [HollywoodHeads](https://www.robots.ox.ac.uk/~vgg/software/headmview/), to download the original dataset, click [here](https://www.robots.ox.ac.uk/~vgg/software/headmview/head_data/hollywood-heads.zip).

## Usage

1. Run by command line
```bash

python detect.py --model_weights path/to/model_weights.pt --input_img_folder path/to/imgs --txt_file path/to/save.txt

```

2. Adopted for running folders containing subfolders

```python
model_weights = 'path/to/model_weights.pt'
model, device, imgsz = load_model(model_weights)

base_folder = 'path/to/folders'

for folder in os.listdir(base_folder):
    input_img_folder = os.path.join(base_folder, folder)
    txt_file = 'path/to/save.txt'
    dataset = LoadImages(input_img_folder, img_size=imgsz, stride=model.stride, auto=model.pt and not model.jit)
    run_single_folder(model, device, dataset, txt_file)
    LOGGER.info('Done: %s'%input_img_folder)

```
