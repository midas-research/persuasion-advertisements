import torch
import timm

import json
import numpy as np

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from PIL import Image
import cv2

bounding_box_json = "output/symbol_box_train.json"
image_dir = "data/train_images/"
output_feature_path = "output/roi_features_vit_train.npy"
batch_size = 64
max_number_of_regions = 10
default_image_size = 224

device = 'cuda:3'

def image_crop_and_resize(image, bbox, crop_size):
    """Crops roi from an image and resizes it.

    Args:
        image: the image.
        bbox: bounding box information.
        crop_size: the expected output size of the roi image.

    Returns:
        a [crop_size, crop_size, 3] roi image.
    """
    height, width, _ = image.shape

    x1, y1, x2, y2 = bbox
    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)

    return image[y1: y2, x1: x2, :]

with open(bounding_box_json, 'r') as fp:
    data = json.loads(fp.read())

model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=0)
model = model.to(device)
print(model)

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

results = {}

image_ids, batch = [], []
for index, (image_id, example) in enumerate(data.items()):
    if index % batch_size == 0 and len(batch) > 0:
        imgs = torch.stack(batch).to(device)
        print(imgs.shape)

        with torch.no_grad():
            features = model(imgs)

        print(features.shape)
        print(torch.max(features), torch.min(features))
        #print(features.type())

        for i, img_id in enumerate(image_ids):
            results[img_id] = features[10*i: 10*(i+1), :].detach().cpu().numpy()

        image_ids, batch = [], []

        print("On img {}/{}".format(index, len(data)))
    
    filename = image_dir + image_id
    bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
    rgb = bgr[:, :, ::-1]

    for region in example['regions'][:max_number_of_regions]:
        roi = image_crop_and_resize(rgb, 
            bbox=(
            region['bbox']['xmin'], 
            region['bbox']['ymin'], 
            region['bbox']['xmax'], 
            region['bbox']['ymax']), 
            crop_size=(default_image_size, default_image_size))
            

        img = Image.fromarray(roi)
        img_tensor = transform(img)
        batch.append(img_tensor)

    image_ids.append(image_id)

if len(batch) > 0:
    imgs = torch.stack(batch).to(device)
    print(imgs.shape)

    with torch.no_grad():
        features = model(imgs)

    print(features.shape)
    print(torch.max(features), torch.min(features))
    #print(features.type())

    for i, img_id in enumerate(image_ids):
        results[img_id] = features[10*i: 10*(i+1), :].detach().cpu().numpy()

# Write results.
print(len(results), len(data))
assert len(results) == len(data)

with open(output_feature_path, 'wb') as fp:
    np.save(fp, results)

print('Exported features for {} images.'.format(len(data)))

        
