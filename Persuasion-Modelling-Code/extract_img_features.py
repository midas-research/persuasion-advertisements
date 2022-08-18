import torch
import timm

import json
import numpy as np

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from PIL import Image

action_reason_annot_path = "data/train/QA_Combined_Action_Reason_train.json"
image_dir = "data/train_images/"
output_feature_path = "output/img_features_vit_train.npy"
batch_size = 512
device = 'cuda:2'

with open(action_reason_annot_path, 'r') as fp:
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
            results[img_id] = features[i, :].detach().cpu().numpy()

        image_ids, batch = [], []

        print("On img {}/{}".format(index, len(data)))
    
    filename = image_dir + image_id
    img = Image.open(filename).convert('RGB')

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

    for i, img_id in enumerate(image_ids):
        results[img_id] = features[i, :].cpu().numpy()

# Write results.
print(len(results), len(data))
assert len(results) == len(data)

with open(output_feature_path, 'wb') as fp:
    np.save(fp, results)

print('Exported features for {} images.'.format(len(data)))

        
