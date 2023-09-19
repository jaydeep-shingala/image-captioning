import numpy as np
import timm
import torch
from PIL import Image
from vit import VisionTransformer
import csv
import os

torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_name = "vit_base_patch16_384"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()

# Read the text file
with open("/data/home/jaydeeps/ImageCaptioning/flickr8k/captions.txt", "r") as f:
    reader = csv.reader(f)
    rows = list(reader)

# Create a new text file
count = 1
tensors = []
captions = []
with open("imgfeature_captions.csv", "w") as f:
    writer = csv.writer(f)

    # Iterate over the rows
    for row in rows[1:4002]:
        image_name = row[0]
        caption = row[1]

        # Pass the image to the custom model
        image_path = os.path.join("/data/home/jaydeeps/ImageCaptioning/flickr8k/images/", image_name)
        im = Image.open(image_path)
        resized_image = im.resize((384, 384))
        img = (np.array(resized_image) / 128) - 1  # in the range -1, 1
        inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        features = model_official.forward_features(inp)
        # features.grad_fn = None
        tensors.append(features.detach().numpy())
        captions.append(caption)
        count = count + 1
        if (count%100 == 0):
            print(count)

print(len(tensors))
print(len(captions))
np.save("Image_feature_tensors_2.npy", tensors)
with open("Image_text_captions_2", "w") as f:
    for item in captions:
        f.write(str(item) + "\n")
        # Save the output to the new text file
        # print()
        # writer.writerow([features, caption])
        # count = count + 1
        # if (count%100 == 0):
        #     print(count)

print("Done!!!!:)")