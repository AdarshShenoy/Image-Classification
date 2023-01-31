import torch
import json
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='flowers/test/10/image_07090.jpg', help='Path to the image')
parser.add_argument('--top_k', type=int, default=6, help='How many probabilities')
parser.add_argument('--json', type=str, default='cat_to_name.json')
parser.add_argument('--gpu', type=str, default='cuda')
args = parser.parse_args()

with open(args.json, 'r') as f:
    cat_to_name = json.load(f)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    optimizer = checkpoint['optimizer_state_dict']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


model = load_checkpoint('checkpoint.pth')


def process_image(image_path):
    image = Image.open(image_path)

    width, height = image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        image = image.resize((round(aspect_ratio * 256), 256))
    else:
        image = image.resize((256, round(aspect_ratio * 256)))

    image.thumbnail(image.size, Image.ANTIALIAS)

    left = (256 - 224) / 2
    upper = (256 - 224) / 2
    right = (256 + 224) / 2
    lower = (256 + 224) / 2

    image = image.crop((left, upper, right, lower))
    image = np.array(image)
    image = image / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = ((image - mean) / std)

    image = np.transpose(image, (2, 0, 1))

    return image


def predict(image_path, model, topk, device):
    model.to(device)
    model.eval()

    image = process_image(image_path)
    image = torch.from_numpy(np.array([image]))
    image = image.float()

    with torch.no_grad():
        image = image.to(device)
        output = model.forward(image)
        ps = torch.exp(output)
        top_ps, top_classes = ps.topk(topk, dim=1)
        top_p = top_ps.tolist()[0]

        index_to_class = {val: cat_to_name[k] for k, val in model.class_to_idx.items()}
        top_class = [index_to_class[i] for i in top_classes.tolist()[0]]

    print(top_p, top_class)
    return top_p, top_class


if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

image = process_image(args.image_path)
top_ps, top_classes = predict(args.image_path, model, args.top_k, device)
top_class = top_classes[0]
# labels = [cat_to_name[str(index)] for index in top_classes]
print("Predicted Class: {}".format(top_class))

for i in range(args.top_k):
    print("Class: {} || Probability: {}".format(top_classes[i], top_ps[i]))
