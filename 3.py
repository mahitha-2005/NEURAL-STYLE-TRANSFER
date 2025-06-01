import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# --- Load image ---
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')

    size = max_size if max(image.size) > max_size else max(image.size)
    if shape is not None:
        size = shape

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)  # add batch dimension
    return image.to(device)

# --- Display image ---
def imshow(tensor, title=None):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')  # Hide axes
    plt.show()

# --- Get features from VGG19 ---
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content layer
            '28': 'conv5_1'
        }

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# --- Gram Matrix for style representation ---
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg19(pretrained=True).features.to(device).eval()

content = load_image("content.jpg")
style = load_image("style.jpg", shape=content.shape[-2:])

# --- Get features ---
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# --- Calculate style gram matrices ---
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# --- Target image to optimize ---
target = content.clone().requires_grad_(True).to(device)


style_weight = 1e6
content_weight = 1


optimizer = optim.Adam([target], lr=0.003)


for step in range(1, 301):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    style_loss = 0
    for layer in style_grams:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss
total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Total Loss: {total_loss.item()}")


imshow(target, title='Styled Image')  
