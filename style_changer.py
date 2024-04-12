from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import utils as vutils
import os
import copy
import imageio
import glob


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 判断是否使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义图像加载和变换函数
def loader_transform(new_size=None):
    if new_size:  # 如果提供了新尺寸，使用新尺寸
        transform = transforms.Compose([
            transforms.Resize(new_size),
            transforms.ToTensor()])
    else:  # 否则使用原始尺寸
        transform = transforms.Compose([
            transforms.ToTensor()])
    return transform

def image_loader(image_name, new_size=None):
    loader = loader_transform(new_size)
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

unloader = transforms.ToPILImage()  # 张量转为 PIL Image 格式

# 显示图像的函数
def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    # plt.figure()
    plt.axis('off')
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.001)

# 定义内容损失和风格损失类
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

# 载入预训练的 VGG19 模型
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# 标准化操作
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# 定义修改后的模型以及损失计算
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses

# 优化器
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

# 风格迁移函数
def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps=300, style_weight=1000000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    input_img.data.clamp_(0, 1)
    vutils.save_image(input_img.data,'output/output_img_%03d.png' % (run[0]), normalize=True)
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
            return loss

        optimizer.step(closure)
        input_img.data.clamp_(0, 1)
        imshow(input_img, title='Processing Image')
        vutils.save_image(input_img.data,'output/output_img_%03d.png' % (run[0]), normalize=True)

    input_img.data.clamp_(0, 1)
    return input_img

def run_style_transfer_with_fallback(cnn, normalization_mean, normalization_std,
                                     content_img_path, style_img_path, num_steps=300,
                                     style_weight=1000000, content_weight=1, resize_step=100):
    # 初始化图像尺寸调整步长
    current_step = 0
    
    # 首先加载内容图像以获取其原始尺寸
    content_img = image_loader(content_img_path)
    _, _, content_height, content_width = content_img.size()
    
    while True:
        try:
            # 根据当前步长调整尺寸
            if current_step > 0:
                new_size = max(content_height - current_step, 1), max(content_width - current_step, 1)
                content_img = image_loader(content_img_path, new_size=new_size)
                style_img = image_loader(style_img_path, new_size=new_size)
            else:
                # 首次尝试使用原始尺寸
                style_img = image_loader(style_img_path, new_size=(content_height, content_width))
            
            input_img = content_img.clone()
            output = run_style_transfer(cnn, normalization_mean, normalization_std,
                                        content_img, style_img, input_img, num_steps,
                                        style_weight, content_weight)
            return output
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA memory is insufficient, trying a smaller image size by reducing {resize_step} pixels.")
                current_step += resize_step  # 增加调整步长
                if current_step >= content_height or current_step >= content_width:
                    raise ValueError("Image has been reduced too much, unable to proceed.")
            else:
                raise e


# 主函数
def main():
    if not os.path.isdir('./output'):
        os.makedirs('./output')
    
    content_img_path = "C:/Users/mumua/Downloads/EGS_Cyberpunk2077_CDPROJEKTRED_S2_03_1200x1600-b1847981214ac013383111fc457eb9c5.jpg"
    style_img_path = "C:/Users/mumua/Downloads/MV5BMDBmYTZjNjUtN2M1MS00MTQ2LTk2ODgtNzc2M2QyZGE5NTVjXkEyXkFqcGdeQXVyNzAwMjU2MTY@._V1_.jpg"
    
    try:
        output = run_style_transfer_with_fallback(cnn, cnn_normalization_mean, cnn_normalization_std,
                                                  content_img_path, style_img_path)
        # plt.figure()
        # imshow(output, title='Output Image')
    except ValueError as e:
        print(e)
    
    # plt.ioff()
    # plt.show()
    
    anim_file = './output/process.gif'

    with imageio.get_writer(anim_file, mode='I',  duration=300, loop=0) as writer:
        filenames = glob.glob('./output/output_img*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image) 

if __name__ == "__main__":
    main()