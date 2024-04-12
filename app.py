from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import threading
import os
import glob
import torch
from torchvision import transforms, models
from torchvision.utils import save_image
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil
import sys

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
    image = Image.open(image_name).convert("RGB")  # Ensure image is RGB
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# 损失函数定义
class ContentLoss(nn.Module):
    def __init__(self, target):
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
    a, b, c, d = input.size()  # a=batch size(=1)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

# 使用VGG19模型

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

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

# 风格迁移函数
def run_style_transfer(cnn, normalization_mean, normalization_std, 
                       content_img, style_img, input_img, slider, 
                       num_steps=300, style_weight=1000000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)
    
    # 使用LBFGS优化器
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]  # 用于跟踪迭代次数的列表
    early_stop = False  # 早停标志

    
    save_image(input_img.data, 'output/output_img_0.png')

    while run[0] < num_steps and not early_stop:
        
        def closure():
            nonlocal early_stop  # 允许在闭包内修改early_stop变量
            # 对图像的像素值进行裁剪，以确保它们在[0, 1]的范围内
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


            if content_score.item() > 50:
                print(f"Stopping early at iteration {run[0]} due to high content loss: {content_score.item()}")
                early_stop = True  # 如果内容损失超过50，则设置early_stop为True



            run[0] += 1
            if run[0] % 10 == 0:
                print(f"run [{run[0]}]:")
                print(f'Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}')
                # 保存当前步骤的输出图像
                
                save_image(input_img.data, f'output/output_img_{run[0]}.png')

            # 更新滑动条的最大值
            slider.config(to=run[0])
            
            return loss

        optimizer.step(closure)

        # 对图像的像素值进行裁剪，以确保它们在[0, 1]的范围内
        input_img.data.clamp_(0, 1)

        if early_stop:
            print("Early stopping triggered.")
            break

    return input_img

def download_image():
    current_step = slider.get()  # 获取滑块当前的值
    source_img_path = f'output/output_img_{current_step}.png'  # 构建源图片的路径
    if os.path.exists(source_img_path):
        # 提取用户选择的内容图像的目录作为目标文件夹
        dest_folder = os.path.dirname(content_img_path_var.get())
        # 构建目标路径
        dest_img_path = os.path.join(dest_folder, f'output_img_{current_step}.png')
        # 移动文件
        shutil.move(source_img_path, dest_img_path)
        print(f"Image moved to {dest_img_path}")
    else:
        print("Selected step image does not exist.")

# Tkinter GUI 设置和新功能

def update_image_label(image_path, label, max_size=(250, 250)):
    img = Image.open(image_path)
    # 保持宽高比的情况下调整图像大小
    img.thumbnail(max_size, Image.ANTIALIAS)
    
    img = ImageTk.PhotoImage(img)
    label.configure(image=img)
    label.image = img  # 保持对图像的引用，防止被垃圾回收

def choose_file(var, label):
    filename = filedialog.askopenfilename(initialdir="/", title="Select File",
                                          filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    var.set(filename)
    update_image_label(filename, label)

def start_style_transfer_thread(content_path, style_path, slider):
    def run():
        # 清理旧的输出图片
        for img_file in glob.glob('output/output_img_*.png'):
            os.remove(img_file)

        content_img = image_loader(content_path, new_size=new_size)
        style_img = image_loader(style_path, new_size=new_size)
        input_img = content_img.clone()
        # 传递滑动条给run_style_transfer函数
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img, slider=slider)
        update_image_label('output/output_img_0.png', result_img_label)

    threading.Thread(target=run).start()



def update_result_image(value):
    # 当脚本被打包成exe时，使用sys.executable获取exe的路径
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))

    step_img_path = os.path.join(application_path, 'output', f'output_img_{value}.png')
    
    if os.path.exists(step_img_path):
        update_image_label(step_img_path, result_img_label)
    else:
        print("Step image does not exist.")
        print(step_img_path)


if __name__ == '__main__':
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 檢查資料夾是否存在
    if not os.path.exists('output'):
        # 如果資料夾不存在，則創建資料夾
        os.makedirs('output')
        print(f"'{'output'}' 資料夾已創建。")

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # 标准化操作
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    root = tk.Tk()
    root.title("Style Transfer App")

    content_img_path_var = tk.StringVar()
    style_img_path_var = tk.StringVar()

    frame = tk.Frame(root)
    frame.pack()

    # 图片预览Label
    content_img_label = Label(frame)
    content_img_label.pack(side=tk.LEFT)

    style_img_label = Label(frame)
    style_img_label.pack(side=tk.LEFT)

    result_img_label = Label(frame)  # 显示最终或中间步骤输出图像的Label
    result_img_label.pack()

    open_content_btn = tk.Button(frame, text="Choose Content Image", command=lambda: choose_file(content_img_path_var, content_img_label))
    open_content_btn.pack(side=tk.LEFT)

    open_style_btn = tk.Button(frame, text="Choose Style Image", command=lambda: choose_file(style_img_path_var, style_img_label))
    open_style_btn.pack(side=tk.LEFT)

    # 添加滑块
    slider = tk.Scale(frame, from_=0, to=300, orient=tk.HORIZONTAL, length=300, resolution=10)
    slider.pack()

    # 假设我们将所有图像调整到这个尺寸
    new_size = (256, 256)

    start_btn = tk.Button(frame, text="Start Transfer", command=lambda: start_style_transfer_thread(content_img_path_var.get(), style_img_path_var.get(), slider))
    start_btn.pack(side=tk.LEFT)

    download_btn = tk.Button(frame, text="Download Image", command=download_image)
    download_btn.pack(side=tk.LEFT)

    slider.config(command=update_result_image)
    root.mainloop()