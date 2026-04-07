import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimage
import torchvision.utils as vutils
import matplotlib.colors as c
import torch
from torch import tensor
import glob
import os
from PIL import Image
import torchvision.transforms as T
import seaborn as sns


def save_image_co(image_tensor, where, image_name, channels, savepath, single_channel_images=True, save_as_pdf=False, div_boundary=5):
    print(div_boundary)
    
    os.makedirs(savepath, exist_ok=True)
    ending = ".pdf" if save_as_pdf else ".png"

    cmap_R = c.LinearSegmentedColormap.from_list("cmap_R",
                                                 ['black', 'white', 'black'])  # hier sind die colormaps definiert,
    cmap_G = c.LinearSegmentedColormap.from_list("cmap_G", ['#0f0', 'white',
                                                            '#f00'])  # in denen die channel angezeigt werden
    cmap_B = c.LinearSegmentedColormap.from_list("cmap_B", ['yellow', 'white',
                                                            '#00f'])

    boundary_red = max(torch.max(image_tensor[0]), -torch.min(image_tensor[0]))/div_boundary


    img_norm = (image_tensor - torch.min(image_tensor)) / (torch.max(image_tensor) - torch.min(image_tensor))
    # img_norm = image_tensor / torch.max(image_tensor)
    vutils.save_image(img_norm, "" + savepath + "/" + image_name + "-CO" + ending)
    if single_channel_images:
        mpimage.imsave("" + savepath + "/" + image_name + "-bw" + ending, image_tensor[0].detach().cpu(), cmap=cmap_R,
                   vmin=-boundary_red, 
                   vmax=boundary_red)

    if channels > 1 and single_channel_images:
        boundary_green = max(torch.max(image_tensor[1]), -torch.min(image_tensor[1]))/div_boundary
        boundary_blue = max(torch.max(image_tensor[2]), -torch.min(image_tensor[2]))/div_boundary
        mpimage.imsave("" + savepath + "/" + image_name + "-rg" + ending, image_tensor[1].detach().cpu(), cmap=cmap_G,
                       vmin=-boundary_green, vmax=boundary_green)
        mpimage.imsave("" + savepath + "/" + image_name + "-bg" + ending, image_tensor[2].detach().cpu(), cmap=cmap_B,
                       vmin=-boundary_blue, vmax=boundary_blue)

    plt.close()

def save_image_bw(image_tensor, where, image_name, channels, savepath, single_channel_images=True, save_as_pdf=False, div_boundary=5):


    ending = ".pdf" if save_as_pdf else ".png"
    os.makedirs(savepath, exist_ok=True)


    cmap_R = c.LinearSegmentedColormap.from_list("cmap_R",
                                                 ['black', 'white', 'black'])  # hier sind die colormaps definiert, in denen die channel angezeigt werden


    boundary_red = max(torch.max(image_tensor[0]), -torch.min(image_tensor[0]))/div_boundary

    f, axarr = plt.subplots(5, 4, figsize=(20, 20))

    img_norm = (image_tensor - torch.min(image_tensor)) / (torch.max(image_tensor) - torch.min(image_tensor))
    # img_norm = image_tensor / torch.max(image_tensor)
    img_norm = img_norm*0.9
    vutils.save_image(img_norm, "" + savepath + "/" + image_name + "-GRAY" + ending)
    if single_channel_images:
        mpimage.imsave("" + savepath + "/" + image_name + "-bw" + ending, image_tensor[0].detach().cpu(), cmap=cmap_R,
                   vmin=-boundary_red, vmax=boundary_red)

    if False and channels > 1 and single_channel_images:
        boundary_green = max(torch.max(image_tensor[1]), -torch.min(image_tensor[1]))/div_boundary
        boundary_blue = max(torch.max(image_tensor[2]), -torch.min(image_tensor[2]))/div_boundary
        mpimage.imsave("" + savepath + "/" + image_name + "-rg" + ending, image_tensor[1].detach().cpu(), cmap=cmap_R,
                       vmin=-boundary_green, vmax=boundary_green)
        mpimage.imsave("" + savepath + "/" + image_name + "-bg" + ending, image_tensor[2].detach().cpu(), cmap=cmap_R,
                       vmin=-boundary_blue, vmax=boundary_blue)

    plt.close()

def save_image_rgb(image_tensor, where, image_name, channels, savepath, single_channel_images=True, save_as_pdf=False, div_boundary=5):

    ending = ".pdf" if save_as_pdf else ".png"


    os.makedirs(savepath, exist_ok=True)

    # Path(path + "/" + savepath +"/" + save_name).mkdir(parents=True, exist_ok=True)

    cmap_R = c.LinearSegmentedColormap.from_list("cmap_R",
                                                 ['black', 'white', '#f00'])  # hier sind die colormaps definiert,
    cmap_G = c.LinearSegmentedColormap.from_list("cmap_G", ['black', 'white',
                                                            '#0f0'])  # in denen die channel angezeigt werden
    cmap_B = c.LinearSegmentedColormap.from_list("cmap_B", ['black', 'white', '#00f'])

    boundary_red = max(torch.max(image_tensor[0]), -torch.min(image_tensor[0]))/div_boundary

    f, axarr = plt.subplots(5, 4, figsize=(20, 20))

    img_norm = (image_tensor - torch.min(image_tensor)) / (torch.max(image_tensor) - torch.min(image_tensor))
    # img_norm = image_tensor / torch.max(image_tensor)
    vutils.save_image(img_norm, "" + savepath + "/" + image_name + "-RGB" + ending)
    if single_channel_images:
        mpimage.imsave("" + savepath + "/" + image_name + "-r" + ending, image_tensor[0].detach().cpu(), cmap=cmap_R,
                   vmin=-boundary_red, vmax=boundary_red)

    if channels > 1 and single_channel_images:
        boundary_green = max(torch.max(image_tensor[1]), -torch.min(image_tensor[1]))/div_boundary
        boundary_blue = max(torch.max(image_tensor[2]), -torch.min(image_tensor[2]))/div_boundary
        mpimage.imsave("" + savepath + "/" + image_name + "-g" + ending, image_tensor[1].detach().cpu(), cmap=cmap_G,
                       vmin=-boundary_green, vmax=boundary_green)
        mpimage.imsave("" + savepath + "/" + image_name + "-b" + ending, image_tensor[2].detach().cpu(), cmap=cmap_B,
                       vmin=-boundary_blue, vmax=boundary_blue)

    plt.close()
  
  
  
# helper methods for preprocessing
def calculate_weight(channels, depth, single_color, color_opponency, black_white):
    weight_array = np.ones((channels, depth * 3, 1, 1))

    if depth == 1:
        if not single_color and not color_opponency and not black_white: # Average black-white image
            weight_array[:, :3, :, :] *= 1 / 3
        elif color_opponency and not single_color and not black_white: #color opponency depth 0
            weight_array[0,:,:,:] = 1/3
            for i in range(3):
                weight_array[1, i, 0, 0] = [0.5, -0.5, 0][i]
                weight_array[2, i, 0, 0] = [-0.5 / 3, -0.5 / 3, 1 / 3][i]
        elif black_white and not color_opponency and not single_color: # Luminance black-white image
            print("black_white")
            weight_array[:, 0, :, :] *= 0.299
            weight_array[:, 1, :, :] *= 0.587
            weight_array[:, 2, :, :] *= 0.114
    else:
        weight_array[:, :3, :, :] *= 1 / 3
        weight_array[:, 3:, :, :] *= -(1 / (depth * 3 - 3))

        if channels == 3 and single_color:
            print("single_color")
            for c in range(channels):
                weight_array[c, :, 0, 0] = 0  # Setze alles auf 0
                weight_array[c, c, 0, 0] = 1  # Setze die 1 an die richtige Stelle

                for i in range(1, depth):
                    weight_array[c, i * 3 + c, 0, 0] = -1 / (depth - 1)

        elif channels == 3 and color_opponency:
            print("color_opponency")
            copy = weight_array[0, :, :, :]
            weight_array = np.zeros((channels, depth * 3, 1, 1))
            weight_array[0, :, :, :] = copy
            base_r_g = [0.5, -0.5, 0]
            r_g_value = [-(1 / ((depth - 1) * 2)), (1 / ((depth - 1) * 2)), 0]
            base_b_y = [-0.5 / 3, -0.5 / 3, 1 / 3]
            b_y_value = [(0.5 / (depth * 3 - 3)), (0.5 / (depth * 3 - 3)), -(1 / (depth * 3 - 3))]

            for i in range(depth * 3):
                if i < 3:
                    # Setze die ersten drei Werte direkt auf 0.5, -0.5, 0
                    weight_array[1, i, 0, 0] = base_r_g[i]
                    weight_array[2, i, 0, 0] = base_b_y[i]
                else:
                    # Fülle den Rest mit dem Pattern entsprechend depth
                    index = (i - 3) % 3
                    weight_array[1, i, 0, 0] = r_g_value[index]
                    weight_array[2, i, 0, 0] = b_y_value[index]



    return weight_array

def create_blur_kernel():
    kernel = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            kernel[i, j] = 1 / 9

    return kernel

class BlurPreprocessing(nn.Module):
    def __init__(self, blur_bool, blur_depth, single_color, color_opponency, channels, path, training, black_white):
        super().__init__()
        self.blur = blur_bool
        self.num_images = blur_depth + 1
        self.single_color = single_color
        self.color_opponency = color_opponency
        self.channels = channels
        self.write = False
        self.path = path
        self.training = training
        self.black_white = black_white

        if self.blur:

            blur_kernel = create_blur_kernel()
            self.conv_blur = nn.Conv2d(3, 3 * self.num_images, 3, stride=(1, 1), padding=1, padding_mode='replicate', groups=3, bias=False)

            self.conv_blur.weight = nn.Parameter(tensor(np.array([[blur_kernel],
                                                                  [blur_kernel],
                                                                  [blur_kernel]]), requires_grad=False).float())

            for param in self.conv_blur.parameters():
                param.requires_grad = False

            self.custom_layer = nn.Conv2d(self.num_images * 3, out_channels=channels, kernel_size=1,
                                          stride=1, padding=0, bias=False)

            weight_array = calculate_weight(self.channels, self.num_images, self.single_color, self.color_opponency, self.black_white)
            self.custom_layer.weight = nn.Parameter(tensor(np.array(weight_array), requires_grad=True).float())


            # freezing the preprocessing
            for param in self.custom_layer.parameters():
                param.requires_grad = False

            self.change_channel_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0)

            print("preprocessing")
            print(self.conv_blur.weight)
            print(self.custom_layer.weight)

    def forward(self, x, save_name="image", sig=True):
        if self.blur:


            concat_image = x

            for i in range(self.num_images - 1):
                x = self.conv_blur(x)
                concat_image = torch.concat([concat_image, x], dim=1)

            x = self.custom_layer(concat_image)

            if self.channels == 1:
                x = self.change_channel_layer(x)


        output_tensor = x
        print(f"Output tensor shape: {output_tensor.shape}")
        if sig:
            save_name = save_name + "_sig"          
            # save a copy as png (normalized)
            output_img = x[0].detach().cpu().numpy()
            
            #output_img = np.transpose(output_img, (1, 2, 0))  # (H, W, C)

            boundary = abs(output_img).max()
            # normalize to 0-1 with 0->0.5
            output_img = (output_img + boundary) / (boundary + boundary + 1e-8)
            
            # clip between 0.4 and 0.6
            output_img = np.clip(output_img, 0.47, 0.53)
            
            # rescale clipped range to 0-1
            output_img = (output_img - 0.47) / (0.53 - 0.47)
            output_tensor = torch.tensor(output_img).unsqueeze(0)
            print(f"Output image shape: {output_tensor.shape}")
            
            #rescale to -1 to 1
            output_tensor = output_tensor * 2 - 1

        if self.write:
            if(save_name=="image"):
                if(self.color_opponency):
                    save_name = "color_opponency"
                if self.single_color:
                    save_name = "single_color"
                if self.black_white:
                    save_name = "black_white"
            
            

            
            #save_image(output_tensor, "sig", save_name, self.channels, self.path, self.training, self.single_color, self.color_opponency, False)
            #imageio.imwrite(f"{save_name}_after.png", output_img)
            # convert to 0-255 and uint8
            #output_img = (output_img * 255).astype(np.uint8)
            
            if(self.color_opponency):
                print("saving image after preprocessing")
                save_image_co(output_tensor, "result", save_name, self.channels, self.path, True)
            if self.single_color:
                print("saving image after preprocessing")
                save_image_rgb(output_tensor, "result", save_name, self.channels, self.path, True)
            if self.black_white:
                print("saving image after preprocessing")
                save_image_bw(output_tensor, "result", save_name, self.channels, self.path, True)
        



        return output_tensor #x



class SparsityPreprocessing(nn.Module):
    def __init__(self, sparsity_type, sparsity_threshold, training):
        super().__init__()
        self.sparsity_type = sparsity_type
        self.sparsity_threshold = sparsity_threshold
        self.training = training

    def forward(self, x):
        if self.sparsity_type is not None:

            if self.sparsity_type == 'percentage':
                num_elements = x.numel()
                k = int(self.sparsity_threshold * num_elements)

                if k > 0:
                    abs_vals = x.abs().flatten()
                    threshold = torch.topk(abs_vals, k, largest=False).values.max()
                    sparse_image = torch.where(x.abs() <= threshold, torch.tensor(0.0, device=x.device), x)
                    x = sparse_image
            else:
                #value based sparsity
                sparse_image = torch.where(x.abs() < self.sparsity_threshold, torch.tensor(0.0, device=x.device), x)
                x = sparse_image

                if not self.training:
                    image_pixel_number = x.numel()
                    number_of_zero_pixels = (sparse_image == 0.0).sum().item()
                    print(f"Eval: Percentage of zero pixels in the sparse image: {number_of_zero_pixels/image_pixel_number}")

        return x


def get_blur_module(blur_bool, blur_depth, single_color, color_opponency, black_white, channels, path, training):
    if blur_bool:
        return BlurPreprocessing(blur_bool, blur_depth, single_color, color_opponency, channels, path, training, black_white)
    else:
        return nn.Identity()
    
    
