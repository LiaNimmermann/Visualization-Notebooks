import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from mmseg.registry import MODELS
import os
import matplotlib.image as mpimage
import torchvision.utils as vutils
import matplotlib.colors as c
from torch import tensor

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

def save_image(image_tensor, where, save_name, channels, path, training):

    if training:
        savepath = path + "/images"
    else:
        savepath = path + "/images_test"


    os.makedirs(savepath, exist_ok=True)

    image_name = save_name + "_" + where
    # Path(path + "/" + savepath +"/" + save_name).mkdir(parents=True, exist_ok=True)

    cmap_R = c.LinearSegmentedColormap.from_list("cmap_R",
                                                 ['black', 'white', '#f00'])  # hier sind die colormaps definiert,
    cmap_G = c.LinearSegmentedColormap.from_list("cmap_G", ['black', 'white',
                                                            '#0f0'])  # in denen die channel angezeigt werden
    cmap_B = c.LinearSegmentedColormap.from_list("cmap_B", ['black', 'white', '#00f'])

    boundary_red = max(torch.max(image_tensor[0]), -torch.min(image_tensor[0]))

    f, axarr = plt.subplots(5, 4, figsize=(20, 20))

    img_norm = (image_tensor - torch.min(image_tensor)) / (torch.max(image_tensor) - torch.min(image_tensor))
    # img_norm = image_tensor / torch.max(image_tensor)
    axarr[0][0].imshow(img_norm.detach().cpu().permute(1, 2, 0))
    axarr[0][1].imshow(img_norm[0].detach().cpu(), cmap=cmap_R)
    vutils.save_image(img_norm, "" + savepath + "/" + image_name + "RGB.png")
    mpimage.imsave("" + savepath + "/" + image_name + "-r.png", img_norm[0].detach().cpu(), cmap=cmap_R,
                   vmin=-boundary_red, vmax=boundary_red)

    if channels > 1:
        boundary_green = max(torch.max(image_tensor[1]), -torch.min(image_tensor[1]))
        boundary_blue = max(torch.max(image_tensor[2]), -torch.min(image_tensor[2]))
        axarr[0][2].imshow(img_norm[1].detach().cpu(), cmap=cmap_G)
        axarr[0][3].imshow(img_norm[2].detach().cpu(), cmap=cmap_B)
        mpimage.imsave("" + savepath + "/" + image_name + "-g.png", img_norm[1].detach().cpu(), cmap=cmap_G,
                       vmin=-boundary_green, vmax=boundary_green)
        mpimage.imsave("" + savepath + "/" + image_name + "-b.png", img_norm[2].detach().cpu(), cmap=cmap_B,
                       vmin=-boundary_blue, vmax=boundary_blue)

    plt.close()

@MODELS.register_module()
class BlurPreprocessing(nn.Module):
    def __init__(self, blur_bool, blur_depth, single_color, color_opponency, channels, path, training, black_white):
        super().__init__()
        self.blur = blur_bool
        self.num_images = blur_depth + 1
        self.single_color = single_color
        self.color_opponency = color_opponency
        self.channels = channels
        self.write = True
        self.path = path
        self.training = training
        self.black_white = black_white

        if self.blur:

            blur_kernel = create_blur_kernel()
            self.conv_blur = nn.Conv2d(3, 3 * self.num_images, 3, stride=(1, 1), padding=1, groups=3, bias=False)

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

    def forward(self, x):
        if self.blur:

            if self.write:
                print("saving image before preprocessing")
                save_image(x[0], "before", "image", self.channels, self.path, self.training)

            concat_image = x

            for i in range(self.num_images - 1):
                x = self.conv_blur(x)
                concat_image = torch.concat([concat_image, x], dim=1)

            x = self.custom_layer(concat_image)

            if self.channels == 1:
                x = self.change_channel_layer(x)


        if self.write:
            print("saving image after preprocessing")
            save_image(x[0], "after", "image", self.channels, self.path, self.training)
            self.write = False

        return x
