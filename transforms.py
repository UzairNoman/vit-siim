'''
Here we want to define all the transformations that we need to apply to the image.
The idea is to be able to call them in one go with ComposeTransform class (look at the bottom of this file). This will be later used by data generator

- input: 1 image obtained by calling Cifar100 class. (I do not know yet what shape exactly it will have but lets assume numpy 32x32x3)
- output: list of patches transformed (numpy array of dimensionality patches_num x 32 x 32 x 3)

what kind of transforms do we need:
- resizing (some class that would perform resizing of the image). i would say input and output dimensionalities should be both parameters (and not fixed 32x32)
- cutting into patches (Patches class)
- maybe normalization? but let's treat it as second priority, because maybe we can just add 1 special layer at the beginning of the model which is doing it for us

so far I just copied couple of classes that might be a good start
again, this is heavily inspired by I2DL, exercise 1
'''


import numpy as np
import torch
import torch.nn.functional as F


### MAYBE APPLY TRANSFORMS ONCE AND SAVE THE DATA LOCALLY?

class Resizeit:

    def __init__(self, new_size, interpolation='trilinear'):
        self.new_size = new_size
        self.interpolation = interpolation
    
    def __call__(self, img_dict):
        print(img_dict)
        #resized_img = F.interpolate(img_dict['image'], size=self.new_size, mode=self.interpolation)
        #img_dict['image'] = resized_img
        return img_dict


# class Patches:

#     def __init__(self, patch_num):
#         self.patch_num = patch_num
    
#     def __call__(self, img_dict):

#         def img_to_patch(x):
#             """
#             Inputs:
#                 x - torch.Tensor representing the image of shape [B, C, H, W]
#                 patch_size - Number of pixels per dimension of the patches (integer)

#             """
#             B, C, H, W = x.shape
#             H_new = H // self.patch_num * self.patch_num
#             W_new = W // self.patch_num * self.patch_num
#             H_pixels = int(H_new / self.patch_num)
#             W_pixels = int(W_new / self.patch_num)
#             x = x[:, :, :H_new, :W_new]
#             #x = x.reshape(B, H_new//H_pixels, H_pixels, W_new//W_pixels, W_pixels, C)
#             x = x.reshape(B, C, self.patch_num, H_pixels, self.patch_num, W_pixels)
#             x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
#             x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
#             return x

#         image = img_to_patch(img_dict['image'])
#         #label = img_dict['label'].unsqueeze(1).expand(-1, image.shape[1])
#         label = img_dict['label']
#         #full_tensor1 = torch.full((), img_dict['label'])
#         img_dict['image'], img_dict['label'] = image, label

#         return img_dict


# class RescaleTransform:
#     """Transform class to rescale images to a given range"""
#     def __init__(self, out_range=(0, 1), in_range=(0, 255)):
#         """
#         :param out_range: Value range to which images should be rescaled to
#         :param in_range: Old value range of the images
#             e.g. (0, 255) for images with raw pixel values
#         """
#         self.min = out_range[0]
#         self.max = out_range[1]
#         self._data_min = in_range[0]
#         self._data_max = in_range[1]

#     def __call__(self, img_dict):
#         '''
#         Rescales given image
#            - from (self._data_min, self._data_max)
#            - to (self.min, self.max)
#         '''
#         images = img_dict['image']
#         #image.shape is (2056, 2124, 3) actual image size
#         images -= self._data_min
#         images /= (self._data_max - self._data_min) / (self.max - self.min)
#         images += self.min
#         img_dict['image'] = images

#         return img_dict

# ####################################################

# class PassThroughCNN:

#     def __init__(self, model):
#         self.model = model

#     def __call__(self, img_dict):
#         #img_dict['image'] is of [4,3,224,224]
#         image_before_cnn = torch.tensor(img_dict['image'], dtype=torch.float32)
#         images = self.model.forward(image_before_cnn)
#         #images is of [4,512,7,7]
#         img_dict['image'] = images
#         return img_dict

# class ReshapeToTensor:
#     """Transform class to reshape images to a 32x32x3"""
#     def __init__(self):
#         pass

#     def __call__(self, img_dict):
#         image = img_dict['image']
#         #image.shape is (3072,) for Cifar100
#         image = image.reshape(3,32,32).transpose(1,2,0)
#         img_dict['image'] = image
#         return img_dict

# class Patches_new:

#     def __init__(self, patch_num):
#         self.patch_num = patch_num
    
#     def __call__(self, img_dict):

#         def img_to_patch(x):
#             """
#             Inputs:
#                 x - torch.Tensor representing the image of shape [B, C, H, W]
#                 patch_size - Number of pixels per dimension of the patches (integer)

#             """
#             x = x[None, :]
#             x = torch.tensor(x)

#             B, H, W, C = x.shape
#             H_new = H // self.patch_num * self.patch_num
#             W_new = W // self.patch_num * self.patch_num
#             H_pixels = int(H_new / self.patch_num)
#             W_pixels = int(W_new / self.patch_num)
#             x = x[:, :H_new, :W_new, :]
#             #x = x.reshape(B, H_new//H_pixels, H_pixels, W_new//W_pixels, W_pixels, C)
#             x = x.reshape(B, self.patch_num, H_pixels, self.patch_num, W_pixels, C)
#             x = x.permute(0, 1, 3, 5, 2, 4) # [B, H', W', C, p_H, p_W]
#             x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
#             x = x.squeeze()
#             return x

#         image = img_to_patch(img_dict['image']).numpy()
#         label = np.array([img_dict['label'] for i in image])
#         img_dict['image'], img_dict['label'] = image, label
#         return img_dict

