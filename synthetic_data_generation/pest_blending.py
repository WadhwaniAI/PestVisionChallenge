import numpy as np
import torch
import torch.optim as optim
from PIL import Image
#from skimage.io import imsave
#from torchvision.utils import save_image
from utils_deep_image_blending import compute_gt_gradient, make_canvas_mask, numpy2tensor, laplacian_filter_tensor, \
                  MeanShift, Vgg16, gram_matrix
#import pdb
import os
import torch.nn.functional as F
import json
import natsort
import glob
import re
from tqdm import tqdm
import argparse
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from skimage.io import imsave
from data_utils.foreground_loader import AbstractForegroundPestDataset
from data_utils.background_loader import AbstractBackgroundDataset
import cv2
from libcom import ImageHarmonizationModel


# abstract class for synthetic data generation: pest blending
# concrete classes for different pest blending methods

class AbstractPestBlending(ABC):
    
    def __init__(self, 
                 outputImagesDir: str, 
                 outputLabelsDir: str, 
                 outputMetadataDir: str, 
                 device: str,
                 max_pests_per_image: int):
        
        self.outputImagesDir = outputImagesDir
        self.outputLabelsDir = outputLabelsDir
        self.outputMetadataDir = outputMetadataDir
        self.device = device
        self.max_pests = max_pests_per_image

    @abstractmethod
    def generate_blended_image(self,
                               foreground_dataset: AbstractForegroundPestDataset, 
                               N_foreground: int,
                               background_dataset: AbstractBackgroundDataset,
                               N_background: int,
                               split: str):
        pass


class DeepImageBlending(AbstractPestBlending):

    def __init__(self, 
                 outputImagesDir: str, 
                 outputLabelsDir: str, 
                 outputMetadataDir: str, 
                 device: str,
                 max_pests_per_image: int = 10,
                 target_image_size: int = 512,
                 source_image_range_big: Tuple[int, int] = (40, 60),
                 source_image_range_small: Tuple[int, int] = (60, 80),
                 num_steps1: int = 1000,
                 num_steps2: int = 0):
        
        super().__init__(outputImagesDir, outputLabelsDir, outputMetadataDir, device, max_pests_per_image)
        
        self.source_image_range_small = source_image_range_small
        self.source_image_range_big = source_image_range_big
        self.ts = target_image_size
        self.num_steps1 = num_steps1
        self.num_steps2 = num_steps2

        for split in ['train', 'val', 'test']:

            os.makedirs(os.path.join(self.outputImagesDir, split), exist_ok = True)
            os.makedirs(os.path.join(self.outputLabelsDir, split), exist_ok = True)
            os.makedirs(os.path.join(self.outputMetadataDir, split), exist_ok = True)


    def generate_blended_image(self, 
                               foreground_dataset: AbstractForegroundPestDataset,
                               N_foreground: int,
                               background_dataset: AbstractBackgroundDataset,
                               N_background: int,
                               split: str,
                               file_save_index: int):
        
        background_sample = background_dataset[np.random.randint(N_background)]
        target_img = np.array(background_sample['image'])
        target_filename = background_sample['image_filename']
        no_pests = np.random.randint(low = 0, high = self.max_pests + 1)   

        if no_pests == 0: #no labels or metadata stored

            no_pest_img_file = os.path.join(self.outputImagesDir, split ,f'{file_save_index}.png')
            imsave(no_pest_img_file, target_img.astype(np.uint8))

            str_metadata = f"{file_save_index}.png {0}/{no_pests} {target_filename}\n" 

            metadata_path = os.path.join(self.outputMetadataDir, split, "metadata.txt")
            with open(metadata_path, 'a') as f:
                f.write(str_metadata)

        centers = np.random.randint(low = self.source_image_range_small[1], high = self.ts -  self.source_image_range_small[1], size = (no_pests, 2))

        for j in tqdm(range(no_pests), desc= " Iterating over no_pests", leave = False):
                                
                foreground_sample = foreground_dataset[np.random.randint(N_foreground)]

                source_img = np.array(foreground_sample["source_img_resized"])
                source_filename = foreground_sample['source_filename']
                mask_img = np.array(foreground_sample["mask_img_resized"])
                mask_img[mask_img>0] = 1
                pest_class_id = foreground_sample["pest_class_id"]
                ss = foreground_sample["source_size"]
                is_big = foreground_sample["is_big"]

                #print(f"shapes source, mask, target: {source_img.shape}, {mask_img.shape}, {target_img.shape}")
                #x_start, y_start = config["input_data"]["x_center"], config["input_data"]["y_center"]
                x_start = centers[j][0]
                y_start = centers[j][1]
                #need to check -> in the image x_start and y_start looks interchanged


                ###################################
                ########### First Pass ###########
                ###################################

                # Default weights for loss functions in the first pass
                grad_weight = 1e4; style_weight = 1e4; content_weight = 1; tv_weight = 1e-6


                # Make Canvas Mask
                # print(x_start, y_start, target_img.shape, mask_img.shape)

                canvas_mask = make_canvas_mask(x_start, y_start, target_img, mask_img)
                canvas_mask = numpy2tensor(canvas_mask, self.device)
                canvas_mask = canvas_mask.squeeze(0).repeat(3,1).view(3, self.ts, self.ts).unsqueeze(0)

                # Compute Ground-Truth Gradients
                gt_gradient = compute_gt_gradient(x_start, y_start, source_img, target_img, mask_img, self.device)

                # Convert Numpy Images Into Tensors
                source_img = torch.from_numpy(source_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(self.device)
                target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(self.device)
                input_img = torch.randn(target_img.shape).to(self.device)

                mask_img = numpy2tensor(mask_img, self.device)
                mask_img = mask_img.squeeze(0).repeat(3,1).view(3,ss,ss).unsqueeze(0)

                # Define LBFGS optimizer
                def get_input_optimizer(input_img):
                    optimizer = optim.LBFGS([input_img.requires_grad_()])
                    return optimizer
                optimizer = get_input_optimizer(input_img)

                # Define Loss Functions
                mse = torch.nn.MSELoss()

                # Import VGG network for computing style and content loss
                mean_shift = MeanShift(self.device)
                vgg = Vgg16().to(self.device)


                run = [0]
                while run[0] < self.num_steps1:
                    
                    def closure():
                        # Composite Foreground and Background to Make Blended Image
                        blend_img = torch.zeros(target_img.shape).to(self.device)
                        blend_img = input_img*canvas_mask + target_img*(canvas_mask-1)*(-1) 
                        
                        # Compute Laplacian Gradient of Blended Image
                        pred_gradient = laplacian_filter_tensor(blend_img, self.device)
                        
                        # Compute Gradient Loss
                        grad_loss = 0
                        for c in range(len(pred_gradient)):
                            grad_loss += mse(pred_gradient[c], gt_gradient[c])
                        grad_loss /= len(pred_gradient)
                        grad_loss *= grad_weight
                        
                        # Compute Style Loss
                        target_features_style = vgg(mean_shift(target_img))
                        target_gram_style = [gram_matrix(y) for y in target_features_style]
                        
                        blend_features_style = vgg(mean_shift(input_img))
                        blend_gram_style = [gram_matrix(y) for y in blend_features_style]
                        
                        style_loss = 0
                        for layer in range(len(blend_gram_style)):
                            style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
                        style_loss /= len(blend_gram_style)  
                        style_loss *= style_weight           

                        
                        # Compute Content Loss
                        blend_obj = blend_img[:,:,int(x_start-source_img.shape[2]*0.5):int(x_start+source_img.shape[2]*0.5), int(y_start-source_img.shape[3]*0.5):int(y_start+source_img.shape[3]*0.5)]
                        source_object_features = vgg(mean_shift(source_img*mask_img))
                        blend_object_features = vgg(mean_shift(blend_obj*mask_img))
                        content_loss = content_weight * mse(blend_object_features.relu2_2, source_object_features.relu2_2)
                        content_loss *= content_weight
                        
                        # Compute TV Reg Loss
                        tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + \
                                torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :]))
                        tv_loss *= tv_weight
                        
                        # Compute Total Loss and Update Image
                        loss = grad_loss + style_loss + content_loss + tv_loss
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # Print Loss
                        # if run[0] % 100 == 0:
                        #     print("run {}:".format(run))
                        #     print('grad : {:4f}, style : {:4f}, content: {:4f}, tv: {:4f}'.format(\
                        #                 grad_loss.item(), \
                        #                 style_loss.item(), \
                        #                 content_loss.item(), \
                        #                 tv_loss.item()
                        #                 ))
                        #     print()
                        
                        run[0] += 1
                        return loss
                    
                    optimizer.step(closure)

                # clamp the pixels range into 0 ~ 255
                input_img.data.clamp_(0, 255)

                # Make the Final Blended Image
                blend_img = torch.zeros(target_img.shape).to(self.device)
                blend_img = input_img*canvas_mask + target_img*(canvas_mask-1)*(-1) 
                blend_img_np = blend_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]

                if self.num_steps1 > 0:
                    # Save image from the first pass

                    first_pass_img_file = os.path.join(self.outputImagesDir, split, f'{file_save_index}.png')
                    imsave(first_pass_img_file, blend_img_np.astype(np.uint8))

                    str_metadata = f"{file_save_index}.png {j+1}/{no_pests} {target_filename} {source_filename} {(y_start, x_start)}\n" #interchanged x and y start

                    metadata_path = os.path.join(self.outputMetadataDir, split, "metadata.txt")
                    with open(metadata_path, 'a') as f:
                        f.write(str_metadata)
                        f.close()

                    label_path = os.path.join(self.outputLabelsDir, split, f'{file_save_index}.txt')
                    with open(label_path, 'a') as f:
                        f.write(f"{pest_class_id} {y_start/self.ts} {x_start/self.ts} {ss/self.ts} {ss/self.ts}\n") #interchanged x and y start
            
                target_img = np.array(Image.open(first_pass_img_file).convert('RGB').resize((self.ts, self.ts)))


class LibcomImageHarmonization(AbstractPestBlending):

    def __init__(self, 
                 outputImagesDir: str, 
                 outputLabelsDir: str, 
                 outputMetadataDir: str, 
                 device: str,
                 outputTempDir: str,
                 max_pests_per_image: int = 10,
                 target_image_size: int = 512,
                 source_image_range_big: Tuple[int, int] = (40, 60),
                 source_image_range_small: Tuple[int, int] = (60, 80),
                 model_type: str = "PCTNet"):
        
        super().__init__(outputImagesDir, outputLabelsDir, outputMetadataDir, device, max_pests_per_image)

        self.source_image_range_small = source_image_range_small
        self.source_image_range_big = source_image_range_big
        self.ts = target_image_size
        self.model_type = model_type
        self.outputTempDir = outputTempDir
        self.harmonization_model = ImageHarmonizationModel(device= self.device, model_type= model_type)  

        for split in ['train', 'val', 'test']:

            os.makedirs(os.path.join(self.outputImagesDir, split), exist_ok = True)
            os.makedirs(os.path.join(self.outputLabelsDir, split), exist_ok = True)
            os.makedirs(os.path.join(self.outputMetadataDir, split), exist_ok = True)
            os.makedirs(os.path.join(self.outputTempDir, split), exist_ok = True)

    def create_composite(self, bg_path, fg_path, mask_path, bbox):
        # Load the background, foreground, and mask
        bg = cv2.imread(bg_path)
        fg = cv2.imread(fg_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure the mask is boolean
        mask_bool = mask.astype(bool)

        # Adjust the size of fg and mask to fit the bbox
        fg_resized = cv2.resize(fg, (bbox[2]-bbox[0], bbox[3]-bbox[1]))
        mask_resized = cv2.resize(mask, (bbox[2]-bbox[0], bbox[3]-bbox[1]))

        # Create composite image: place fg on bg according to bbox
        bg[bbox[1]:bbox[3], bbox[0]:bbox[2]][mask_resized > 0] = fg_resized[mask_resized > 0]

        # Create a new mask for the composite
        new_mask = np.zeros_like(bg[:, :, 0])
        new_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask_resized
        return bg, new_mask

    def generate_blended_image(self, 
                               foreground_dataset: AbstractForegroundPestDataset,
                               N_foreground: int,
                               background_dataset: AbstractBackgroundDataset,
                               N_background: int,
                               split: str,
                               file_save_index: int):
        
        background_sample = background_dataset[np.random.randint(N_background)]
        target_img = np.array(background_sample['image'])
        target_filename = background_sample['image_filename']

        no_pests = np.random.randint(low = 0, high = self.max_pests + 1)   

        if no_pests == 0:

            no_pest_img_file = os.path.join(self.outputImagesDir, split ,f'{file_save_index}.png')
            cv2.imwrite(no_pest_img_file, target_img.astype(np.uint8))

            str_metadata = f"{file_save_index}.png {0}/{no_pests} {target_filename}\n" 

            metadata_path = os.path.join(self.outputMetadataDir, split, "metadata.txt")
            with open(metadata_path, 'a') as f:
                f.write(str_metadata)

        bbox_corners = np.random.randint(low = 0, high = self.ts - self.source_image_range_small[1], size = (no_pests, 2)) #top-left corners

        for j in tqdm(range(no_pests), desc= " Iterating over no_pests", leave = False):
                                
                foreground_sample = foreground_dataset[np.random.randint(N_foreground)]

                source_img = np.array(foreground_sample["source_img_resized"])
                source_filename = foreground_sample['source_filename']
                mask_img = np.array(foreground_sample["mask_img_resized"])
                #mask_img[mask_img>0] = 1
                pest_class_id = foreground_sample["pest_class_id"]
                ss = foreground_sample["source_size"]
                is_big = foreground_sample["is_big"]

                bbox = [bbox_corners[j][0], bbox_corners[j][1], bbox_corners[j][0] + ss, bbox_corners[j][1] + ss]

                try:

                    tmp_source_img = os.path.join(self.outputTempDir, split, 'source_img.png')
                    tmp_mask_img = os.path.join(self.outputTempDir, split, 'mask_img.png')
                    tmp_target_img = os.path.join(self.outputTempDir, split, 'target_img.png')

                    cv2.imwrite(tmp_source_img, source_img)
                    cv2.imwrite(tmp_mask_img, mask_img)
                    cv2.imwrite(tmp_target_img, target_img)
                    
                    composite_img = self.create_composite(tmp_target_img, tmp_source_img, tmp_mask_img, bbox)
                    cut_paste_img = composite_img[0]
                    mask_img = composite_img[1]
                    
                    comp_img = self.harmonization_model(cut_paste_img, mask_img)
                    comp_img_file = os.path.join(self.outputImagesDir,split, f'{file_save_index}.png')
                    cv2.imwrite(comp_img_file, comp_img)

                    
                    # comp_img = draw_bbox_on_image(comp_img, bbox)
                    # grid_img = make_image_grid([tmp_target_img, t,p_source_img, cut_paste_img, mask_img,comp_img])
                    # grid_img_file = os.path.join(outputGrid_folder,split, f'run{run}_{i}_{j}.png')
                    # cv2.imwrite(grid_img_file, grid_img)
                    
                    
                    str_metadata = f"{file_save_index}.png {j+1}/{no_pests} {target_filename} {source_filename} {(bbox[0], bbox[1], bbox[2], bbox[3])}\n" 
                    metadata_path = os.path.join(self.outputMetadataDir,split, f"metadata.txt")
                    with open(metadata_path, 'a') as f:
                        f.write(str_metadata)
                        f.close()


                    label_path = os.path.join(self.outputLabelsDir,split, f'{file_save_index}.txt')
                    with open(label_path, 'a') as f:
                        x_center = int((bbox[0] + bbox[2])/2)
                        y_center = int((bbox[1] + bbox[3])/2)

                        f.write(f"{pest_class_id} {x_center/self.ts} {y_center/self.ts} {ss/self.ts} {ss/self.ts}\n")

                    target_img = cv2.imread(comp_img_file)
                    #print("Generated image:", comp_img_file)
                
                except Exception as e:
                    print('Exception:', source_filename)
                    print(e)

                