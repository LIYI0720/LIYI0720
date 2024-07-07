import numpy as np
import os, openslide, cv2, torch
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm,trange
from Utils.Utils import make_folder

def main(Output_path,Checkpoint_path, Test_path, device):
    svs_paths = [x.replace('.svs','') for x in os.listdir(Test_path) if '.svs' in x]
    print(len(svs_paths),'\'svs\' files found in folder '+ Test_path)

    model_ft = models.resnet18()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load(Checkpoint_path,map_location=device))
    transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.6799, 0.5269, 0.6096], [0.2699, 0.2932, 0.2703])
        ])

    im_size =256
    level = 1
    Image.MAX_IMAGE_PIXELS = None
    model_ft.eval()
    with torch.no_grad():
        for idx in tqdm(range(0, len(svs_paths))):
            filename = svs_paths[idx]
            filename = filename + '.svs'
            npy_check= os.path.exists(os.path.join('Output', filename.replace('.svs','.npy')))
            if npy_check:
                continue#跳过已生成标注的图片
            else:
                slide = openslide.open_slide(os.path.join(Test_path,filename))
                downsample = int(slide.level_downsamples[level])
                [m,n] = slide.level_dimensions[level]
                image_level2 = slide.read_region((0, 0),level, (m,n))
                image = image_level2.convert("RGB")
                image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                m_level = int(m/im_size)
                n_level = int(n/im_size)
                roi = np.zeros((n_level, m_level))
                ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
                for x in range(m_level):
                    for y in range(n_level):
                        crop_image = th1[y*im_size:(y+1)*im_size, x*im_size:(x+1)*im_size]
                        if np.sum(crop_image)<ret1*255*255*1.2:
                            roi[y,x] = 1
                plt.imshow(roi)
                plt.show()
                roi = roi>0
                # roi = morphology.remove_small_objects(roi, 10)
                # roi = morphology.remove_small_holes(roi, 5)
                # 去除小块和小洞
                labelmaps = roi
                x_len,y_len = labelmaps.shape
                downsample = downsample
                labelmaps_l0 = np.zeros((x_len*downsample,y_len*downsample))
                for x in range(x_len):
                    for y in range(y_len):
                        if labelmaps[x,y] == 1:
                            labelmaps_l0[x*downsample:(x+1)*downsample,y*downsample:(y+1)*downsample] = 1
                pbar = tqdm(range(x_len), leave=False)
                for x in pbar:
                    for y in range(y_len):
                        if labelmaps[x,y] == 1:
                            labelmaps_tiles = np.zeros(downsample**2) 
                            batch_images = torch.zeros((downsample**2,3,256,256))
                            for x1 in range(downsample):
                                for y1 in range(downsample):
                                    image = slide.read_region(((y*downsample+y1)*im_size, (x*downsample+x1)*im_size),0, (im_size,im_size))
                                    image_rgb = image.convert("RGB")
                                    image_rgb = transform_image(image_rgb)
                                    batch_images[x1*downsample+y1,:,:,:] = image_rgb
                            batch_images = batch_images.to(device)
                            outputs = model_ft(batch_images)
                            _, preds = torch.max(outputs, 1)
                            preds = preds.cpu().detach().numpy()
                            labelmaps_tiles = preds
                            labelmaps_tiles = labelmaps_tiles.reshape((downsample,downsample))
                            labelmaps_l0[x*downsample:(x+1)*downsample,y*downsample:(y+1)*downsample] = labelmaps_tiles
                np.save(os.path.join(Output_path, filename.replace('.svs','.npy')), labelmaps_l0)
            
if __name__ == '__main__':
    Output_path = 'Output_annotation'
    make_folder(Output_path)
    Checkpoint_path = r"Checkpoint_roi/resnet18_more_22_01_11.pt"
    Test_path = r'D:/liyi/TCGA_germline and wild_DX'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(Output_path,Checkpoint_path, Test_path, device)