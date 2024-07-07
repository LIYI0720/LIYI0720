import time
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import cv2, random, os, glob

import soupsieve
import Utils.utils as utils
from Utils.vahadane import vahadane
from tqdm import tqdm, trange
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support

def normalize_img(Source_img_path, vhd, img_t, Ws, Wt, Output_Folder):
    source_image = utils.read_image(Source_img_path)
    output_image = vhd.SPCN_2(source_image, img_t, Ws, Wt)
    RESULT_PATH = os.path.join(Output_Folder, os.path.basename(Source_img_path))
    cv2.imwrite(RESULT_PATH, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

def main(Source_rootdir,Target_Folder,output_rootdir, W_matrix_result_dir, W_matrix_target_dir, Target_img_path):
    Source_Folders = glob.glob(Source_rootdir+'/*/*')

    vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, getH_mode=0, ITER=50)

    _, file_name = os.path.split(Target_Folder)
    W_matrix_target = np.load(os.path.join(W_matrix_target_dir, file_name+'.npy'))
    target_image = utils.read_image(Target_img_path)

    with tqdm(Source_Folders) as pbar:
        for Source_Folder in pbar:
            temp, file_name = os.path.split(Source_Folder)
            _, wsi_type = os.path.split(temp)
            Output_Folder = os.path.join(output_rootdir, wsi_type, file_name)
            if not os.path.exists(Output_Folder):
                os.makedirs(Output_Folder)

            Source_img_paths = glob.glob(Source_Folder+'/*.jpg')
            pbar.set_description(os.path.basename(Source_Folder))
            pbar.set_postfix(length = len(Source_img_paths))
            W_matrix_source = np.load(os.path.join(W_matrix_result_dir, file_name+'.npy'))

            pool = Pool(processes=8)
            pool.map(partial(normalize_img, vhd=vhd, img_t = target_image, Ws = W_matrix_source, Wt = W_matrix_target, Output_Folder = Output_Folder), Source_img_paths)


if __name__=="__main__":
    freeze_support()
    Source_rootdir = r'D:/liyi/output'#WSI的tile位置
    Target_Folder = 'F:/CQ_Tiles/WILD/20-03404-10-chenmenghua-HE-zhongwu'#目标WSI的tile位置
    output_rootdir = r'D:/liyi/TCGA_normalization'#输出归一化的tile文件夹
    W_matrix_result_dir = 'TCGA_matrix'#输入特征矩阵
    W_matrix_target_dir = 'W_matrix'#目标特征矩阵
    Target_img_path = r'F:/CQ_Tiles/WILD/20-03404-10-chenmenghua-HE-zhongwu/20-03404-10-chenmenghua-HE-zhongwu_tumor_50176_64256.jpg'#目标参考tile
    main(Source_rootdir,Target_Folder,output_rootdir, W_matrix_result_dir, W_matrix_target_dir, Target_img_path)