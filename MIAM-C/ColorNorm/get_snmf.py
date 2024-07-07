#随机采样一张WSI的200个tile，生成平均特征非负稀疏矩阵
from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
import spams
import cv2, random, os, glob, json
import Utils.utils as utils
from Utils.vahadane import vahadane
from tqdm import tqdm, trange
from sklearn.manifold import TSNE

def main(Source_rootdir, W_matrix_result_dir, sample_number = 200):
    if not os.path.exists(W_matrix_result_dir):
        os.makedirs(W_matrix_result_dir)
    Source_Folders = glob.glob(Source_rootdir+'/*/*')
    print(Source_Folders[:2])
    vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, getH_mode=0, ITER=50)
    vhd.show_config()

    with tqdm(Source_Folders) as t:
        for Source_Folder in t:
            temp, file_name = os.path.split(Source_Folder)
            npy_file_name = os.path.join(W_matrix_result_dir, file_name+'.npy')
            if os.path.exists(npy_file_name):
                continue
            else:
                t.set_description(file_name)
                Source_img_paths = glob.glob(Source_Folder+'/*.jpg')
                W_matrix = []
                for _ in range(np.minimum(sample_number, len(Source_img_paths)-1)):
                    Source_img_path = Source_img_paths[random.randint(0, len(Source_img_paths)-1)]
                    source_image = utils.read_image(Source_img_path)
                    Ws, Hs = vhd.stain_separate(source_image)
                    W_matrix.append(Ws)

                W_matrix_result =np.mean(W_matrix, axis=0)
                np.save(os.path.join(W_matrix_result_dir, file_name.replace('svs','')), W_matrix_result)

if __name__=="__main__":
    Source_rootdir = r'../ColorNormalization/datasets/TCGA/trainA'#WSI的tile位置
    W_matrix_result_dir = 'W_matrix'#输出矩阵位置
    main(Source_rootdir, W_matrix_result_dir, sample_number = 200)