from torch.utils.tensorboard import SummaryWriter
import torch, os, random, glob, json, datetime, itertools
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import  transforms
import torch.nn.functional as F
# from dataloader_bag import TrainDataset_Fold_Bag, TestDataset
from dataloader_bag import TrainDataset_Fold_Bag, TestDataset
from model import AttentionSlide_Batch, AttentionSlide_MultiBatch
from train_support import train_model, make_folder_list, pinyin, val_slide, make_folder_list_slide
from torch.backends import cudnn as cudnn
from tqdm import tqdm

cudnn.benchmark = True
# log_dir = "Logs/Tumor_BRCA/"+current_timestamp
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





bag_size = 35
val_batchsize = 2

norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]



data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
}


# ###生成test的json文件
# def make_test_folder(testfolder):
#     test_list = []
#     for test_i in testfolder:
#         # print(test_i)
#         test_list.append(test_i)
#     with open('F:/Mutation_Predictaion/CQ_classify/Val_list_OV_tumor/test_TCGA_OV_tumor_folder.json', 'w') as ft:
#         json.dump(test_list, ft)
    
#     return test_list



# # # ###test的tile的路径
# data_dir = 'D:/CUCH_BRCA_OV_tiles/TCGA_BRCA_OV_tumor_tiles_nor_test'
# #data_dir = 'D:/CUCH_BRCA_OV_tiles/CUCH_BRCA_OV_tumor_tiles_nor_test'
# test_folder_ = glob.glob(data_dir+'/*/*')
# #Test_Folder = make_test_folder(test_folder_)


print('Slide validation\n')
wild = 0
brca = 0
wild_wrong = 0
brca_wrong = 0


for k_fold in tqdm(range(0,5)):
    wsi_Tclass = []
    pred_list = []
    prob_allslide_mean = []
    wsi_names = []
##第一个是test的json文件的路径，第二个是val的json文件的路径
    val_file = 'F:/Mutation_Predictaion/CQ_classify/Val_list/tcga_test_folder.json'
    
    #val_file = 'F:/Mutation_Predictaion/CQ_classify/Val_list/val_folder'+str(k_fold)+'.json'

    with open(val_file, 'r') as f:
        val_folder = json.load(f)

    ##val的时候要这句，test的时候注释掉
    #val_folder = list(itertools.chain.from_iterable(val_folder))
    

    
    for wsi in val_folder:
        pred_aggregate = []
        prob_aggregate = []
        _,wsi_name = os.path.split(wsi)
        wsi_names.append(wsi_name)
        

        bag_image_dataset = TestDataset(wsi, data_transforms['val'], bag_size=bag_size)
        bag_image_dataloader = torch.utils.data.DataLoader(bag_image_dataset, batch_size=val_batchsize, shuffle=False, drop_last=False) 
        


        ###模型参数的地址
        #Checkpoint_path = 'F:/Mutation_Predictaion/CQ_classify/Checkpoint/resnet34_fold'+str(k_fold)+'.pt'
        Checkpoint_path = 'F:/Mutation_Predictaion/CQ_classify/Checkpoint/resnet34_fold'+str(k_fold)+'.pt'
        model_ft = AttentionSlide_MultiBatch()
        model_ft = model_ft.to(device)
        model_ft.load_state_dict(torch.load(Checkpoint_path,map_location=device))

        model_ft.eval()
        
        with torch.no_grad(): 
            for bag_image in bag_image_dataloader:
                bag_image = bag_image.to(device)
                prob,preds= model_ft(bag_image)

                prob = prob.cpu().numpy()
                prob_aggregate.append(prob)
            
                
                preds = preds.cpu().numpy()
                pred_aggregate.append(preds)

                
                
    
        if 'wild' in wsi:
            wild += 1
            pred_aggregate_ = list(itertools.chain.from_iterable(pred_aggregate))
            slide_result = np.mean(pred_aggregate_)


            wsi_Tclass.append(0)
            
            if slide_result > 0.5:
                wild_wrong += 1
                pred_list.append(1)
            else:
                pred_list.append(0)
        else:
            brca += 1
            pred_aggregate_ = list(itertools.chain.from_iterable(pred_aggregate))
            slide_result = np.mean(pred_aggregate_)

            # prob_aggregate_ = list(itertools.chain.from_iterable(prob_aggregate_brac))
            # prob_result = np.mean(prob_aggregate_)
            # prob_slide.append(prob_result)

            wsi_Tclass.append(1)
        
            if slide_result < 0.5:
                brca_wrong += 1
                pred_list.append(0)
            else:
                pred_list.append(1)

        prob_aggregate = list(itertools.chain.from_iterable(prob_aggregate))


        prob_slide_mean = np.mean(prob_aggregate, axis=0)
        print(wsi_name)

        print(prob_slide_mean)
        prob_allslide_mean.append(prob_slide_mean.tolist())
        



    np_prob = np.array(prob_allslide_mean)
    np_wsi = np.array(wsi_Tclass)
    np_pred = np.array(pred_list)
    np_wsi_names = np.array(wsi_names)

    pd_prob = pd.DataFrame({'wsi_name':np_wsi_names,'True_class': np_wsi, 'probability1': np_prob[:,0],'probability2':np_prob[:,1],'pred_class':np_pred}, columns=['wsi_name','True_class', 'probability1','probability2','pred_class'])

    pd_prob.to_csv('./pro_output_cuch_ov_tumor/{}test_TCGA_ov_tumor_fold0_3000_tiles.csv'.format(k_fold))

