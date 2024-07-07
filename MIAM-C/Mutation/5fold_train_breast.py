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
from dataloader_bag import TrainDataset_Fold_Bag, TestDataset
from model import AttentionSlide_Batch, AttentionSlide_MultiBatch
from train_support import train_model, make_folder_list, pinyin, val_slide, make_folder_list_slide
from torch.backends import cudnn as cudnn

random.seed(0)


plt.ion()   # interactive mode

now = datetime.datetime.now()
current_timestamp = now.strftime('%m-%d-%H_%M')

cudnn.benchmark = True
log_dir = "Logs/tumor_OV/"+current_timestamp
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data_dir = 'F:/CQ_Tiles_Normalized'
batchsize = 18
bag_size = 35



epoch = 20
norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]

if not os.path.exists('Checkpoint'):
    os.makedirs('Checkpoint')



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


brca_slide_folder  = glob.glob(data_dir+'/BRCA/*')
wild_slide_folder  = glob.glob(data_dir+'/WILD/*')
kf_wild = KFold(n_splits=5, shuffle=True, random_state=0).split(wild_slide_folder)
kf_brca = KFold(n_splits=5, shuffle=True, random_state=0).split(brca_slide_folder)



#Slide
print('Training: Split by Slide\n')
print('bag_size: ', bag_size)
##生成json文件

for k_fold ,(wild,brca) in enumerate(zip(kf_wild, kf_brca)):
    print(k_fold)
    if len(os.listdir('Val_list')) < 10:
            train_folder, val_folder = make_folder_list_slide(wild, brca, wild_slide_folder, brca_slide_folder, k_fold)

##训练模型           
for i_fold in range (0,5):
    print('i_fold: ', i_fold)
    train_file = 'Val_list/train_folder'+str(i_fold)+'.json'
    with open(train_file, 'r') as f:
        train_folder = json.load(f)

    val_file = 'Val_list/val_folder'+str(i_fold)+'.json'
    with open(val_file, 'r') as f:
        val_folder = json.load(f)
    image_datasets = {'train': TrainDataset_Fold_Bag(train_folder, data_transforms['train'], bag_size, ratio=1), 'val': TrainDataset_Fold_Bag(val_folder, data_transforms['val'], bag_size)}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize, shuffle=True, drop_last = True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    model_ft = AttentionSlide_MultiBatch()

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_ft,milestones=[4,7,9],gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft,mode= 'min', factor= 0.1,patience=1)


    writer = SummaryWriter(log_dir+'/'+str(i_fold))
    model_ft = train_model(model_ft, criterion, optimizer_ft, dataloaders, dataset_sizes, epoch, writer,scheduler)
    torch.save(model_ft.state_dict(), os.path.join('Checkpoint', 'resnet34_fold' + str(i_fold) + '.pt'))
    
    
    

    print('Slide validation\n')
    wild = 0
    brca = 0
    wild_wrong = 0
    brca_wrong = 0
    val_folder_ = list(itertools.chain.from_iterable(val_folder))
    for wsi in val_folder_:
        pred_aggregate = []
        prob_aggregate = []
        
        bag_image_dataset = TestDataset(wsi, data_transforms['val'], bag_size=bag_size)
        bag_image_dataloader = torch.utils.data.DataLoader(bag_image_dataset, batch_size=10, shuffle=False, drop_last = True) 
        
        model_ft.eval()
        with torch.no_grad():
            for bag_image in bag_image_dataloader:
                bag_image = bag_image.to(device)
                outputs, preds = model_ft(bag_image)

                outputs = outputs.cpu().numpy() 
                preds = preds.cpu().numpy()
                pred_aggregate.append(preds)
                prob_aggregate.append(outputs)

                
        if 'wild' in wsi:
            wild += 1
            pred_aggregate_ = list(itertools.chain.from_iterable(pred_aggregate))
            slide_result = np.mean(pred_aggregate_)
            
            if slide_result > 0.5:
                wild_wrong += 1
                
            
        else:
            brca += 1
            pred_aggregate_ = list(itertools.chain.from_iterable(pred_aggregate))
            slide_result = np.mean(pred_aggregate_)
            
            # print(slide_result)
            if slide_result < 0.5:
                brca_wrong += 1

    print('fold: ',i_fold, 'wild: ',wild, 'brca: ',brca, 'wild_wrong: ',wild_wrong, 'brca_wrong: ', brca_wrong)





