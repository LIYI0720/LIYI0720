import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os, copy, torch, itertools, json, pypinyin, glob
from tqdm import tqdm, trange
from PIL import Image
from torchvision.io import read_image
import torch.nn.functional as F
from dataloader_bag import TestDataset
from sklearn.metrics import accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs, writer, use_amp = True):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    running_step = 0
    early_stopped = 0.00001

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            early_times = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            with tqdm(dataloaders[phase]) as t:
                t.set_description("Epoch = {:d}/{:d}".format(epoch, num_epochs))
                for inputs, labels in t:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            outputs, preds, A = model(inputs)
                            loss = criterion(outputs, labels.long())
                            # try:
                            #     outputs, preds, A = model(inputs)
                            #     loss = criterion(outputs, labels.float())
                            # except:
                            #     outputs, preds, A = model(inputs)
                            #     print("error")
                            #     print(outputs, labels)
                            writer.add_scalar(phase+'loss', loss.item(), running_step)
                            running_step += 1
                            t.set_postfix(batch_loss=loss.item())
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                                if loss.item() < early_stopped:
                                    early_times += 1
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if early_times >= 10:
                        break

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def validation(model, criterion, dataloaders, dataset_sizes, use_amp = True):

    running_step = 0
    # Each epoch has a training and validation phase
    phase = 'val'
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    with tqdm(dataloaders) as t:
        for inputs, labels in t:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs, preds, A = model(inputs)
                    loss = criterion(outputs, labels.long())
                    running_step += 1
                    t.set_postfix(batch_loss=loss.item())
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes
    epoch_acc = running_corrects.double() / dataset_sizes

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))

def make_folder_list_slide(wild, brca, wild_slide_folder, brca_slide_folder, k_fold):
    train_wild = [wild_slide_folder[x] for x in wild[0]]
    val_wild = [wild_slide_folder[x] for x in wild[1]]
    train_brca = [brca_slide_folder[x] for x in brca[0]]
    val_brca = [brca_slide_folder[x] for x in brca[1]]
    train_folder = [train_wild, train_brca]
    val_folder = [val_wild, val_brca]
    # print(train_folder)
    # print('-----------------------')
    # print(val_folder)
    train_folder_ = list(itertools.chain.from_iterable(train_folder))
    val_folder_ = list(itertools.chain.from_iterable(val_folder))
    if not os.path.exists('Val_list'):
        os.makedirs('Val_list')
    with open('Val_list/val_folder'+ str(k_fold) +'.json', 'w') as f:
        json.dump(val_folder, f)
    with open('Val_list/test_folder'+ str(k_fold) +'.json', 'w') as f:
        json.dump(val_folder_, f)
    with open('Val_list/train_folder'+ str(k_fold) +'.json', 'w') as ft:
        json.dump(train_folder, ft)
    
    return train_folder, val_folder

def make_folder_list(wild, brca, wild_slide_folder, brca_slide_folder, pinyin_name, k_fold):
    train_wild = [wild_slide_folder[x] for x in wild[0]]
    val_wild = [wild_slide_folder[x] for x in wild[1]]
    train_brca_name = [pinyin_name[x] for x in brca[0]]
    val_brca_name = [pinyin_name[x] for x in brca[1]]
    train_brca = []
    val_brca = []
    for folder_name in brca_slide_folder:
        flag = 0
        for name in val_brca_name:
            if name in folder_name:
                val_brca.append(folder_name)
                flag = 1
                break
        if flag == 0:
            train_brca.append(folder_name) 

    train_folder = [train_wild, train_brca]
    val_folder = [val_wild, val_brca]
    # print(train_folder)
    # print('-----------------------')
    # print(val_folder)
    train_folder_ = list(itertools.chain.from_iterable(train_folder))
    val_folder_ = list(itertools.chain.from_iterable(val_folder))
    if not os.path.exists('Val_list'):
        os.makedirs('Val_list')
    with open('Val_list/val_folder'+ str(k_fold) +'.json', 'w') as f:
        json.dump(val_folder_, f)
    
    return train_folder, val_folder

def pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s

def create_dataset(val_folder, wsi_number, transform, bag_size):
    wsi_dir = val_folder[wsi_number]
    Test_Dataset = TestDataset(wsi_dir, transform, bag_size)
    return val_folder[wsi_number], Test_Dataset

def val_slide(model_ft, val_folder, transform, k_fold, batchsize, bag_size):
    label = []
    prediction = []
    for num_wsi in range(len(val_folder)):
        wsi_dir, Test_Dataset = create_dataset(val_folder, num_wsi, transform, bag_size)
        if 'WILD' in wsi_dir:
            print('WILD')
            label.append(0)
        else:
            print('BRCA')
            label.append(1)
        wsi_dir = os.path.basename(wsi_dir)
        wsi_dir = wsi_dir.replace('.svs','')
        Test_Dataloader = torch.utils.data.DataLoader(Test_Dataset, batch_size=batchsize, shuffle=False)
        preds = []
        model_ft.eval()
        with torch.no_grad():
            for image, loc in Test_Dataloader:
                outputs = model_ft(image.to(device))
                pred = torch.sigmoid(outputs)
                preds.append(pred.cpu()[:,1].tolist())
        pred_ = list(itertools.chain(*preds))
        slide_prediction = np.mean(pred_)
        print(wsi_dir,'slide prediction: ', slide_prediction)
        prediction.append(np.round(slide_prediction))
    print('Acc:', accuracy_score(label, prediction))

def convert20to40(data_dir, data_output_dir):
    for type_ in ('BRCA','WILD'):
        if not os.path.exists(os.path.join(data_output_dir,type_)):
            os.mkdir(os.path.join(data_output_dir,type_))
        slide_dirs = os.listdir(os.path.join(data_dir,type_))
        for slide_dir in tqdm(slide_dirs):
            slide_output_dir = os.path.join(data_output_dir,type_, slide_dir)
            if not os.path.exists(slide_output_dir):
                os.mkdir(slide_output_dir)
            jpg_dirs = glob.glob(os.path.join(data_dir,type_,slide_dir)+'/*.jpg')
            for jpg_dir in jpg_dirs:
                image_20 = Image.open(jpg_dir)
                jpg_basename = os.path.basename(jpg_dir)
                image_ul = image_20.crop((0,0,256,256))
                image_ur = image_20.crop((256,0,512,256))
                image_ll = image_20.crop((0,256,256,512))
                image_lr = image_20.crop((256,256,512,512))
                ul_name = os.path.join(slide_output_dir,jpg_basename.replace('.jpg','_ul.jpg'))
                image_ul.save(ul_name)
                ur_name = os.path.join(slide_output_dir,jpg_basename.replace('.jpg','_ur.jpg'))
                image_ur.save(ur_name)
                ll_name = os.path.join(slide_output_dir,jpg_basename.replace('.jpg','_ll.jpg'))
                image_ll.save(ll_name)
                lr_name = os.path.join(slide_output_dir,jpg_basename.replace('.jpg','_lr.jpg'))
                image_lr.save(lr_name)

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)