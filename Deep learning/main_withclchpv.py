# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:50:49 2022

This is a DeepSurv loss based net for RFS prediction of OPC cancer, input can be only image (CT/PET) and/or GTV(real), clinical data

@author: MaB 
"""

import logging
import os
import sys
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from utlis import make_surv_array , get_breaks , surv_likelihood, get_surv_time, get_data_dict, get_model, plot_images, get_data_dict_new,get_clc_model
import torch.nn as nn
from para_opts import parse_opts
from torch.optim.lr_scheduler import MultiStepLR
import wandb
from test import test_epoch, test_epoch_tumor_new

import losses

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset, Dataset, DistributedWeightedRandomSampler
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    Compose,
    RandRotate90d,
    Resized,
    ScaleIntensityd,
    EnsureTyped,
    EnsureType,
    RandFlipd,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandRotated,
    RandZoomd,
    RandAffined,
    Rand3DElasticd,
    OneOf,
    CenterSpatialCropd,
    Resized
)

from scipy.ndimage import binary_dilation

def main():
    
    opt = parse_opts()
    opt.usehpv =True
    if opt.resume_id != '': 
          wandb.init(project="Hecktor2022_DeepSurv", id = opt.resume_id, resume = 'must', entity="mbq1137723824")
    else: 
          wandb.init(project="Hecktor2022_DeepSurv", entity="mbq1137723824")
    

    # input_modality = ['CT','PT','gtv']

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set data directory

    opt.result_path = opt.result_path + str(opt.model)  + ' '+ str(opt.event_name) +'_lr' + str(opt.learning_rate)+ '_' + opt.optimizer + '_break(d)_' + str(opt.breaks_interval) + '_bs' + str(opt.batch_size) + '_input_' + str(opt.input_modality)+ '_intype_' + opt.input_type + '_OS_' + str(opt.oversample) + '_pretrain_' + str(opt.pretrain) + '_fold' + str(opt.fold) + '_MDA5_withclchpv'
    
    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)      
    #opt.checkpoint_path = opt.result_path + opt.checkpoint_path  
    
    # load patients ID and endpoint data
    endpoint_path  = opt.endpoint_path
    #endpoint_info = pd.read_csv(endpoint_path) 
    endpoint_info = pd.read_excel(endpoint_path) 
    
    # time to event labels for RFS prediction and processing , and data_dict preperation
    opt.breaks  = get_breaks(break_interval = opt.breaks_interval, longest_fp = endpoint_info[opt.event_time_name].max())
    
    print (opt.breaks,'opt.breaks ' )
   
    
    # load radiomics
    radiomics_vector = pd.read_excel('/data/pg-umcg_mii/data/hecktor2022/Patients after split_withWeight_task2.xlsx')
    radiomics_vector =  radiomics_vector.iloc[:, 15:99]

    
    print (radiomics_vector)
    radiomics_vector =  torch.tensor(np.array(radiomics_vector /  radiomics_vector.max(0)), dtype =torch.float)    
    
    wandb.config.update(opt, allow_val_change=True)
    all_patientsID  = list( endpoint_info['PatientID'] )
     
    test_ID = list(endpoint_info.loc[endpoint_info['CV_MDA5'] == 'val' + str(opt.fold)]['PatientID'])  # Use val as test
    #test_ID = list(endpoint_info.loc[endpoint_info['CV_MDA5'] == 'test']['PatientID'])
    val_ID = list(endpoint_info.loc[endpoint_info['CV_MDA5'] == 'val' + str(opt.fold)]['PatientID'])
    train_ID  = list(endpoint_info.loc[endpoint_info['CV_MDA5'] != 'val' + str(opt.fold)]['PatientID'])
    print ('test_ID', test_ID )
    print ('val_ID', val_ID )
    print ('train_ID', train_ID )
    train_data_dict = get_data_dict(train_ID, opt)    # juest for test
    val_data_dict = get_data_dict(val_ID, opt)  
    test_data_dict = get_data_dict(test_ID, opt)  

    # Define transforms
    train_transforms = Compose([LoadImaged(keys = [image for image in opt.input_modality]),
                                AddChanneld(keys = [image for image in opt.input_modality]) , 
                                #CenterSpatialCropd(keys = [image for image in opt.input_modality] ),
                                #Resized(keys = [image for image in opt.input_modality], spatial_size = (128, 128 ,128)) , 
                                
                                Resized(keys = ['CT','PT','PT_nobrain','gtv','clinical','radiomics'], spatial_size = (128, 128 ,128),  mode = ('trilinear', 'trilinear','trilinear','nearest','nearest','nearest'), align_corners= (True,True,True,None,None,None),  allow_missing_keys = True) , 
                                
                                ScaleIntensityRanged( keys = 'CT', a_min = -200 , a_max = 200, b_min=0, b_max=1, 
                                                     clip= True, allow_missing_keys = True ),
                                ScaleIntensityRanged( keys = 'PT', a_min = 0 , a_max = 25, b_min=0, b_max=1, 
                                                     clip= True , allow_missing_keys = True),
                                ScaleIntensityRanged( keys = 'PT_nobrain', a_min = 0 , a_max = 25, b_min=0, b_max=1, 
                                                     clip= True , allow_missing_keys = True),                                                     
                                RandFlipd(keys=[image for image in opt.input_modality], prob=0.5, spatial_axis=0),
                                RandFlipd(keys=[image for image in opt.input_modality], prob=0.5, spatial_axis=1),
                                RandFlipd(keys=[image for image in opt.input_modality], prob=0.5, spatial_axis=2),                            
                                # shape transforms
                                RandAffined(keys=['CT','PT','PT_nobrain','gtv'],
                                            prob=0.5, translate_range=(7, 7, 7),  
                                            rotate_range=(np.pi / 24, np.pi / 24, np.pi / 24),
                                            scale_range=(0.07, 0.07, 0.07), padding_mode="border", mode = ('bilinear', 'bilinear','bilinear','nearest'), allow_missing_keys = True), 
                                
                                Rand3DElasticd( keys=['CT','PT','PT_nobrain','gtv'],
                                                prob=0.2,
                                                sigma_range=(5, 8),
                                                magnitude_range=(100, 200),
                                                translate_range=(7, 7, 7),
                                                rotate_range=(np.pi / 24, np.pi / 24, np.pi / 24),
                                                scale_range=(0.07, 0.07, 0.07),
                                                padding_mode="border", mode = ('bilinear', 'bilinear','bilinear','nearest'), allow_missing_keys = True),
                                # add noise
                                #RandGaussianNoised(keys =['CT','PT'], prob=0.2, mean=0.0, std=0.1, allow_missing_keys = True ),                                                             
                               ])
    print ([image for image in opt.input_modality])
    val_transforms = Compose([LoadImaged(keys = [image for image in opt.input_modality]),
                                AddChanneld(keys = [image for image in opt.input_modality]) , 
                                #Resized(keys = [image for image in opt.input_modality] , spatial_size = (128, 128 ,128)) , 
                                Resized(keys = ['CT','PT','PT_nobrain','gtv','clinical','radiomics'], spatial_size = (128, 128 ,128),  mode = ('trilinear', 'trilinear','trilinear','nearest','nearest','nearest'), align_corners= (True,True,True,None,None,None),  allow_missing_keys = True) , 
                                #AddChanneld(keys = [image for image in opt.input_modality]) , 
                                #CenterSpatialCropd(keys = [image for image in opt.input_modality]),
                                ScaleIntensityRanged( keys = 'CT', a_min = -200 , a_max = 200, b_min=0, b_max=1, 
                                                     clip= True, allow_missing_keys = True),
                                ScaleIntensityRanged( keys = 'PT', a_min = 0 , a_max = 25, b_min=0, b_max=1, 
                                                     clip= True, allow_missing_keys = True),   
                                ScaleIntensityRanged( keys = 'PT_nobrain', a_min = 0 , a_max = 25, b_min=0, b_max=1, 
                                                     clip= True, allow_missing_keys = True),                                                                                                  
                                ])
    # Define nifti dataset, data loader
    num_workers = 16
    train_ds = Dataset(data=train_data_dict, transform=train_transforms)
    train_ds_test = Dataset(data=train_data_dict, transform=val_transforms)
    # try oversampling 
    #opt.oversample == True
    if opt.oversample:
        label_raw_train = np.array(list(endpoint_info.loc[endpoint_info['PatientID'].isin(train_ID)][opt.event_name]))
        weights = 1/ np.array([np.count_nonzero(1 - label_raw_train), np.count_nonzero(label_raw_train)]) # check event and no events samples numbers
        
        samples_weight = np.array([weights[t] for t in label_raw_train])
        samples_weight = torch.from_numpy(samples_weight)
        
        print ('samples_weight :' , samples_weight)
        #sampler  = DistributedWeightedRandomSampler(samples_weight, len(samples_weight))
        sampler  = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=num_workers, pin_memory=pin_memory, sampler = sampler)
    else:
        
        train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    
    train_loader_test = DataLoader(train_ds_test, batch_size=opt.batch_size, num_workers=num_workers, pin_memory=pin_memory)
    val_ds = Dataset(data=val_data_dict, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, num_workers=num_workers, pin_memory=pin_memory)
    
    test_ds = Dataset(data=test_data_dict, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=num_workers, pin_memory=pin_memory)
    
    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    model = get_model(opt).to(device) 
    
    model_clc =  get_clc_model(opt).to(device) # fully connected model with (84 radiomics and 4 clincial features)
    
    #criterion = surv_likelihood(n_intervals = len(opt.breaks) - 1)
    criterion = losses.NegativeLogLikelihood()
    
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(list(model.parameters()) + list(model_clc.parameters()) , opt.learning_rate, weight_decay = opt.weight_decay)
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(list(model.parameters()) + list(model_clc.parameters()), lr=opt.learning_rate, momentum=opt.momentum, weight_decay = opt.weight_decay)
    scheduler  = MultiStepLR(optimizer , milestones = [200, 300], gamma = 0.2)
    
    # start a typical PyTorch training
    val_interval = 1
    best_metric = - 100000
    best_loss =  100000    
    best_cindex = - 1 
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    model_save_interval = 20 # save model every 20 epochs
    #writer = SummaryWriter(opt.result_path + '/my_experiemnt')
    max_epochs = opt.n_epochs
    
    # train epoch
    patience_number = opt.esn # for early stopping 
    if not opt.no_train:
        
        if wandb.run.resumed:  # reume a training 
            print (opt.checkpoint_path)
            try:
                model_restore = wandb.restore('current_model.pth')
                #print (model_restore)
                checkpoint = torch.load(model_restore.name) 
            except:
                checkpoint = torch.load(opt.result_path + '/' + opt.checkpoint_path)    

            model.load_state_dict(checkpoint['model_state_dict'])
            model_clc.load_state_dict(checkpoint['model_clc_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            opt.begin_epoch = checkpoint['epoch'] + 1   # traning straing from the next epoch of saved epoch
            patience_number = checkpoint['patience_number']
            best_metric = checkpoint['best_metric']
            best_metric_epoch =   checkpoint['best_metric_epoch']
            best_loss =  checkpoint['best_loss']
            best_cindex =  checkpoint['best_cindex']  
            print ('best_loss',best_loss)
            #best_loss =  np.array(best_loss)[0][0]

        for epoch in range(opt.begin_epoch, max_epochs):
            print("-" * 10)
            print(f"epoch {epoch}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0                    
            for batch_data in train_loader:
                step += 1
             
                inputs = torch.Tensor().to(device)
                
                for image in list(set(opt.input_modality) - set(['gtv','clinical','radiomics'])):
                    if opt.input_type == 'tumor':
                        mask  = batch_data['gtv'].to(device)
                        mask[mask > 0] = 1
                        sub_data  = batch_data[image].to(device) * mask
                    if opt.input_type == 'tumor_dilate':
                        mask  = batch_data['gtv'].to(device)
                        mask[mask > 0] = 1
                        mask  = binary_dilation(mask.cpu() , iterations = 5)
                        mask = torch.tensor(mask).to(device)
                        sub_data  = batch_data[image].to(device) * mask                                   
                    if opt.input_type == 'cat':
                        sub_data = batch_data[image].to(device)
                    inputs = torch.cat((inputs,sub_data), 1)
                if 'gtv' in opt.input_modality:
                    if opt.input_type == 'cat':
                       mask  = batch_data['gtv'].to(device)
                       mask[mask > 0] = 1
                       inputs = torch.cat((inputs,mask), 1)       
                    if opt.input_type == 'cat_pt':
                       mask  = batch_data['gtv'].to(device)
                       mask[mask > 1] = 0
                       inputs = torch.cat((inputs,mask), 1)          
                if 'clinical' in opt.input_modality:                    
                       #print ('have Nan is' , torch.isnan(batch_data['clinical'].to(device)).any())
                       '''
                       clicdata =  batch_data['clinical']
                       clicdata[:,:,0:28,:,:] = 0
                       batch_data['clinical'] = clicdata      
                       '''      
                       inputs = torch.cat((inputs,batch_data['clinical'].to(device)), 1)           
                if 'radiomics' in opt.input_modality:                                        
                       radiomicsdata =  batch_data['radiomics']
                       batch_data['radiomics'] = radiomicsdata                           
                       inputs = torch.cat((inputs,batch_data['radiomics'].to(device)), 1)                             

                
                optimizer.zero_grad()
                #print (inputs)
                outputs = model(inputs)
                #print (outputs ,torch.max(outputs), torch.min(outputs))
                #print (outputs)
                #outputs= nn.Sigmoid()(outputs)       
                
                # FCN inputs
                outputs_clc =  model_clc(batch_data['clinical_vector'])     
                #outputs_clc = nn.Sigmoid()(outputs_clc)   
                   
                # average CNN and FCN predictions
                outputs =  (outputs + outputs_clc)/2.
                 
                loss,neglog_loss,l2_norm = criterion(outputs, batch_data['RFS'],batch_data['Recurrence'],  model)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                #writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
                wandb.log({"train_loss": loss.item()})
                
                '''
                # Plotting (training)
                for idx in range(len(inputs)):
                    plot_images(arr_list=[inputs[idx, 0].detach().cpu().numpy()], nr_images=8, figsize=[12*8, 8],
                                cmap_list=['hot'], colorbar_title_list=[''], 
                                filename='/data/pg-umcg_mii/data/hecktor2022/outpred/figures/epoch_{}_{}'.format(epoch, idx), 
                                vmin_list=[0], vmax_list=[1])
                '''    

            test_epoch_tumor_new(model, model_clc, train_loader_test, device, opt, endpoint_info, train_ID, mode ='train')
        
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
            wandb.log({"train_epoch_loss": epoch_loss})
            scheduler.step()
            
            # # Plotting
            # for idx in range(len(inputs)):
            #     plot_images(arr_list=[inputs[idx, 0].detach().cpu().numpy()], nr_images=8, figsize=[12*8, 8],
            #                 cmap_list=['hot'], colorbar_title_list=[''], 
            #                 filename='/data/pg-umcg_mii/data/hecktor2022/outpred/figures/epoch_{}_{}'.format(epoch, idx), 
            #                 vmin_list=[0], vmax_list=[1])
             
            # Validation
            if not opt.no_val:
                if epoch % val_interval == 0:
                    
                    epoch_score = [] # predicted time            
                    model.eval()
                    model_clc.eval()
                    val_outputs_cat = torch.Tensor().to(device)
                    
                    
                    val_epoch_loss = 0
                    val_step = 0
                    
                    for val_data in val_loader:
                        
                        val_step += 1
                        
                        val_images = torch.Tensor().to(device)
                        '''
                        for image in opt.input_modality:
                            val_images = torch.cat((val_images,val_data[image].to(device)), 1)
                        '''                               
                        for image in list(set(opt.input_modality) - set(['gtv','clinical','radiomics'])):
                            if opt.input_type == 'tumor':
                                mask  = val_data['gtv'].to(device)
                                mask[mask > 0] = 1
                                sub_data  = val_data[image].to(device) * mask
                            if opt.input_type == 'tumor_dilate':
                                mask  = val_data['gtv'].to(device)
                                mask[mask > 0] = 1
                                mask  = binary_dilation(mask.cpu() , iterations = 5)
                                mask = torch.tensor(mask).to(device)
                                sub_data  = val_data[image].to(device) * mask                                   
                                
                            if opt.input_type == 'cat':
                                sub_data = val_data[image].to(device)
                            val_images = torch.cat((val_images,sub_data), 1)
                        if 'gtv' in opt.input_modality:
                           if opt.input_type == 'cat':
                              mask  = val_data['gtv'].to(device)
                              mask[mask > 0] = 1
                              val_images = torch.cat((val_images,mask), 1)                
                           if opt.input_type == 'cat_pt':
                              mask  = val_data['gtv'].to(device)
                              mask[mask > 1] = 0
                              val_images = torch.cat((val_images,mask), 1)               
                              
                        if 'clinical' in opt.input_modality:            
                              # exclude center Info
                              '''
                              clicdata =  val_data['clinical']
                              clicdata[:,:,0:28,:,:] = 0
                              val_data['clinical'] = clicdata
                              '''
                              val_images = torch.cat((val_images,val_data['clinical'].to(device)), 1)                                                                       
                        if 'radiomics' in opt.input_modality:                                        
                              radiomicsdata =  val_data['radiomics']
                              val_data['radiomics'] = radiomicsdata                           
                              val_images = torch.cat((val_images,val_data['radiomics'].to(device)), 1)     
     
                        
                        with torch.no_grad():
                            val_outputs =model(val_images)
                            #val_outputs= nn.Sigmoid()(val_outputs)
                            
                            # FCN inputs
                            val_outputs_clc =  model_clc(val_data['clinical_vector'])     
                            #val_outputs_clc = nn.Sigmoid()(val_outputs_clc)   
                     
                            # average CNN and FCN predictions
                            val_outputs =  (val_outputs + val_outputs_clc)/2.
                            
                            
                            val_outputs_cat = torch.cat((val_outputs_cat,val_outputs), 0)
                            
                            
                            val_loss,val_neglog_loss,val_l2_norm = criterion(val_outputs, val_data['RFS'],val_data['Recurrence'],  model)               
                            val_epoch_loss += val_loss.item()
                        '''
                        # Plotting (validation)
                        for idx in range(len(val_images)):
                            plot_images(arr_list=[val_images[idx, 0].detach().cpu().numpy()], nr_images=8, figsize=[12*8, 8],
                                        cmap_list=['hot'], colorbar_title_list=[''], 
                                        filename='/data/pg-umcg_mii/data/hecktor2022/outpred/figures/epoch_{}_{}_val'.format(epoch, val_step), 
                                        vmin_list=[0], vmax_list=[1])
                        '''   
                    #print (val_outputs_cat,val_outputs_cat.size())
                    val_epoch_loss /= val_step                   
                    wandb.log({"val_epoch_loss": val_epoch_loss})
                    
                    val_outputs_cat = val_outputs_cat.detach().cpu().numpy()
                    epoch_score = - val_outputs_cat
                        
                    epoch_time =  list(endpoint_info.loc[endpoint_info['PatientID'].isin(val_ID)][opt.event_time_name])
                    epoch_event = list(endpoint_info.loc[endpoint_info['PatientID'].isin(val_ID)][opt.event_name])
                    print (epoch_score , epoch_time)
                    metric = concordance_index(epoch_time, epoch_score, epoch_event) # val_cindex
                    metric_values.append(metric)
                    
                    torch.save(model , os.path.join(wandb.run.dir, "model.h5")) # save model architecture to wandb.run.dir
                    
                    torch.save({ # Save our current model, optimizer status to local
                                     'epoch': epoch,
                                      'model_state_dict': model.state_dict(),
                                      'model_clc_state_dict': model_clc.state_dict(),
                                      'optimizer_state_dict': optimizer.state_dict(),
                                      'patience_number' : patience_number,
                                      'best_metric': best_metric,
                                      'best_metric_epoch' : best_metric_epoch,
                                      'best_loss' : best_loss,
                                      'best_cindex' : best_cindex
                                      }, opt.result_path + '/' + opt.checkpoint_path)
                    
                    torch.save({ # Save our current model, optimizer status to wandb cloud
                                     'epoch': epoch,
                                      'model_state_dict': model.state_dict(),
                                      'model_clc_state_dict': model_clc.state_dict(),
                                      'optimizer_state_dict': optimizer.state_dict(),
                                      'patience_number' : patience_number,
                                      'best_metric': best_metric,
                                      'best_metric_epoch' : best_metric_epoch,
                                      'best_loss' : best_loss,
                                      'best_cindex' : best_cindex
                                      }, os.path.join(wandb.run.dir, opt.checkpoint_path))
                    if epoch % model_save_interval == 0:
                            torch.save({ # Save our current model, optimizer status to local
                                     'epoch': epoch,
                                      'model_state_dict': model.state_dict(),
                                      'model_clc_state_dict': model_clc.state_dict(),
                                      'optimizer_state_dict': optimizer.state_dict(),
                                      'patience_number' : patience_number,
                                      'best_metric': best_metric,
                                      'best_metric_epoch' : best_metric_epoch,
                                      'best_loss' : best_loss,
                                      'best_cindex' : best_cindex
                                      }, opt.result_path + '/' + str(epoch)+ '_epoch_model.pth')
                            '''
                            torch.save({ # Save our current model, optimizer status to wandb cloud
                                     'epoch': epoch,
                                      'model_state_dict': model.state_dict(),
                                      'optimizer_state_dict': optimizer.state_dict(),
                                      'patience_number' : patience_number,
                                      'best_metric': best_metric,
                                      'best_metric_epoch' : best_metric_epoch,
                                      'best_loss' : best_loss,
                                      'best_cindex' : best_cindex
                                      }, os.path.join(wandb.run.dir, str(epoch)+ '_epoch_model.pth'))
                           '''
                    if opt.es_metric == 'loss':
                           es_metric = - val_epoch_loss
                    else:
                           es_metric = metric
                        
                    if epoch > 0:                    
                        #if val_epoch_loss < best_loss and metric > best_cindex:
                        if  metric > best_cindex:
                            best_loss = val_epoch_loss
                            best_cindex = metric
                        #if es_metric > best_metric:    
                            best_metric = es_metric
                            best_metric_epoch = epoch
                            torch.save(model.state_dict(), opt.result_path + "/best_metric_model.pth")
                            torch.save(model_clc.state_dict(), opt.result_path + "/best_metric_model_clc.pth")
                            print("saved new best metric model !")
                            wandb.run.summary["best_val-cindex"] = metric
                            wandb.run.summary["best_val-loss"] = val_epoch_loss
                            wandb.run.summary["best_epoch"] = best_metric_epoch
                            patience_number = opt.esn
                        else:
                            patience_number -= 1
            
                    print(f"Current epoch: {epoch} current val_c-index: {metric:.4f} current val_loss: {val_epoch_loss:.4f} ")
                    print(f"Best val_metric: {best_metric:.4f} at epoch {best_metric_epoch}")
                    #writer.add_scalar("val_c-index", metric, epoch)
                    wandb.log({"val_c-index": metric, 'epoch' : epoch})  
                    
                    test_epoch_tumor_new(model, model_clc, test_loader, device, opt, endpoint_info, test_ID, mode ='test')
            if patience_number  <  0:
                print ('Early stopping at epoch' + str(epoch))
                break
        print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        #writer.close()
    if not opt.no_test:
        
        test_ds = Dataset(data=test_data_dict, transform=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=num_workers, pin_memory=pin_memory)
        
        model.load_state_dict(torch.load(opt.result_path + "/best_metric_model.pth"))
        model_clc.load_state_dict(torch.load(opt.result_path + "/best_metric_model_clc.pth"))
        # this is for use intermediate model for test
        '''
        checkpoint = torch.load(opt.result_path + "/80_epoch_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        '''
        model.eval()
        model_clc.eval()
        test_outputs_cat = torch.Tensor().to(device)
        with torch.no_grad():
            for test_data in test_loader:
    
                test_images = torch.Tensor().to(device)
                '''
                for image in opt.input_modality:
                    test_images = torch.cat((test_images,test_data[image].to(device)), 1)
                '''
                for image in list(set(opt.input_modality) - set(['gtv','clinical','radiomics'])):
                            if opt.input_type == 'tumor':
                                mask  = test_data['gtv'].to(device)
                                mask[mask > 0] = 1
                                sub_data  = test_data[image].to(device) * mask
                            if opt.input_type == 'tumor_dilate':
                                mask  = test_data['gtv'].to(device)
                                mask[mask > 0] = 1
                                mask  = binary_dilation(mask.cpu() , iterations = 5)
                                mask = torch.tensor(mask).to(device)
                                sub_data  = test_data[image].to(device) * mask                                
                                
                            if opt.input_type == 'cat':
                                sub_data = test_data[image].to(device)
                            test_images = torch.cat((test_images,sub_data), 1)
                if 'gtv' in opt.input_modality:
                     if opt.input_type == 'cat':
                       mask  = test_data['gtv'].to(device)
                       mask[mask > 0] = 1
                       test_images = torch.cat((test_images,mask), 1)                                  
                     if opt.input_type == 'cat_pt':
                       mask  = test_data['gtv'].to(device)
                       mask[mask > 1] = 0
                       test_images = torch.cat((test_images,mask), 1)       
                if 'clinical' in opt.input_modality:     
                       '''     
                       clicdata =  test_data['clinical']
                       clicdata[:,:,0:28,:,:] = 0
                       test_data['clinical'] = clicdata    
                       '''      
                       test_images = torch.cat((test_images,test_data['clinical'].to(device)), 1)       
                       
                if 'radiomics' in opt.input_modality:                                        
                       radiomicsdata =  test_data['radiomics']
                       test_data['radiomics'] = radiomicsdata                           
                       test_images = torch.cat((test_images,test_data['radiomics'].to(device)), 1)                                                  
               
               
                with torch.no_grad():
                    test_outputs =model(test_images)
                    #test_outputs= nn.Sigmoid()(test_outputs)
                    
                    # FCN inputs
                    test_outputs_clc =  model_clc(test_data['clinical_vector'])     
                    #test_outputs_clc = nn.Sigmoid()(test_outputs_clc)   
                     
                    # average CNN and FCN predictions
                    test_outputs =  (test_outputs + test_outputs_clc)/2.
                    
                    test_outputs_cat = torch.cat((test_outputs_cat,test_outputs), 0)
                    
            #print (val_outputs_cat,val_outputs_cat.size())
            test_outputs_cat = test_outputs_cat.detach().cpu().numpy()
            
            epoch_score  = - test_outputs_cat
            
            epoch_time =  list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_time_name])
            epoch_event = list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_name])
            
            metric = concordance_index(epoch_time, epoch_score, epoch_event) # test_cindex
            
            print ('Final Test C-index: ', metric)
            wandb.log({"Final test_c-index": metric})
            wandb.run.summary["Final test_c-index"] = metric
            wandb.finish()

if __name__ == '__main__':
    main()
     
