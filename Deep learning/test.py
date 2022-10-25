# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:33:20 2022

@author: MaB
"""

import torch 
import torch.nn as nn
from utlis import get_surv_time
from lifelines.utils import concordance_index
import wandb
from scipy.ndimage import binary_dilation

def test_epoch(model, test_loader, device, opt, endpoint_info, test_ID):
        model.eval()
        test_outputs_cat = torch.Tensor().to(device)
        with torch.no_grad():
            for test_data in test_loader:
    
                test_images = torch.Tensor().to(device)
                for image in opt.input_modality:
                    test_images = torch.cat((test_images,test_data[image].to(device)), 1)
                    
               
                with torch.no_grad():
                    test_outputs =model(test_images)
                    #test_outputs= nn.Sigmoid()(test_outputs)
                    test_outputs_cat = torch.cat((test_outputs_cat,test_outputs), 0)
                    
            #print (val_outputs_cat,val_outputs_cat.size())
            test_outputs_cat = test_outputs_cat.detach().cpu().numpy()
            
            epoch_score = - test_outputs_cat[:,0]
            
            epoch_time =  list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_time_name])
            epoch_event = list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_name])
            
            metric = concordance_index(epoch_time, epoch_score, epoch_event) # test_cindex
            print ('Test_epoch C-index: ', metric)
            wandb.log({"test_epoch_c-index": metric})
            
def test_epoch_tumor(model, test_loader, device, opt, endpoint_info, test_ID, mode = 'test'):
        model.eval()
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
                    '''
                    if opt.model != 'ViT':
                        test_outputs= nn.Sigmoid()(test_outputs)
                    else:
                        test_outputs= nn.Sigmoid()(test_outputs[0])                       
                    '''
                    test_outputs_cat = torch.cat((test_outputs_cat,test_outputs), 0)
                    
            #print (val_outputs_cat,val_outputs_cat.size())
            test_outputs_cat = test_outputs_cat.detach().cpu().numpy()
            epoch_score = - test_outputs_cat[:,0]
            
            epoch_time =  list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_time_name])
            epoch_event = list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_name])
            #print (str(mode),epoch_score,epoch_time,epoch_event)
            metric = concordance_index(epoch_time, epoch_score, epoch_event) # test_cindex
            print (str(mode) + '_epoch C-index: ', metric)
            wandb.log({str(mode) +"_epoch_c-index": metric})
            
            
def test_epoch_tumor_new(model,model_clc,  test_loader, device, opt, endpoint_info, test_ID, mode = 'test'):
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
                    '''
                    if opt.model != 'ViT':
                        test_outputs= nn.Sigmoid()(test_outputs)
                    else:
                        test_outputs= nn.Sigmoid()(test_outputs[0])       
                    '''    
                    # FCN inputs
                    test_outputs_clc =  model_clc(test_data['clinical_vector'])     
                    #test_outputs_clc = nn.Sigmoid()(test_outputs_clc)   
                     
                    # average CNN and FCN predictions
                    test_outputs =  (test_outputs + test_outputs_clc)/2.
                                                
                    #test_outputs= nn.Sigmoid()(test_outputs)
                    test_outputs_cat = torch.cat((test_outputs_cat,test_outputs), 0)
                    
            #print (val_outputs_cat,val_outputs_cat.size())
            test_outputs_cat = test_outputs_cat.detach().cpu().numpy()
            epoch_score = - test_outputs_cat[:,0]
            
            epoch_time =  list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_time_name])
            epoch_event = list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_name])
            #print (str(mode),epoch_score,epoch_time,epoch_event)
            
            metric = concordance_index(epoch_time, epoch_score, epoch_event) # test_cindex
            print (str(mode) + '_epoch C-index: ', metric)
            wandb.log({str(mode) +"_epoch_c-index": metric})            
            

def test_epoch_tumor_FC_clinical(model, test_loader, device, opt, endpoint_info, test_ID, mode = 'test'):
 

        model.eval()
        test_outputs_cat = torch.Tensor().to(device)
        with torch.no_grad():
            for test_data in test_loader:              
                if opt.model =='FC':
                     test_images = test_data['radiomics_vector'].to(device)    
                if opt.model =='FC_clinical':
                     test_images = test_data['clinical_vector'].to(device)    
                if opt.model =='FC_clinical4':
                     test_images = test_data['clinical_vector'].to(device)        
                     test_images = test_images[:,1:]
                if opt.model =='FC_combine':
                     test_images = test_data['clc_radio_vector'].to(device)                                     
                #test_labels = test_data['label'].to(device)
               
                with torch.no_grad():
                    test_outputs =model(test_images)
                    '''
                    if opt.model != 'ViT':
                        test_outputs= nn.Sigmoid()(test_outputs)
                    else:
                        test_outputs= nn.Sigmoid()(test_outputs[0])   
                    '''
                    test_outputs_cat = torch.cat((test_outputs_cat,test_outputs), 0)
                    
            #print (val_outputs_cat,val_outputs_cat.size())
            test_outputs_cat = test_outputs_cat.detach().cpu().numpy()
            epoch_score = - test_outputs_cat[:,0]
            
            epoch_time =  list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_time_name])
            epoch_event = list(endpoint_info.loc[endpoint_info['PatientID'].isin(test_ID)][opt.event_name])
            #print (str(mode),epoch_score,epoch_time,epoch_event)
            
            metric = concordance_index(epoch_time, epoch_score, epoch_event) # test_cindex
            print (str(mode) + '_epoch C-index: ', metric)
            wandb.log({str(mode) +"_epoch_c-index": metric})               
 