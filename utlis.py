# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:37:32 2022

@author: MaB
"""

import pandas as pd
import numpy as np
import torch
import os
import monai
import matplotlib.pyplot as plt


def make_surv_array(t,f,breaks):
    """Transforms censored survival data into vector format
      Arguments
          t: Array of failure/censoring times.
          f: Censoring indicator. 1 if failed, 0 if censored.
          breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
      Returns
          Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
    """
    n_samples=t.shape[0]
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5*timegap
    y_train = np.zeros((n_samples,n_intervals*2))
    for i in range(n_samples):
        if f[i]: #if failed (not censored)
            y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks[1:]) #give credit for surviving each time interval where failure time >= upper limit
            if t[i]<breaks[-1]: #if failure time is greater than end of last time interval, no time interval will have failure marked
                y_train[i,n_intervals+np.where(t[i]<breaks[1:])[0][0]]=1 #mark failure at first bin where survival time < upper break-point
        else: #if censored
            y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks_midpoint) #if censored and lived more than half-way through interval, give credit for surviving the interval.
    return y_train

def get_breaks(break_interval  , longest_fp ):
    
    longest_fp = 5888
    break_number = int(round (longest_fp / 365)) + 1 
    breaks = [ ] 
    
    for i in range(0, 11): # half year interval 
        breaks.append(i* break_interval)
    for i in range(6, break_number  + 1):
        breaks.append(365 * i) # one year intervak
    return np.array(breaks)
    
def get_new_breaks(break_interval  , longest_fp ):
    '''
    longest_fp = 5888
    break_number = int(round (longest_fp / break_interval)) + 1 
    breaks = [ ] 
    
    for i in range(0, break_number + 1): # internal set to 3 before 3 year
        breaks.append(i* break_interval)
    '''
    breaks = [0, 91.25 , 182.5, 273.75, 365, 456.25, 547.5, 638.75, 730, 821.25, 912.5, 1003.75, 1095, 1277.5, 1460, 1642.5, 1825]    
    return np.array(breaks)

def surv_likelihood(n_intervals):
        """
        Arguments
            n_intervals: the number of survival time intervals
        Returns
            Custom loss function
        """
        def loss(y_pred, y_true):
            """
            Arguments
                y_true: Tensor.
                  First half of the values is 1 if individual survived that interval, 0 if not.
                  Second half of the values is for individuals who failed, and is 1 for time interval during which failure occured, 0 for other intervals.
                  See make_surv_array function.
                y_pred: Tensor, predicted survival probability (1-hazard probability) for each time interval.
            Returns
                Vector of losses for this minibatch.
            """
            cens_uncens = 1. + y_true[:,0:n_intervals] * (y_pred-1.) #component for all individuals
            uncens = 1. - y_true[:,n_intervals:2*n_intervals] * y_pred #component for only uncensored individuals
            epsilon = 1e-7
            loglik = torch.sum(-torch.log(torch.clamp(torch.cat((cens_uncens,uncens), dim=-1), epsilon ,None)), axis=-1) #return -log likelihood
            return torch.mean(loglik)
        return loss
    
def get_surv_time(breaks, surv_prob):
        breaks = np.array(breaks).astype(float)
        surv_prob = np.array(surv_prob).astype(float)
    
        mu = 0 
        for ii in range(1, len(breaks)):
            # old: definitely wrong 
            #mu += np.prod(surv_prob[:ii]) * breaks[ii]
            # new: I think it's right
            mu += np.prod(surv_prob[:ii]) * (breaks[ii] - breaks[ii - 1])
            #print (np.prod(surv_prob[:ii]) * breaks[ii] , len(breaks))
            #print (mu, breaks[ii] - breaks[ii - 1])
        print 
            
        return mu
    
    
     
    
def get_data_dict(patientsID , opt): 
    data_dict = []
    clinical_data = pd.read_excel('/data/pg-umcg_mii/data/hecktor2022/Task2_clic_endpoint_IBM.xlsx')
    
    #clinical_data = clinical_data[['CenterID'	,'Gender' ,'Age',	'Weight', 'Chemotherapy']]
    clinical_data['CenterID'] = clinical_data['CenterID'] / 10.
    clinical_data['Gender'].loc[clinical_data['Gender'] == 'M'] = 1.0
    clinical_data['Gender'].loc[clinical_data['Gender'] == 'F'] = 0.0
    clinical_data['Age'] = clinical_data['Age'] / 100.
    clinical_data['Weight'] = clinical_data['Weight'] / 200.
    clinical_data['Chemotherapy'] = clinical_data['Chemotherapy']/1.
    # hpv
    clinical_data['hpv'] = 2. - clinical_data['HPV.statusorg'] # 0: positive, 1: unknown, 2: negative
    clinical_data['hpv'] = clinical_data['hpv'] /2
    
    # radiomics
    clinical_data['GTVLNcomb_original_shape_SurfaceArea'] = clinical_data['GTVLNcomb_original_shape_SurfaceArea'] /0.3
    clinical_data['GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity'] = clinical_data['GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity'] /15  
    # radiomics risk score
    clinical_data['radiorisk'] = clinical_data['radio_risk_fold' + str(opt.fold)] /2            
    
    for i , pID in enumerate(patientsID):
        #index = np.array(endpoint_info.loc[endpoint_info['PatientID'] == pID].index)[0]
        data_single_dict =  {str(image) : os.sep.join([opt.data_path, pID + '__' + str(image) +'.nii.gz']) for image in opt.input_modality  }
        #data_single_dict['label']= labels[index]
        if opt.usehpv == True:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','hpv']])[0].astype(float)
            #print (clc_data_sub)
        else:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy']])[0].astype(float)
        #print (clc_data_sub)
        
        if opt.useradio == True:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','GTVLNcomb_original_shape_SurfaceArea', 'GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity']])[0].astype(float)
            
        if opt.useradiorisk == True:
                clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','GTVLNcomb_original_shape_SurfaceArea', 'GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity', 'radiorisk']])[0].astype(float)    
        
        
        data_single_dict['clinical_vector']=  torch.tensor(clc_data_sub,dtype = torch.float)

        relapse  = np.array(clinical_data.loc[clinical_data['PatientID'] == pID]['Relapse'])[0].astype(float)
        RFS  = np.array(clinical_data.loc[clinical_data['PatientID'] == pID]['RFS'])[0].astype(float)
        data_single_dict['Recurrence']=  torch.tensor(relapse,dtype = torch.float)
        data_single_dict['RFS']=  torch.tensor(RFS,dtype = torch.float)
        #print (data_single_dict['Recurrence'],data_single_dict['RFS'],pID  )
        data_dict.append(data_single_dict)
    return data_dict

def get_data_dict_new(patientsID , opt, endpoint_info,  radiomics): 
    data_dict = []
    
    clinical_data = pd.read_excel('/data/pg-umcg_mii/data/hecktor2022/Task2_clic_endpoint_IBM.xlsx')
    
    #clinical_data = clinical_data[['CenterID'	,'Gender' ,'Age',	'Weight', 'Chemotherapy']]
    clinical_data['CenterID'] = clinical_data['CenterID'] / 10.
    clinical_data['Gender'].loc[clinical_data['Gender'] == 'M'] = 1.0
    clinical_data['Gender'].loc[clinical_data['Gender'] == 'F'] = 0.0
    clinical_data['Age'] = clinical_data['Age'] / 100.
    clinical_data['Weight'] = clinical_data['Weight'] / 200.
    clinical_data['Chemotherapy'] = clinical_data['Chemotherapy']/1.
    # hpv
    clinical_data['hpv'] = 2. - clinical_data['HPV.statusorg'] # 0: positive, 1: unknown, 2: negative
    clinical_data['hpv'] = clinical_data['hpv'] /2    
    # radiomics
    clinical_data['GTVLNcomb_original_shape_SurfaceArea'] = clinical_data['GTVLNcomb_original_shape_SurfaceArea'] /0.3
    clinical_data['GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity'] = clinical_data['GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity'] /15    
    # radiomics risk score
    clinical_data['radiorisk'] = clinical_data['radio_risk_fold' + str(opt.fold)] /2    
    
    for i , pID in enumerate(patientsID):
        index = np.array(endpoint_info.loc[endpoint_info['PatientID'] == pID].index)[0]
        #print (pID,index)
        data_single_dict =  {str(image) : os.sep.join([opt.data_path, pID + '__' + str(image) +'.nii.gz']) for image in opt.input_modality  }
        
        data_single_dict['radiomics_vector']= radiomics[index]
        if opt.usehpv == True:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','hpv']])[0].astype(float)
            #print (clc_data_sub)
        else:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy']])[0].astype(float)
        if opt.useradio == True:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','GTVLNcomb_original_shape_SurfaceArea', 'GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity']])[0].astype(float)
        if opt.useradiorisk == True:
                clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','GTVLNcomb_original_shape_SurfaceArea', 'GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity', 'radiorisk']])[0].astype(float)    
   
        
        data_single_dict['clinical_vector']=  torch.tensor(clc_data_sub,dtype = torch.float)
        
        data_single_dict['weight_vector']=  torch.tensor(clc_data_sub[2],dtype = torch.float)
        #print (data_single_dict['clinical_vector'])
        #data_single_dict['clc_radio_vector']=  torch.cat((data_single_dict['clinical_vector'], data_single_dict['radiomics_vector']), 0)
        #print (data_single_dict['weight_vector'],data_single_dict['weight_vector'].size(), data_single_dict['radiomics_vector'].size())
        #data_single_dict['clc_radio_vector']=  torch.cat((torch.tensor([data_single_dict['weight_vector']]), data_single_dict['radiomics_vector']), 0)[0:6]
        data_single_dict['clc_radio_vector']=  torch.cat((torch.tensor(data_single_dict['clinical_vector']), data_single_dict['radiomics_vector']), 0)
        
        #print (data_single_dict['clc_radio_vector'])
        
        relapse  = np.array(clinical_data.loc[clinical_data['PatientID'] == pID]['Relapse'])[0].astype(float)
        RFS  = np.array(clinical_data.loc[clinical_data['PatientID'] == pID]['RFS'])[0].astype(float)
        data_single_dict['Recurrence']=  torch.tensor(relapse,dtype = torch.float)
        data_single_dict['RFS']=  torch.tensor(RFS,dtype = torch.float)
        
        data_dict.append(data_single_dict)
    return data_dict


def get_data_dict_trainusingHungSeg(patientsID , opt): 
    data_dict = []
    
    clinical_data = pd.read_excel('/data/pg-umcg_mii/data/hecktor2022/Task2_clic_endpoint_IBM.xlsx')
    
    #clinical_data = clinical_data[['CenterID'	,'Gender' ,'Age',	'Weight', 'Chemotherapy']]
    clinical_data['CenterID'] = clinical_data['CenterID'] / 10.
    clinical_data['Gender'].loc[clinical_data['Gender'] == 'M'] = 1.0
    clinical_data['Gender'].loc[clinical_data['Gender'] == 'F'] = 0.0
    clinical_data['Age'] = clinical_data['Age'] / 100.
    clinical_data['Weight'] = clinical_data['Weight'] / 200.
    clinical_data['Chemotherapy'] = clinical_data['Chemotherapy']/1.    
    # hpv
    clinical_data['hpv'] = 2. - clinical_data['HPV.statusorg'] # 0: positive, 1: unknown, 2: negative
    clinical_data['hpv'] = clinical_data['hpv'] /2    
    
    # radiomics
    clinical_data['GTVLNcomb_original_shape_SurfaceArea'] = clinical_data['GTVLNcomb_original_shape_SurfaceArea'] /0.3
    clinical_data['GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity'] = clinical_data['GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity'] /15        

    # radiomics risk score
    clinical_data['radiorisk'] = clinical_data['radio_risk_fold' + str(opt.fold)] /2    
    
    for i , pID in enumerate(patientsID):
        
        if 'gtv' in opt.input_modality:        
           data_single_dict =  {str(image) : os.sep.join([opt.data_path, pID + '__' + str(image) +'.nii.gz']) for image in list(set(opt.input_modality) - set(['gtv']))  }
           data_single_dict['gtv'] =   os.sep.join(['/data/pg-umcg_mii/yan/seg_predicted_larger/', pID + '__' + 'gtv__gtv_hung' +'.nii.gz'])  # using seg as gtv
        else:      
           data_single_dict =  {str(image) : os.sep.join([opt.data_path, pID + '__' + str(image) +'.nii.gz']) for image in opt.input_modality  }
    
        if opt.usehpv == True:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','hpv']])[0].astype(float)
            #print (clc_data_sub) 
        else:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy']])[0].astype(float)
       
        if opt.useradio == True:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','GTVLNcomb_original_shape_SurfaceArea', 'GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity']])[0].astype(float)               
        if opt.useradiorisk == True:
                clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','GTVLNcomb_original_shape_SurfaceArea', 'GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity', 'radiorisk']])[0].astype(float)    
          
        data_single_dict['clinical_vector']=  torch.tensor(clc_data_sub,dtype = torch.float)
            
        relapse  = np.array(clinical_data.loc[clinical_data['PatientID'] == pID]['Relapse'])[0].astype(float)
        RFS  = np.array(clinical_data.loc[clinical_data['PatientID'] == pID]['RFS'])[0].astype(float)
        data_single_dict['Recurrence']=  torch.tensor(relapse,dtype = torch.float)
        data_single_dict['RFS']=  torch.tensor(RFS,dtype = torch.float)
        #print (data_single_dict['Recurrence'],data_single_dict['RFS'],pID  )
        
        
           
        data_dict.append(data_single_dict)
        #print (data_single_dict)
    return data_dict


def get_data_dict_trainusingHungSeg_finaltest(opt): 
    data_dict = []
    
    #clinical_data = pd.read_csv('/data/pg-umcg_mii/data/hecktor2022_testing/hecktor2022_clinical_info_testing.csv')
    
    clinical_data = pd.read_excel('/data/pg-umcg_mii/data/hecktor2022_testing/TEST_2IBM5Risk.xlsx')
    
    #clinical_data = clinical_data[['CenterID'	,'Gender' ,'Age',	'Weight', 'Chemotherapy']]
    #clinical_data['CenterID'] = clinical_data['CenterID'] / 10.
    clinical_data['Gender'].loc[clinical_data['Gender'] == 'M'] = 1.0
    clinical_data['Gender'].loc[clinical_data['Gender'] == 'F'] = 0.0
    clinical_data['Age'] = clinical_data['Age'] / 100.
    clinical_data['Weight'] = clinical_data['Weight'] / 200.
    clinical_data['Chemotherapy'] = clinical_data['Chemotherapy']/1.    
    # hpv
    #clinical_data['hpv'] = 2. - clinical_data['HPV.statusorg'] # 0: positive, 1: unknown, 2: negative
    #clinical_data['hpv'] = clinical_data['hpv'] /2    
    
    # radiomics
    clinical_data['GTVLNcomb_original_shape_SurfaceArea'] = clinical_data['GTVLNcomb_original_shape_SurfaceArea'] /0.3
    clinical_data['GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity'] = clinical_data['GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity'] /15        

    # radiomics risk score
    clinical_data['radiorisk'] = clinical_data['radio_risk_fold' + str(opt.fold)] /2    
    # read ID from clinical_data_excel
    patientsID = list(clinical_data['PatientID'])
    
    for i , pID in enumerate(patientsID):
        
        if 'gtv' in opt.input_modality:        
           data_single_dict =  {str(image) : os.sep.join(['/data/pg-umcg_mii/data/hecktor2022_testing/resampled_larger/', pID + '__' + str(image) +'.nii.gz']) for image in list(set(opt.input_modality) - set(['gtv']))  }
           data_single_dict['gtv'] =   os.sep.join(['/data/pg-umcg_mii/yan/testset_hungseg_v2/testset_hungseg_v2/', pID + '__' + 'gtv_hung' +'.nii.gz'])  # using seg as gtv
        else:      
           #data_single_dict =  {str(image) : os.sep.join([opt.data_path, pID + '__' + str(image) +'.nii.gz']) for image in opt.input_modality  }
           data_single_dict =  {str(image) : os.sep.join(['/data/pg-umcg_mii/data/hecktor2022_testing/resampled_larger/', pID + '__' + str(image) +'.nii.gz']) for image in opt.input_modality  }
    
    
    
        if opt.usehpv == True:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','hpv']])[0].astype(float)
            #print (clc_data_sub) 
        else:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy']])[0].astype(float)
       
        if opt.useradio == True:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','GTVLNcomb_original_shape_SurfaceArea', 'GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity']])[0].astype(float)               
        if opt.useradiorisk == True:
                clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','GTVLNcomb_original_shape_SurfaceArea', 'GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity', 'radiorisk']])[0].astype(float)    
          
        data_single_dict['clinical_vector']=  torch.tensor(clc_data_sub,dtype = torch.float)
        data_single_dict['PatientID']=  pID
        
        '''
        # no label in real test set
        relapse  = np.array(clinical_data.loc[clinical_data['PatientID'] == pID]['Relapse'])[0].astype(float)
        RFS  = np.array(clinical_data.loc[clinical_data['PatientID'] == pID]['RFS'])[0].astype(float)
        data_single_dict['Recurrence']=  torch.tensor(relapse,dtype = torch.float)
        data_single_dict['RFS']=  torch.tensor(RFS,dtype = torch.float)
        '''          
        #print (data_single_dict)
        data_dict.append(data_single_dict)
        #print (data_single_dict)
    return data_dict

def get_data_dict_trainusingHungSeg_finaltest_inumcg(opt): 
    data_dict = []
    
    #clinical_data = pd.read_csv('/data/pg-umcg_mii/data/hecktor2022_testing/hecktor2022_clinical_info_testing.csv')
    
    clinical_data = pd.read_excel('/data/pg-umcg_mii/data/hecktor2022/outpred/DeepSurv/UMCGsetresults_fromHectorModel.xlsx')
    
    #clinical_data = clinical_data[['CenterID'	,'Gender' ,'Age',	'Weight', 'Chemotherapy']]
    #clinical_data['CenterID'] = clinical_data['CenterID'] / 10.
    #clinical_data['Gender'].loc[clinical_data['Gender'] == 'M'] = 1.0
    #clinical_data['Gender'].loc[clinical_data['Gender'] == 'F'] = 0.0
    
    clinical_data['Gender'] =  1 - clinical_data['GESLACHT_codes']
    
    clinical_data['Age'] = clinical_data['AGE'] / 100.
    clinical_data['Weight'] = clinical_data['Weight'] / 200.
    clinical_data['Chemotherapy'] = clinical_data['chemotherapy']/1.    
    # hpv
    #clinical_data['hpv'] = 2. - clinical_data['HPV.statusorg'] # 0: positive, 1: unknown, 2: negative
    #clinical_data['hpv'] = clinical_data['hpv'] /2    
    
    # radiomics
    clinical_data['GTVLNcomb_original_shape_SurfaceArea'] = clinical_data['original_shape_SurfaceArea'] /0.3
    clinical_data['GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity'] = clinical_data['CT_original_glrlm_RunLengthNonUniformity'] /15        

    # radiomics risk score
    clinical_data['radiorisk'] = clinical_data['radio_risk_fold' + str(opt.fold)] /2    
    # read ID from clinical_data_excel
    patientsID = list(clinical_data['PatientID'])
    
    for i , pID in enumerate(patientsID):
        
        pID_full = 'UMCG-' + str(pID).zfill(7)
        '''
        if 'gtv' in opt.input_modality:        
           data_single_dict =  {str(image) : os.sep.join(['/data/p303924/UMCG_OPC_PET_192/', pID + '__' + str(image) +'.nii.gz']) for image in list(set(opt.input_modality) - set(['gtv']))  }
           data_single_dict['gtv'] =   os.sep.join(['/data/p303924/UMCG_OPC_PET_192/hung_seg/', pID + '__' + 'gtv_hung' +'.nii.gz'])  # using seg as gtv
        else:      
           #data_single_dict =  {str(image) : os.sep.join([opt.data_path, pID + '__' + str(image) +'.nii.gz']) for image in opt.input_modality  }
           data_single_dict =  {str(image) : os.sep.join(['/data/p303924/UMCG_OPC_PET_192/', pID + '__' + str(image) +'.nii.gz']) for image in opt.input_modality  }
        '''
        
        #data_single_dict =  {'CT': os.sep.join(['/data/p303924/UMCG_OPC_PET_192/', pID_full + '_ct.nii.gz'])}
        #data_single_dict['PT_nobrain'] =   os.sep.join(['/data/p303924/UMCG_OPC_PET_192/', pID_full + '_pt.nii.gz'])
        data_single_dict =  {'PT_nobrain': os.sep.join(['/data/p303924/UMCG_OPC_PET_192/', pID_full + '_pt.nii.gz'])}
        data_single_dict['gtv'] =   os.sep.join(['/data/p303924/UMCG_OPC_PET_192/hung_seg/', pID_full + '_DLsegNobrain.nii.gz'])
    
        if opt.usehpv == True:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','hpv']])[0].astype(float)
            #print (clc_data_sub) 
        else:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy']])[0].astype(float)
       
        if opt.useradio == True:
            clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','GTVLNcomb_original_shape_SurfaceArea', 'GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity']])[0].astype(float)               
        if opt.useradiorisk == True:
                clc_data_sub = np.array(clinical_data.loc[clinical_data['PatientID'] == pID][['Gender' ,'Age',	'Weight', 'Chemotherapy','GTVLNcomb_original_shape_SurfaceArea', 'GTVLNcomb_CT_original_glrlm_RunLengthNonUniformity', 'radiorisk']])[0].astype(float)    
          
        data_single_dict['clinical_vector']=  torch.tensor(clc_data_sub,dtype = torch.float)
        data_single_dict['PatientID']=  pID_full
        
        '''
        # no label in real test set
        relapse  = np.array(clinical_data.loc[clinical_data['PatientID'] == pID]['Relapse'])[0].astype(float)
        RFS  = np.array(clinical_data.loc[clinical_data['PatientID'] == pID]['RFS'])[0].astype(float)
        data_single_dict['Recurrence']=  torch.tensor(relapse,dtype = torch.float)
        data_single_dict['RFS']=  torch.tensor(RFS,dtype = torch.float)
        '''          
        #print (data_single_dict)
        data_dict.append(data_single_dict)
        #print (data_single_dict)
    return data_dict

def get_data_dict_testusingHungSeg(patientsID , opt, endpoint_info, labels): 
    data_dict = []
    
    for i , pID in enumerate(patientsID):
        index = np.array(endpoint_info.loc[endpoint_info['PatientID'] == pID].index)[0]
        if 'gtv' in opt.input_modality:        
           data_single_dict =  {str(image) : os.sep.join([opt.data_path, pID + '__' + str(image) +'.nii.gz']) for image in list(set(opt.input_modality) - set(['gtv']))  }
           data_single_dict['gtv'] =   os.sep.join(['/data/pg-umcg_mii/yan/seg_predicted_larger/', pID + '__' + 'gtv__gtv_hung' +'.nii.gz'])  # using seg as gtv
        else:      
           data_single_dict =  {str(image) : os.sep.join([opt.data_path, pID + '__' + str(image) +'.nii.gz']) for image in opt.input_modality  }
        data_single_dict['label']= labels[index]
        #print (index, pID,labels[index] )
        data_dict.append(data_single_dict)
    return data_dict
    
#    special resnet model with less feature maps
from monai.networks.nets.resnet import *
from monai.networks.nets.resnet import  _resnet, ResNetBlock
from typing import Any, Callable, List, Optional, Tuple, Type, Union
def  get_inplanes_less():
     return [64, 128, 128, 128]    
def  get_inplanes_muchless():
     return [16, 32, 64, 128]          
def resnet18less(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 with optional pretrained support when `spatial_dims` is 3.
        
    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.
        
    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", ResNetBlock, [2, 2, 2, 2], get_inplanes_less(), pretrained, progress, **kwargs)       
    
def resnet10less(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 with optional pretrained support when `spatial_dims` is 3.
        
    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.
        
    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet10", ResNetBlock, [1, 1, 1, 1], get_inplanes_less(), pretrained, progress, **kwargs)       
    
def resnet10muchless(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 with optional pretrained support when `spatial_dims` is 3.
        
    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.
        
    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet10", ResNetBlock, [1, 1, 1, 1], get_inplanes_muchless(), pretrained, progress, **kwargs)           

def get_model(opt): 
    
    if opt.input_type == 'cat' or opt.input_type == 'cat_pt': 
       input_channels  = len(opt.input_modality)
    if opt.input_type == 'tumor': 
       input_channels  = len(opt.input_modality) - 1
       '''
       if 'gtv_wei' in opt.input_modality or 'gtv' in opt.input_modality:
            input_channels  = len(opt.input_modality) - 1
       #input_channels  = len(opt.input_modality) - 1
       '''
    if opt.input_type == 'tumor_dilate': 
       #input_channels  = len(opt.input_modality)
       input_channels  = len(opt.input_modality) - 1
    
    if opt.model == 'Densenet121':
        return monai.networks.nets.DenseNet121(spatial_dims=3, in_channels= input_channels, out_channels= 1)
    if opt.model == 'Densenetless':
        return monai.networks.nets.DenseNet121(spatial_dims=3, in_channels= input_channels, out_channels= 1, growth_rate=16, block_config=(6, 6, 12, 8))
    if opt.model == 'SEResNext50':
        return monai.networks.nets.SEResNext50(spatial_dims=3, in_channels= input_channels, num_classes=  1)
    if opt.model == 'resnet18':
        #return monai.networks.nets.ResNet(spatial_dims=3, block = 'ResNetBlock', n_input_channels = len(opt.input_modality), num_classes= len(opt.breaks) - 1).resnet18()
        model =  monai.networks.nets.resnet18( spatial_dims=3, n_input_channels = input_channels, num_classes=  1)
        if opt.pretrain:
            pretrain = torch.load("/data/pg-umcg_mii/data/hecktor2022/outpred/pretrained_models/pretrain/resnet_18_23dataset.pth")
            pretrain['state_dict'] = {k.replace("module.", ""):v for k, v in pretrain['state_dict'].items()}
            model.load_state_dict(pretrain['state_dict'], strict=False)
            print ('Load pretrained weights successfully ! ')
        return model
    if opt.model == 'resnet10':
        #return monai.networks.nets.ResNet(spatial_dims=3, block = 'ResNetBlock', n_input_channels = len(opt.input_modality), num_classes= len(opt.breaks) - 1).resnet18()
        model =  monai.networks.nets.resnet10( spatial_dims=3, n_input_channels = input_channels, num_classes=  1)
        if opt.pretrain:
            pretrain = torch.load("/data/pg-umcg_mii/data/hecktor2022/outpred/pretrained_models/pretrain/resnet_10_23dataset.pth")
            pretrain['state_dict'] = {k.replace("module.", ""):v for k, v in pretrain['state_dict'].items()}
            model.load_state_dict(pretrain['state_dict'], strict=False)        
            print ('Load pretrained weights successfully ! ')
        return model
        
    if opt.model == 'resnet18less':
        return resnet18less(spatial_dims=3, n_input_channels = input_channels, num_classes=  1)
    if opt.model == 'resnet10less':
        return resnet10less(spatial_dims=3, n_input_channels = input_channels, num_classes=  1)
    if opt.model == 'resnet10muchless':
        return resnet10muchless(spatial_dims=3, n_input_channels = input_channels, num_classes=  1)        
        
    if opt.model == 'ViT':           
        return monai.networks.nets.ViT(in_channels = input_channels, img_size = (128,128,128),patch_size = (16,16,16),  pos_embed='conv', classification=True, num_classes= 1, spatial_dims=3)
    if opt.model =='FC':
        #return monai.networks.nets.FullyConnectedNet(45,  1, [256,128,64], dropout=0.2)    # 84 input radiomics features
        return monai.networks.nets.FullyConnectedNet(11, 1, [256,128,64], dropout=0.2)    # 84 input radiomics features  4 clincial features
    if opt.model =='FC_clinical':
        #return monai.networks.nets.FullyConnectedNet(4,  1, [16,8,4], dropout=0.2)    # 5 clincial features (CenterID)
        return monai.networks.nets.FullyConnectedNet(4,  1, [16,8,4])    # 5 clincial features (CenterID)
    if opt.model =='FC_clinical4':
        return monai.networks.nets.FullyConnectedNet(4,  1, [256,128,64], dropout=0.2)    # 4 clincial features
    if opt.model =='FC_combine':
        return monai.networks.nets.FullyConnectedNet(49,  1, [256,128,64], dropout=0.2)    # 84 input radiomics features  4 clincial features
        #return monai.networks.nets.FullyConnectedNet(6,  1, [ 1])    # 84 input radiomics features  4 clincial features
        
        
def get_clc_model(opt): 
    
    '''
    if opt.model =='FC':
        return monai.networks.nets.FullyConnectedNet(84, len(opt.breaks) - 1, [4096,1024,256,64], dropout=0.2)    # 84 input radiomics features
    if opt.model =='FC_clinical':
        return monai.networks.nets.FullyConnectedNet(5, len(opt.breaks) - 1, [128,64,32], dropout=0.2)    # 5 clincial features (CenterID)
    if opt.model =='FC_clinical4':
        return monai.networks.nets.FullyConnectedNet(4, len(opt.breaks) - 1, [128,64,32], dropout=0.2)    # 4 clincial features
    if opt.model =='FC_combine':
        return monai.networks.nets.FullyConnectedNet(88, len(opt.breaks) - 1, [4096,1024,256,64], dropout=0.2)    # 84 input radiomics features  4 clincial features        
    '''
    if opt.usehpv == True:
        return monai.networks.nets.FullyConnectedNet(5,  1, [64, 32, 16], dropout=0.2)
    elif opt.useradio == True:
        return monai.networks.nets.FullyConnectedNet(6,  1, [64, 32, 16], dropout=0.2)
    elif opt.useradiorisk == True:
        return monai.networks.nets.FullyConnectedNet(7,  1, [64, 32, 16], dropout=0.2)    
    else:
        return monai.networks.nets.FullyConnectedNet(4,  1, [64, 32, 16], dropout=0.2)    # 84 input radiomics features  4 clincial features        
        
def plot_images(arr_list, nr_images, figsize, cmap_list, colorbar_title_list, filename, vmin_list, vmax_list):
    """
    Plot slices of multiple arrays. Each Numpy on a different row, e.g. CT (row 1), RTDOSE (row 2) and
    segmentation_map (row 3).
    """
    # Make sure that every input array has the same number of slices
    nr_slices = arr_list[0].shape[0]
    for i in range(1, len(arr_list)):
        assert nr_slices == arr_list[i].shape[0]

    # Initialize variables
    if nr_images is None:
        nr_images = nr_slices

    # Make sure that nr_images that we want to plot is greater than or equal to the number of slices available
    if nr_slices < nr_images:
        nr_images = nr_slices
    slice_indices = np.linspace(0, nr_slices - 1, num=nr_images)

    # Only consider unique values
    slice_indices = np.unique(slice_indices.astype(int))

    # Determine number of columns and rows
    num_cols = nr_images
    num_rows = len(arr_list)

    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=tuple(figsize))

    for row, arr in enumerate(arr_list):
        cmap = cmap_list[row]
        colorbar_title = colorbar_title_list[row]

        # Colormap
        # Data range
        vmin = vmin_list[row] if vmin_list[row] is not None else arr.min()
        vmax = vmax_list[row] if vmax_list[row] is not None else arr.max()
        # Label ticks
        # '+1' because end of interval is not included
        # ticks = np.arange(vmin, vmax + 1, ticks_steps_list[row]) if ticks_steps_list[row] is not None else None

        for i, idx in enumerate(slice_indices):
            # Consider the first and last slice
            if i == 0:
                idx = 0
            if i == nr_images - 1:
                idx = nr_slices - 1

            idx = int(idx)
            if num_rows >= 2:
              im = ax[row, i].imshow(arr[idx, ...], cmap=cmap, vmin=vmin, vmax=vmax)
              ax[row, i].axis('off')
            else:
              im = ax[i].imshow(arr[idx, ...], cmap=cmap, vmin=vmin, vmax=vmax)
              ax[i].axis('off')

        plt.tight_layout()

        # Add colorbar
        fig.subplots_adjust(right=0.8)
        max_height = 0.925
        min_height = 1 - max_height
        length = max_height - min_height
        length_per_input = length / num_rows
        epsilon = 0.05
        bottom = max_height - (row + 1) * length_per_input + epsilon / 2
        cbar = fig.add_axes(rect=[0.825, bottom, 0.01, length_per_input - epsilon])
        cbar.set_title(colorbar_title)
        fig.colorbar(im, cax=cbar) # , ticks=ticks)

    plt.savefig(filename)
    plt.close(fig)
    
    
    
