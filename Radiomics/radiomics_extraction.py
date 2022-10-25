# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:32:28 2022

@author: LiY05
"""
import os
import re
import json
import math
import numpy as np
import pydicom as pdcm
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from radiomics import featureextractor
import pandas as pd



maskpath='xxx/'#
imagingpath='xxx/'#
savepath=maskpath#

mask_list = [f for f in listdir(maskpath) if isfile(join(maskpath, f))]


settings = {}
settings['binWidth'] = 0.25
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

num=0        
dfpt = pd.DataFrame()
for i in range(0,len(mask_list)): #len(mask_list)
    PTID=mask_list[i].replace('_xxx.nii.gz','')
    num=num+1
    imgFile_PT= imagingpath+PTID+'_xxx.nii.gz'
    maskFile =maskpath+mask_list[i]

    try:
        featureVector_PT = extractor.execute(imgFile_PT, maskFile)
        df_newpt = pd.DataFrame.from_dict(featureVector_PT.values()).T
        df_newpt.columns = featureVector_PT.keys()              
    except:
        df_newpt = pd.DataFrame.from_dict([-1])
    df_newpt.insert(0, column = 'Num', value = int(i))
    df_newpt.insert(1, column = 'Index', value = [PTID])
    
    dfpt = pd.concat([dfpt, df_newpt])
    print('PET Done %s' % (int(i)+1))   
dfpt.to_excel(os.path.join(savepath,'PTfeaturesGTVLN_combgtvln_trunctright'+'.xlsx'))

# ct
settings = {}
settings['binWidth'] = 25
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
num=0        
dfct = pd.DataFrame()
for i in range(0,len(mask_list)): #len(mask_list)
    PTID=mask_list[i].replace('_xxx.nii.gz','')
    num=num+1
    imgFile_CT= imagingpath+PTID+'_xxx.nii.gz'
    maskFile =maskpath+mask_list[i]

    try:
        featureVector_CT = extractor.execute(imgFile_CT, maskFile)
        df_newct = pd.DataFrame.from_dict(featureVector_CT.values()).T
        df_newct.columns = featureVector_CT.keys()              
    except:
        df_newct = pd.DataFrame.from_dict([-1])
    df_newct.insert(0, column = 'Num', value = int(i))
    df_newct.insert(1, column = 'Index', value = [PTID])
    
    dfct = pd.concat([dfct, df_newct])
    print('PET Done %s' % (int(i)+1))   
dfct.to_excel(os.path.join(savepath,'CTfeaturesGTVLN_combgtvln_trunctright'+'.xlsx'))

