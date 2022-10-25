# HECKTOR2022

# Team: RT_UMCG (The department of radiation oncology in University Medical Center Groningen, Netherlands), ranked 4th in Task 2
The script of using radiomics and deep learning for outcome prediction in head and neck cancer patients, which was used for HECKTOR 2022 challenge.

The GTV used in for radiomics or input of deep learning model can be real GTV countour or auto-segmented GTV (we used a SwinTransformer to segment GTV) 

#Data:
Can be download from https://hecktor.grand-challenge.org/data-download-2/ with an approval
Prcocessing:
1. Data_prepare/bounding_box.py
2. Data_prepare/resample.py
