# HECKTOR2022

# Team: RT_UMCG (The department of radiation oncology in University Medical Center Groningen, Netherlands), ranked 4th in Task 2
The script of using radiomics and deep learning for outcome prediction in head and neck cancer patients, which was used for HECKTOR 2022 challenge.

The GTV used in for radiomics or input of deep learning model can be real GTV countour or auto-segmented GTV (we used a SwinTransformer to segment GTV) 

#Data:
Can be download from https://hecktor.grand-challenge.org/data-download-2/ with an approval
Prcocessing:
1. Data_prepare/bounding_box.py
2. Data_prepare/resample.py

#Radiomics workflow

![image](https://user-images.githubusercontent.com/86932526/197784491-d54f134a-bf2e-4f92-9fda-98f9e8daf2d3.png)

#Deep learning workflow

![image](https://user-images.githubusercontent.com/86932526/197784531-e83c68d2-72f6-4089-8b82-5e358474cd2c.png)

Several main.py were provided using different combinations of input.
For example, if use input of CT/PET/GTV(autoseg)/clinical data/radiomics together to train the model, then run:
 
python main_trainusingAutoSeg_withclcradio.py --input_modality CT PT gtv --optimizer sgd --oversample True --model resnet18 --batch_size 12 --input_type tumor --esn 101 --fold 1
