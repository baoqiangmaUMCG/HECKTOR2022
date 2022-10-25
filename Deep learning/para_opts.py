# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:05:46 2022

@author: MaB
"""

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--result_path',
        default='/data/pg-umcg_opc_outpred/Hecktor_result/result_',
        type=str,
        help='Result directory path')
    
    parser.add_argument(
        '--data_path',
        default='/data/pg-umcg_mii/data/hecktor2022/resampled_larger',
        type=str,
        help='Data directory path (after resampled)')
    
    parser.add_argument(
        '--endpoint_path',
        #default='/data/pg-umcg_mii/data/hecktor2022/Patients after split_withWeight_task2.xlsx',
        default='/data/pg-umcg_mii/data/hecktor2022/Task2_clic_endpoint_IBM.xlsx',      
        type=str,
        help='Endpoint information path')
    parser.add_argument(
        '--event_name',
        default= 'Relapse',
        type=str,
        help='Endpoint event name')
    parser.add_argument(
        '--event_time_name',
        default= 'RFS',
        type=str,
        help='Endpoint event time name')
    
    parser.add_argument(
        '--breaks_interval',
        default=182.5, 
        type=float,
        help=
        'Breaks interval time for discrete time-to-event prediction ')
    parser.add_argument(
        '--losses',
        default=['nNet'],
        type=list,
        help=
        'Loss functions for training: (only nnNET loss now))')
    parser.add_argument(
        '--learning_rate',
        default=2e-4,
        type=float,
        help=
        'Initial learning rate ')
    parser.add_argument('--momentum', default=0.99, type=float, help='Momentum')
    
    parser.add_argument(
        '--weight_decay', default=2e-4, type=float, help='Weight Decay')
    
    parser.add_argument(
        '--optimizer',
        default='adam',
        type=str,
        help='Currently only adam, sgd')
    
    parser.add_argument(
        '--batch_size', 
        default=6, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=400,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training, for example: save_100.pth')
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='Pretrained model (.pth)')

    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--no_test',
        action='store_true',
        help='If true, test is not performed.')
    parser.set_defaults(no_test=False)

    parser.add_argument(
        '--checkpoint',
        default=1,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--checkpoint_path',
        default= 'current_model.pth',
        type=str,
        help='Trained model is saved at this name')

    parser.add_argument(
        '--model',
        default='Densenet121',
        type=str,
        help='(DenseNet(121,169,21,264),ResNet, SENet(154),SEResNet(50,101,152) ,SEResNext(50,101), EfficientNet  and all nets from MONAI )')
    parser.add_argument(
        '--model_actfn',
        default='relu',
        type=str,
        help='activation function')
    parser.add_argument(
        '--input_modality',
        nargs='+',
        help='Different types of input modality selection: CT, PT, gtv ')
    parser.add_argument(
        '--fold',
        default=1,
        type=int,
        help='fold number')
    parser.add_argument(
        '--esn',
        default=80,
        type=int,
        help='early stopping patience number')
    '''
    parser.add_argument(
       '--resume',
       action='store_true',
       help='If true, resmue training')
    parser.set_defaults(resume=False)
    '''
    parser.add_argument(
        '--resume_id',
        default='',
        type=str,
        help='The id of wandb runing for resume training')    
    parser.add_argument(
        '--oversample',
        type= bool,
        default= False , 
        help='If true, oversample is performed.')
        
    parser.add_argument(
        '--es_metric',
        default='cindex',
        type=str,
        help='The metric for early stopping in validation set')       
        
    parser.add_argument(
        '--input_type',
        default= 'cat',
        type=str,
        help='The way of combine ct,pet with GTV,  cat: caoncate, tumor: use gtv to extract tumor region from ct, pet')     
        
    parser.add_argument(
        '--pretrain',
        type= bool,
        default= False , 
        help='If true, load pretrained models from Me3D for resnet.')
        
    parser.add_argument(
        '--input_mask',
        default= 'gtv',
        type=str,
        help='The mask for extract tumor region, when input_type is tumor')          
    parser.add_argument(
        '--usehpv',
        default= False,
        type=bool,
        help='If use hpv as clincial data')         
    parser.add_argument(
        '--useradio',
        default= False,
        type=bool,
        help='If use radiomics as input')          
    parser.add_argument(
        '--useradiorisk',
        default= False,
        type=bool,
        help='If use radiomics_risk score as input')            
    
    args = parser.parse_args()



    return args
