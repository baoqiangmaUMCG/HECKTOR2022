

# import data
library(foreign)
library(mice)
library(VIM)
library(Hmisc)
library(ResourceSelection)
library(rms)
library(tidyverse)
library(reshape)
library(forcats)
library(RColorBrewer)
library(gridExtra)
library(mfp)
library(rmarkdown)
library(knitr)
library(utils)
library(readxl) 
library(survival)
library(survminer)
library(tidyverse)
library(Hmisc)
library(survcomp)


# #impute weight by gender
DATA_test <- read.csv("H:/xxx/HECKTOR2022/Analysis_again/hecktor2022_TESTSET.csv") 
DATA_test$Gender=ifelse(DATA_test$Gender=='M',0,1)
DATA_test_gender0=DATA_test[DATA_test$Gender==0,]
DATA_test_gender1=DATA_test[DATA_test$Gender==1,]
# train male:  84.50255
# train female: 67.62651
table(is.na(DATA_test_gender1$Weight))# 2 females miss weight
mean(DATA_test_gender1$Weight[!is.na(DATA_test_gender1$Weight)]) #67.39655

# change hpv 0,nan,1-> 0,1,2
DATA_test$HPV.status=ifelse(DATA_test$HPV.status==1,2,ifelse(DATA_test$HPV.status=="NaN",1,0))
DATA_test$HPV.status[is.na(DATA_test$HPV.status)]=1
DATA_test$HPV.status=as.factor(DATA_test$HPV.status)


#  ibm into similar digit as training ibm
DATA_test$original_shape_SurfaceArea=DATA_test$original_shape_SurfaceArea/10000# unit: dm2
DATA_test$CT_original_glrlm_RunLengthNonUniformity=DATA_test$CT_original_glrlm_RunLengthNonUniformity/1000

# compare the data distribution and proportion between training and testing sets
DATA_TRAIN <- read.csv("H:/xxx/HECKTOR2022/Analysis_again/Traindata_2ibms.csv") 
DATA_TRAIN$HPV.status=as.factor(DATA_TRAIN$HPV.status)
DATA_TRAIN$original_shape_SurfaceArea=DATA_TRAIN$original_shape_SurfaceArea*10

t.test(DATA_test$Weight,DATA_TRAIN$Weight)#p-value = 0.5082
t.test(DATA_test$original_shape_SurfaceArea,DATA_TRAIN$original_shape_SurfaceArea)#p-value = 0.06742, mean 0.7412909 0.7945547  ,sd 0.39, 0.43
hist(DATA_test$original_shape_SurfaceArea,xlim = c(0,0.35),ylim = c(0,110))
hist(DATA_TRAIN$original_shape_SurfaceArea,xlim = c(0,0.35),ylim = c(0,110))
t.test(DATA_test$CT_original_glrlm_RunLengthNonUniformity,DATA_TRAIN$CT_original_glrlm_RunLengthNonUniformity)# p-value = 0.02866, mean 3.508059  3.946894 
table(DATA_TRAIN$HPV.status)
table(DATA_test$HPV.status)
chisq.test()

# output average predict
source("H:/xxx/papers/New PET Model/ScienceDirect_files_16May2022_14-41-09.309/1-s2.0-S0167814020301936-mmc2/full.combinations.R")
library(polycor)
completeDATAori=DATA_TRAIN

promIBMnamePET=list()
promIBMnamePET[[2]]=c("Weight","HPV.status","original_shape_SurfaceArea")#,"original_firstorder_10Percentile")#original_firstorder_Maximum,"Risk_BQ",
promIBMnamePET[[3]]=c("Weight","HPV.status","original_shape_SurfaceArea")#,"original_firstorder_10Percentile")
promIBMnamePET[[4]]=c("Weight","HPV.status","original_shape_SurfaceArea")#,"original_firstorder_10Percentile")
promIBMnamePET[[5]]=c("Weight","HPV.status","original_shape_SurfaceArea")#,"original_firstorder_10Percentile")
promIBMnamePET[[6]]=c("Weight","HPV.status","original_shape_SurfaceArea")#,"original_firstorder_10Percentile")

promIBMnamePETCT=list()
promIBMnamePETCT[[2]]=c(promIBMnamePET[[2]],
                        "CT_original_glrlm_RunLengthNonUniformity")
promIBMnamePETCT[[3]]=c(promIBMnamePET[[3]],
                        "CT_original_glrlm_RunLengthNonUniformity")#CT_original_glszm_ZoneEntropy
promIBMnamePETCT[[4]]=c(promIBMnamePET[[4]],
                        "CT_original_glrlm_RunLengthNonUniformity")
promIBMnamePETCT[[5]]=c(promIBMnamePET[[5]],
                        "CT_original_glrlm_RunLengthNonUniformity")
promIBMnamePETCT[[6]]=c(promIBMnamePET[[6]],
                        "CT_original_glrlm_RunLengthNonUniformity")

model.name=list()
model.combper=list()
model.coef.cor=list()
lplist.train=list()
lplist.test=list()
risklist.train=list()
risklist.test=list()


model.name=promIBMnamePETCT
completeDATAori$HPV.status=as.factor(completeDATAori$HPV.status)
#length(table(completeDATAori$BQCV))
for(center in 2:length(table(completeDATAori$BQCV))){
  print(center)
  
  if (center==2){
    completeDATAori$Risk_BQ=completeDATAori$risk_fold1
  }else if(center==3){
    completeDATAori$Risk_BQ=completeDATAori$risk_fold2
  }else if(center==4){
    completeDATAori$Risk_BQ=completeDATAori$risk_fold3
  }else if(center==5){
    completeDATAori$Risk_BQ=completeDATAori$risk_fold4
  }else {
    completeDATAori$Risk_BQ=completeDATAori$risk_fold5
  }
  
  #For final test set
  CVTest0<-completeDATAori[completeDATAori$BQCV%in%names(table(completeDATAori$BQCV))[center],]
  CVTrain<-completeDATAori[!completeDATAori$PatientID%in%CVTest0$PatientID,]
  CVTest<-DATA_test
  #For CV
  # CVTest<-completeDATAori[completeDATAori$BQCV%in%names(table(completeDATAori$BQCV))[center],]
  # CVTrain<-completeDATAori[!completeDATAori$PatientID%in%CVTest$PatientID,]
  
  CV_train_promisingibmonly=CVTrain[,promIBMnamePETCT[[center]]]
  cor_promisingibm=hetcor(CV_train_promisingibmonly)$correlations##cor(CV_train_promisingibmonly,CV_train_promisingibmonly)
  
  # organize the name and the coef when consider the cor
  comb <- full.combinations(abs(cor_promisingibm) >= 0.7)
  #print(comb)
  candidates=colnames(cor_promisingibm)
  predictor.groups <- lapply(1:length(comb), function(x) {candidates[comb[[x]]]})
  #print(predictor.groups, quote = FALSE)
  
  model.coef.cor[[center]]=matrix(NA,10,length(model.name[[center]]))
  
  for (varnum in 1:length(model.name[[center]])){
    temp.submodels=list()
    for (cor.group.num in 1:length(predictor.groups)) {
      predictor.thisgroup=predictor.groups[[cor.group.num]]
      for (num.this.group in 1:length(predictor.thisgroup)) {
        if(num.this.group==1){
          frm=as.formula(paste('Surv(RFS, Relapse)~', predictor.thisgroup[1]))
          selV=predictor.thisgroup[1]
        }else{
          frm <- as.formula(paste('Surv(RFS, Relapse)~',selV," + ",predictor.thisgroup[num.this.group])) 
          selV=paste(selV,"+",predictor.thisgroup[num.this.group])
        }
      }
      cox_model_CVtrain.cor=cph(frm, data = CVTrain);#print(cox_model_CVtrain.cor)
      temp.submodels[[cor.group.num]]=as.list(as.numeric(cox_model_CVtrain.cor$coefficients))
      names(temp.submodels[[cor.group.num]])=predictor.groups[[cor.group.num]]# start find hpv.status, if yes, give it 2 times
      if ("HPV.status" %in% names(temp.submodels[[cor.group.num]])){
        loc.hpv=which(names(temp.submodels[[cor.group.num]])=="HPV.status")
        latername=c("HPV.status=1","HPV.status=2",predictor.groups[[cor.group.num]][(loc.hpv+1):length(predictor.groups[[cor.group.num]])])
        names(temp.submodels[[cor.group.num]])[(loc.hpv):length(names(temp.submodels[[cor.group.num]]))] = latername
      }
      # train_mulvarsum=summary(coxph(frm, data = CVTrain))
      # temp.submodels[[cor.group.num]]["meancindex_subgroups"]=as.numeric(train_mulvarsum$concordance[1])
    }
    temp.submodels.to1 <- do.call(c, temp.submodels)
    temp.submodels.to1=split(unlist(temp.submodels.to1, use.names = FALSE), rep(names(temp.submodels.to1), lengths(temp.submodels.to1)))
    
    sigvar_locs=which(as.numeric(lengths(temp.submodels.to1))==1) # also average the coef of the single appear variables
    for (pos1 in 1:length(sigvar_locs)) {
      temp.submodels.to1[sigvar_locs[pos1]] =as.numeric(temp.submodels.to1[sigvar_locs[pos1]])/length(temp.submodels)
    }
    res.comb.submodel=lapply(temp.submodels.to1,mean)
    model.coef.cor[[center]]=res.comb.submodel
  }
  
  
  # test the model
  
  for (num.this.group in 1:length(model.name[[center]])) {
    if(num.this.group==1){
      frm=as.formula(paste('Surv(RFS, Relapse)~', model.name[[center]][1]))
      selV=model.name[[center]][1]
    }else{
      frm <- as.formula(paste('Surv(RFS, Relapse)~',selV," + ",model.name[[center]][num.this.group])) 
      selV=paste(selV,"+",model.name[[center]][num.this.group])
    }
  }
  
  # cox_model_CVtrain.appranet=cph(frm, data = CVTrain)
  # for (coefnum in 1:length(cox_model_CVtrain.appranet$coefficients)) {
  #   cox_model_CVtrain.appranet$coefficients[coefnum]=  as.numeric(model.coef.cor[[center]][names(cox_model_CVtrain.appranet$coefficients[coefnum])]) 
  # }
  
  cox_model_CVtrain.appranet=coxph(frm, data = CVTrain)
  for (coefnum in 1:length(cox_model_CVtrain.appranet$coefficients)) {
    currentva=names(cox_model_CVtrain.appranet$coefficients[coefnum])
    if(currentva=="HPV.status1"){
      cox_model_CVtrain.appranet$coefficients[coefnum]=  as.numeric(model.coef.cor[[center]]["HPV.status=1"]) 
    }else if(currentva=="HPV.status2"){
      cox_model_CVtrain.appranet$coefficients[coefnum]=  as.numeric(model.coef.cor[[center]]["HPV.status=2"]) 
    }else{
      cox_model_CVtrain.appranet$coefficients[coefnum]=  as.numeric(model.coef.cor[[center]][names(cox_model_CVtrain.appranet$coefficients[coefnum])])}
    
  }
  
  lp_train=as.numeric(predict(cox_model_CVtrain.appranet,type="lp",newdata = CVTrain)) 
  lp_test=as.numeric(predict(cox_model_CVtrain.appranet,type="lp",newdata = CVTest))
  
  # risklist.train[[center]]=matrix(data=NA,nrow =dim(CVTrain)[1] ,ncol = 1)
  risklist.train[[center]]=data.frame(predict(cox_model_CVtrain.appranet,type="risk",newdata = CVTrain))  
  risklist.test[[center]] =data.frame(predict(cox_model_CVtrain.appranet,type="risk",newdata = CVTest)  )
  
  risklist.train[[center]]["PatientsID"]=CVTrain$PatientID
  risklist.test[[center]]["PatientsID"]=CVTest$PatientID
  
  
  lplist.train[[center]]=data.frame(predict(cox_model_CVtrain.appranet,type="lp",newdata = CVTrain))
  lplist.test[[center]] =data.frame(predict(cox_model_CVtrain.appranet,type="lp",newdata = CVTest))   
  lplist.train[[center]]["PatientsID"]=CVTrain$PatientID
  lplist.test[[center]]["PatientsID"]=CVTest$PatientID
  
  
  
  
  CVTrain_S <- Surv(event = CVTrain$Relapse, time = CVTrain$RFS)
  train_sum=summary(coxph(CVTrain_S ~ lp_train, data = CVTrain))
  C_intrain=as.numeric(train_sum$concordance[1])
  
  CVTest_S <- Surv(event = CVTest$Relapse, time = CVTest$RFS)
  test_sum=summary(coxph(CVTest_S ~ lp_test, data = CVTest))
  C_intest=as.numeric(test_sum$concordance[1])
  
  
  
  model.combper[[center]]=c(C_intrain,C_intest)
  names(model.combper[[center]])=c("C_intrain","C_intest")
  
  
  
  
  model.combper[[center]]  =as.data.frame(model.combper[[center]])
  
  model.coef.cor[[center]]=as.data.frame(model.coef.cor[[center]])
  
}


library(xlsx)
outputdir='H://xxx/HECKTOR2022/Analysis_again/'
openxlsx::write.xlsx(model.combper, paste(outputdir,"modelcombper_corcoef.xlsx"))#withweight_
openxlsx::write.xlsx(model.coef.cor, paste(outputdir,"modelcoefcor_corcoef.xlsx"))#withweight_
openxlsx::write.xlsx(risklist.train, paste(outputdir,"risklisttrain_corcoef.xlsx"))#withweight_
openxlsx::write.xlsx(risklist.test, paste(outputdir,"risklisttest_corcoef.xlsx"))#withweight_
openxlsx::write.xlsx(lplist.train, paste(outputdir,"lplisttrain_corcoef.xlsx"))#withweight_
openxlsx::write.xlsx(lplist.test, paste(outputdir,"lplisttest_corcoef.xlsx"))#withweight_





