
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
source("H:/xxx/data_organize/tiantian/Preselection_function_cox.R")
source('H:/xxx/HECKTOR2022/coxmultianalysis.R')


DATA_ori <- read.csv("H://xxx/HECKTOR2022/Analysis_again/Task2_patients_info_update.csv") 
# DATA_LNori <- read.csv("H://xxx/HECKTOR2022/PtsenterSurvivalAnalysis_LNinfor.csv") 
DATA_GTVLNori <- read.csv("H://xxx/HECKTOR2022/Analysis_again/PtsenterSurvivalAnalysis_GTVLNcombinfor_trunct_hung.csv")#_hung.csv") 
DATA_GTVLNori_CT <- read.csv("H://xxx/HECKTOR2022/Analysis_again/PtsenterSurvivalAnalysis_GTVLNcombinfor_CT_trunct_hung.csv")#_hung.csv") 

DATA_ori$HPV.status=ifelse(DATA_ori$HPV.status==1,2,ifelse(DATA_ori$HPV.status=="NaN",1,0))
DATA_ori$HPV.status[is.na(DATA_ori$HPV.status)]=1
DATA_ori$HPV.status=as.factor(DATA_ori$HPV.status)
cnames=sub("^","CT_",colnames(DATA_GTVLNori_CT)[2:94])
colnames(DATA_GTVLNori_CT)[2:94]=cnames
tempdata=DATA_ori[,c(1:2,6:8,21)]#[,c(1:2,9:10)]
clcdata=DATA_ori[,c(1:12)]
colnames(DATA_GTVLNori_CT)[1]=c("PatientID")
colnames(DATA_GTVLNori)[1]=c("PatientID")
Data_GTVLN_CT=merge(x = tempdata, y = DATA_GTVLNori_CT, by = c("PatientID"), all = TRUE)
Data_GTVLN_PETCT=merge(x = Data_GTVLN_CT, y = DATA_GTVLNori, by = c("PatientID"), all = TRUE)
completeDATAori<-Data_GTVLN_PETCT[complete.cases(Data_GTVLN_PETCT),]

completeDATAori$original_firstorder_Energy<-completeDATAori$original_firstorder_Energy/100000
completeDATAori$original_firstorder_TotalEnergy<-completeDATAori$original_firstorder_TotalEnergy/100000
completeDATAori$original_shape_MeshVolume<-completeDATAori$original_shape_MeshVolume/100000
completeDATAori$original_shape_SurfaceArea<-completeDATAori$original_shape_SurfaceArea/100000


for (qqq in 7:99){#length(completeDATAori)
  digit=floor(log10(abs(mean(completeDATAori[,qqq])/10))) + 1
  completeDATAori[,qqq]=completeDATAori[,qqq]/(10^(digit))
}


completeDATAori_BU=completeDATAori

# hector IBM
ctfeaturesdata=completeDATAori[,c(7:99)]#[,5:97]
petfeaturesdata=completeDATAori[,100:132]
petctfeaturesdata=completeDATAori[,7:132]


#
covariates <- colnames(ctfeaturesdata)[!colnames(ctfeaturesdata)%in%c("Relapse","RFS",
                                                                      "PatientID","PETCTScanner","CV_center","CenterID","CVgroup","BQCV",
                                                                      "original_firstorder_TotalEnergy",
                                                                      "CT_original_firstorder_TotalEnergy"
                                                                      # ,"GTVLNcomb_CT_original_firstorder_Maximum",
                                                                      # "GTVLNcomb_CT_original_firstorder_Minimum",
                                                                      # "GTVLNcomb_original_firstorder_Maximum",
                                                                      # "GTVLNcomb_original_firstorder_Minimum"
)]

univ_formulas <- sapply(covariates,
                        function(x) as.formula(paste('Surv(RFS, Relapse)~', x)))
univ_models <- lapply( univ_formulas, function(x){coxph(x, data = completeDATAori)})
univ_results <- lapply(univ_models,
                       function(x){ 
                         BIC=signif(BIC(x), digits=5)
                         x <- summary(x)
                         p.value<-signif(x$wald["pvalue"], digits=2)
                         wald.test<-signif(x$wald["test"], digits=2)
                         LLRT<-signif(x$logtest["test"], digits=4)
                         beta<-signif(x$coef[1], digits=2);#coeficient beta
                         HR <-signif(x$coef[2], digits=4);#exp(beta)
                         HR.confint.lower <- signif(x$conf.int[,"lower .95"], 4)
                         HR.confint.upper <- signif(x$conf.int[,"upper .95"],4)
                         HR <- paste0(HR, " (", 
                                      HR.confint.lower, "-", HR.confint.upper, ")")
                         res<-c(beta, HR, wald.test, p.value,BIC,LLRT)
                         names(res)<-c("beta", "HR (95% CI for HR)", "wald.test", 
                                       "p.value","BIC","LLRT")
                         return(res)
                         #return(exp(cbind(coef(x),confint(x))))
                       })
res <- t(as.data.frame(univ_results, check.names = FALSE))
univ_results_df=as.data.frame(res)
univ_results_df <- univ_results_df[order(univ_results_df$BIC),]
print(univ_results_df)

varpless0.05=univ_results_df[as.numeric(univ_results_df$p.value)<0.05,]
DATA_onlywithvarpless0.05=completeDATAori[,rownames(varpless0.05)]

cormatrix=cor(DATA_onlywithvarpless0.05)



imp.corr=cormatrix
bb=imp.corr#dim(imp.corr)[1]-1
for (i in 1:dim(imp.corr)[1]){
  aa=try(colnames(bb)%in%names(bb[i,][bb[i,]>0.8]),silent = TRUE)
  if (!"try-error"%in% class(aa)){
    temp=bb[aa,aa]
    temp=try(temp[1,][temp[1,]<0.999999],silent = TRUE)
    if (!"NA"%in% as.numeric(temp)){
      go=try(bb[!colnames(bb)%in%names(temp),])
      if("try-error"%in% class(go)){next; }else{
        bb=go;
        go=try(bb[,!colnames(bb)%in%names(temp)])};
      if("try-error"%in% class(go)){next; }else{
        bb=go}
    }  }
};bb

colnames(bb)


completeDATAori_preselect=completeDATAori[,c(colnames(bb),"PatientID","BQCV","Relapse","RFS")]
openxlsx::write.xlsx(completeDATAori_preselect, paste(outputdir,"Allvarless005lowcor_forBQ_GTVPETCT_updateinfo_hunggt4.xlsx"))

set.seed(1)
cv_re=list()
model.list=list()
# forcevar=c("Weight")
for(center in 2:length(table(completeDATAori_preselect$BQCV))){
  
  # CVTest<-completeDATAori_preselect[completeDATAori_preselect$CV_center%in%names(table(completeDATAori_preselect$CV_center))[center],]
  # CVTrain<-completeDATAori_preselect[!completeDATAori_preselect$PatientID%in%CVTest$PatientID,]
  
  CVTest<-completeDATAori_preselect[completeDATAori_preselect$BQCV%in%names(table(completeDATAori_preselect$BQCV))[center],]
  CVTrain<-completeDATAori_preselect[!completeDATAori_preselect$PatientID%in%CVTest$PatientID,]
  
  DATA_clcPTbas=CVTrain[,!colnames(CVTrain)%in%c("PatientID","BQCV","CVgroup","CenterID","PETCTScanner")]#,"LN_original_firstorder_Energy",
  # "GTV_original_firstorder_TotalEnergy"
  V_1stselect=list();aaa=1;
  V_2ndselect=list();bbb=1;
  V_3rdselect=list();ccc=1;
  V_4thselect=list();ddd=1;
  V_5thselect=list();eee=1;
  V_select=list();fff=1;
  
  for  (boot_n in 1:1000) {
    DATA_clcPTbas$id<-c(1:dim(DATA_clcPTbas)[1])
    DATA_train<-DATA_clcPTbas[sample(DATA_clcPTbas$id,size = dim(DATA_clcPTbas)[1],replace = TRUE),]
    DATA_train<-DATA_train[,!colnames(DATA_train)%in%c("id"),]
    print(paste0("---------------------------- ")) # 
    
    print(paste0("Except Center num: ",center,"Test number:",dim(CVTest)[1],"Train number:",dim(DATA_train)[1]," & BOOT TIMES:",boot_n))
    # 
    covariates <- colnames(DATA_train)[!colnames(DATA_train)%in%c("Relapse","RFS")]
    univ_formulas <- sapply(covariates,
                            function(x) as.formula(paste('Surv(RFS, Relapse)~', x)))
    univ_models <- lapply( univ_formulas, function(x){coxph(x, data = DATA_train)})
    univ_results <- lapply(univ_models,
                           function(x){ 
                             BIC=signif(BIC(x), digits=5)
                             x <- summary(x)
                             p.value<-signif(x$wald["pvalue"], digits=2)
                             wald.test<-signif(x$wald["test"], digits=2)
                             LLRT<-signif(x$logtest["test"], digits=4)
                             beta<-signif(x$coef[1], digits=2);#coeficient beta
                             HR <-signif(x$coef[2], digits=4);#exp(beta)
                             HR.confint.lower <- signif(x$conf.int[,"lower .95"], 4)
                             HR.confint.upper <- signif(x$conf.int[,"upper .95"],4)
                             HR <- paste0(HR, " (", 
                                          HR.confint.lower, "-", HR.confint.upper, ")")
                             res<-c(beta, HR, wald.test, p.value,BIC,LLRT)
                             names(res)<-c("beta", "HR (95% CI for HR)", "wald.test", 
                                           "p.value","BIC","LLRT")
                             return(res)
                             #return(exp(cbind(coef(x),confint(x))))
                           })
    res <- t(as.data.frame(univ_results, check.names = FALSE))
    univ_results_df=as.data.frame(res)
    univ_results_df <- univ_results_df[order(univ_results_df$BIC),]
    
    
    
    # multi forward BIC based
    c_b=-1  
    ii=1
    while(c_b<0){
      # no force variables
      if(ii==1){frm=as.formula(paste('Surv(RFS, Relapse)~', rownames(univ_results_df)[1]));#no force variables
      
      
      M1<-cph(frm, data = DATA_train); selV=rownames(univ_results_df)[1];BIC_M<-BIC(M1);
      
      if(anova(M1)["TOTAL","P"]>0.05){
        c_b=1;break
      }else{
        print(paste0('Loop:',ii,' ','BIC: ',round(BIC_M,4),"-->  ",selV ))
        univ_results_df<-univ_results_df[!rownames(univ_results_df)%in%rownames(univ_results_df)[1],]
        V_1stselect[aaa]=selV;aaa=aaa+1;
        V_select[fff]=selV;fff=fff+1;
      }
      }
      
      # 
      # # force variables, something wrong when combine  frequency list
      # if(ii==1){frm=as.formula(paste('Surv(RFS, Relapse)~', forcevar));#force variables
      # 
      # 
      # M1<-cph(frm, data = DATA_train); selV=forcevar;BIC_M<-BIC(M1);
      # 
      # if(anova(M1)["TOTAL","P"]>0.05){
      #   # c_b=1;break
      #   print(paste0('Loop:',ii,' ','BIC: ',round(BIC_M,4),"-->  ",selV ))
      #   univ_results_df<-univ_results_df[!rownames(univ_results_df)%in%forcevar,]
      #   V_1stselect[aaa]=selV;aaa=aaa+1;
      #   V_select[fff]=selV;fff=fff+1;
      #   
      # }else{
      #   print(paste0('Loop:',ii,' ','BIC: ',round(BIC_M,4),"-->  ",selV ))
      #   univ_results_df<-univ_results_df[!rownames(univ_results_df)%in%forcevar,]
      #   V_1stselect[aaa]=selV;aaa=aaa+1;
      #   V_select[fff]=selV;fff=fff+1;
      # }
      # }
      
      
      
      # selV need to combine with each of the rest, and sort the BIC of each combined result 
      cox_predict=list(type = any)
      nameVariables=matrix(NA,nrow = 1,  ncol = length(rownames(univ_results_df)))
      AIC_BIC_allreset   =matrix(NA,nrow = 2,  ncol = length(rownames(univ_results_df)))
      p_LRTallreset   =matrix(NA,nrow = 1,  ncol = length(rownames(univ_results_df)))
      BaseValue=matrix(NA,nrow = length(rownames(univ_results_df)),  ncol = 4)
      
      
      if (length(rownames(univ_results_df))>0){
        ii=ii+1
        jj=0
        for (kk in 1:length(rownames(univ_results_df))) {
          jj=jj+1
          frm <- as.formula(paste('Surv(RFS, Relapse)~',selV," + ",rownames(univ_results_df)[kk])) 
          cox_predict[[jj]] = try(
            (cph(frm, data = DATA_train)))
          
          if(inherits(cox_predict[[jj]], c("try-error"))|cox_predict[[jj]]$fail){
            AIC_BIC_allreset[jj]=NA;
          } else {
            ###-- basic performance measures
            AIC_BIC=c(AIC( cox_predict[[jj]]),BIC( cox_predict[[jj]]))#AIC_BIC_allreset[,jj]=
            LRT_p=lrtest(M1,cox_predict[[jj]])$stats[3]#p_LRTallreset[,jj]=
            res<-c(rownames(univ_results_df)[kk],AIC_BIC, LRT_p)
            BaseValue[jj,]<-res
          }
        }
      }else{
        c_b=1;break;
      }
      
      
      colnames(BaseValue)  = c("features","AIC","BIC","P_LRT")
      BaseValue=as.data.frame(BaseValue)
      BaseValue <- BaseValue[order(BaseValue$BIC),]
      BaseValue
      
      if ((BaseValue[1,]["P_LRT"]<0.05) & (BaseValue[1,]["BIC"]<BIC_M) ){
        print(paste0('Loop:',ii,' ','BIC: ',round(as.numeric(BaseValue[1,]["BIC"]) ,4),"-->  ",selV," + ",BaseValue[1,]["features"]))
        selV=paste0(selV," + ",BaseValue[1,]["features"])
        univ_results_df<-univ_results_df[!rownames(univ_results_df)%in%BaseValue[1,]["features"]$features,]
        BIC_M=BaseValue[1,]["BIC"]
        
        if(ii==2){
          V_2ndselect[bbb]=BaseValue[1,]["features"]$features;bbb=bbb+1}
        if(ii==3){
          V_3rdselect[ccc]=BaseValue[1,]["features"]$features;ccc=ccc+1}    
        if(ii==4){
          V_4thselect[ddd]=BaseValue[1,]["features"]$features;ddd=ddd+1}    
        if(ii==5){
          V_5thselect[eee]=BaseValue[1,]["features"]$features;eee=eee+1}
        
        V_select[fff]=BaseValue[1,]["features"]$features;fff=fff+1;
        
      }else{c_b=1}
      
      
      
      
    }
    
  }
  
  
  V1res=as.data.frame(sort(table(unlist(V_1stselect)),decreasing = TRUE))
  V2res=as.data.frame(sort(table(unlist(V_2ndselect)),decreasing = TRUE))
  V3res=as.data.frame(sort(table(unlist(V_3rdselect)),decreasing = TRUE))
  V4res=as.data.frame(sort(table(unlist(V_4thselect)),decreasing = TRUE))
  # V5res=as.data.frame(sort(table(unlist(V_5thselect)),decreasing = TRUE))
  Vallres=as.data.frame(sort(table(unlist(V_select)),decreasing = TRUE))
  
  # # no force varibales 
  #put all data frames into list
  df_list <- list(V1res, V2res,V3res,Vallres)#,V3res,V4res,Vallres)  #,V5res)#
  #merge all data frames together
  cv_re[[center]]<-df_list %>% reduce(full_join, by='Var1')
  
  # force variables
  # #put all data frames into list
  # df_list <- list(V2res, V3res,Vallres)#,V3res,V4res,Vallres)  #,V5res)#
  # #merge all data frames together
  # cv_re[[center]]<-df_list %>% reduce(full_join, by='Var1')
  # 
  
  
  # develop the model
  proVarinthistrain=V1res[V1res$Freq>100,]# proVarinthistrain=V2res[V2res$Freq>100,]#
  #note put all promising featurs in 1 model formula
  # if (dim(proVarinthistrain)[1]==1){
  #   formula_train=as.character(proVarinthistrain[1,]$Var1) 
  # }else{
  #   for (vinm in 1:dim(proVarinthistrain)[1]){
  #     if (vinm==1){
  #       formula_train=as.character(proVarinthistrain[1,]$Var1)
  #       }else{
  #       formula_train=paste0(formula_train," + ",as.character(proVarinthistrain[vinm,]$Var1))
  #     }
  #   }
  # }
  #combine promising features in CITOR way
  # model.list=list()
  model.coef=matrix(NA,dim(proVarinthistrain)[1],1)
  model.name=matrix(NA,dim(proVarinthistrain)[1],1)
  model.1per=matrix(NA,dim(proVarinthistrain)[1],2)
  for (vinm in 1:dim(proVarinthistrain)[1]){
    formula_train=as.character(proVarinthistrain[vinm,]$Var1)
    formula_train <- as.formula(paste('Surv(RFS, Relapse)~',as.character(proVarinthistrain[vinm,]$Var1))) 
    cox_model_CVtrain=cph(formula_train, data = CVTrain)
    # model.list[[vinm]]=cox_model_CVtrain
    model.coef[vinm]=as.numeric(cox_model_CVtrain$coefficients)
    model.name[vinm]=as.character(proVarinthistrain[vinm,]$Var1)
    
    train_1varsum=summary(coxph(formula_train, data = CVTrain))
    model.1per[vinm,1]=as.numeric(train_1varsum$concordance[1])
    
    PI_test <- CVTest[,model.name[vinm]] *model.coef[vinm]
    S_test <- Surv(event = CVTest$Relapse, time = CVTest$RFS)
    c_test_1var=as.numeric(summary(coxph(S_test ~ PI_test, data = CVTest))$concordance[1])  
    model.1per[vinm,2]=as.numeric(c_test_1var)
    
    
  }  
  
  # test the model
  for (vinm in 1:dim(proVarinthistrain)[1]){
    if (vinm==1){
      lp_train=CVTrain[,model.name[vinm]]*model.coef[vinm]/dim(proVarinthistrain)[1]
      lp_test=CVTest[,model.name[vinm]]*model.coef[vinm]/dim(proVarinthistrain)[1]
    }else{
      lp_train=lp_train+CVTrain[,model.name[vinm]]*model.coef[vinm]/dim(proVarinthistrain)[1]
      lp_test=lp_test+CVTest[,model.name[vinm]]*model.coef[vinm]/dim(proVarinthistrain)[1]
    }
  }
  
  CVTrain_S <- Surv(event = CVTrain$Relapse, time = CVTrain$RFS)
  train_sum=summary(coxph(CVTrain_S ~ lp_train, data = CVTrain))
  C_intrain=as.numeric(train_sum$concordance[1])
  
  CVTest_S <- Surv(event = CVTest$Relapse, time = CVTest$RFS)
  test_sum=summary(coxph(CVTest_S ~ lp_test, data = CVTest))
  C_intest=as.numeric(test_sum$concordance[1])  
  
  
  model.list[[center]]=c(C_intrain,C_intest,as.data.frame(model.name),as.data.frame(model.coef),as.data.frame(model.1per))
  # names(model.list[[center]]) <- c("C_intrain","C_intest",c(rep('model.name',dim(proVarinthistrain)[1])),c(rep('model.coef',dim(proVarinthistrain)[1])))
  names(model.list[[center]]) <- c("C_intrain_comb","C_intest_comb",'model.name','model.coef',"model.1varperintrain","model.1varperintest")
  model.list[[center]]=as.data.frame(model.list[[center]])
  
}
# model.list=as.data.frame(model.list[[center]])
cv_re

library(xlsx)
outputdir='H://xxx/HECKTOR2022/Analysis_again/'
openxlsx::write.xlsx(cv_re, paste(outputdir,"CenterBQNew_5CVTrainresult_Trunct_GTVPETCT(3Frelist)_updateinfo_hung4.xlsx"))




























































