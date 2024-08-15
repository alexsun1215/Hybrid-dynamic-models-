#!/usr/bin/env python
# coding: utf-8

# # 读取数据

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib
import pandas as pd
plt.rcParams['figure.dpi'] = 150 # 修改图片分辨率
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'Times New Roman'
import matplotlib.pyplot as plt


# In[2]:


import pandas as pd
data=pd.read_excel('时序数据分析.xlsx',header=2)


# In[3]:


y_list=['postprocedural_stroke','postprocedural_atrial_fibrillation', 'death']
for col in y_list:
    data[col]=data[col].map(lambda x: 1 if pd.isnull(x)==False else 0)


# # 数据处理

# In[4]:


for col in data.columns:
    try:
        data[col]=data[col].astype('float32')
    except:
        pass


# In[5]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['race']=le.fit_transform(data['race'])


# # 缺失值

# In[6]:


tmp=data.isnull().sum()/data.shape[0]
tmp=tmp[tmp>0.9]
tmp


# In[7]:


data.drop(columns=tmp.index.tolist(),axis=1,inplace=True)


# In[8]:


data.admission_age=data.admission_age.map(lambda x: 90 if x=='> 89' else x).astype('float32')


# In[9]:


from sklearnex import patch_sklearn
patch_sklearn() # 这个函数用于开启加速sklearn，出现如下语句就表示OK！


# In[10]:


#KNN均值替换
from sklearn.impute import KNNImputer
imputer = KNNImputer()
X=data.drop(columns=y_list,axis=1)
X=pd.DataFrame(imputer.fit_transform(X),columns=X.columns)


# # 异常值

# In[11]:


data=pd.concat([X,data[y_list]],axis=1)


# In[12]:


data=data[data['BMI']<100].reset_index(drop=True).copy()


# In[13]:


# LOF异常值处理
from sklearn.neighbors import LocalOutlierFactor
detector = LocalOutlierFactor(n_neighbors=10) # 构造异常值识别器
data['LOF']=detector.fit_predict(data)


# In[14]:


data['LOF'].value_counts()


# In[15]:


#-1为异常值,去除异常值
data=data[data['LOF']==1].copy()
data.drop(columns=['LOF'],axis=1,inplace=True)


# In[16]:


data=data.reset_index(drop=True)


# # 数据归一化

# In[17]:


cols=[i for i in data.columns if i not in y_list]


# In[18]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler
ss=MinMaxScaler()
df=pd.DataFrame(ss.fit_transform(data[cols]),columns=cols)
for col in y_list:
    df[col]=data[col]


# In[4]:


name='postprocedural_stroke'


# In[20]:


import joblib
joblib.dump(ss,f'{name}/ss.pkl')


# In[21]:


df.to_excel('处理后数据.xlsx',index=False)


# In[5]:


df=pd.read_excel('处理后数据.xlsx')


# In[6]:


df.head()


# # postprocedural_stroke

# ## 平衡数据处理

# In[7]:


cols=[i for i in df.columns if i not in y_list]
X=df[cols]
y=df['postprocedural_stroke']
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
X_resampled, y_resampled = SMOTEENN(random_state=1).fit_resample(X, y)


# In[8]:


len(cols)


# In[9]:


from collections import Counter
print(sorted(Counter(y_resampled).items()))


# In[10]:


list1=['sex', 'admission_age', 'race', 'BMI', 'atrial_fibrillation', 'valvular_disease', 'stroke', 'sleep_apnea', 'chronic_renal_failure', 'delirium', 'myocardial_infarct', 'congestive_heart_failure', 'peripheral_vascular_disease', 'diabetes', 'hypertension_disease', 'albumin_Pre', 'creatinine_Pre', 'glucose_Pre', 'bun_Pre', 'bnp_Pre', 'd_dimer_Pre', 'alt_Pre', 'ast_Pre', 'ck_mb_Pre', 'ctnt_Pre', 'rbc_Pre', 'platelets_Pre', 'hemoglobin_Pre', 'albumin_Post', 'creatinine_Post', 'glucose_Post', 'bun_Post', 'bnp_Post', 'd_dimer_Post', 'alt_Post', 'ast_Post', 'ck_mb_Post', 'ctnt_Post', 'rbc_Post', 'platelets_Post', 'hemoglobin_Post']
list2=['sbp_desc2', 'sbp_desc1', 'sbp_1', 'sbp_2', 'sbp_3', 'dbp_desc2', 'dbp_desc1', 'dbp_1', 'dbp_2', 'dbp_3', 'mbp_desc2', 'mbp_desc1', 'mbp_1', 'mbp_2', 'mbp_3', 'spo2_desc2', 'spo2_desc1', 'spo2_1', 'spo2_2', 'spo2_3', 'calcium_desc2', 'calcium_desc1', 'calcium_1', 'calcium_2', 'calcium_3', 'chloride_desc2', 'chloride_desc1', 'chloride_1', 'chloride_2', 'chloride_3', 'sodium_desc2', 'sodium_desc1', 'sodium_1', 'sodium_2', 'sodium_3', 'potassium_desc2', 'potassium_desc1', 'potassium_1', 'potassium_2', 'potassium_3', 'lymphocytes_abs_desc2', 'lymphocytes_abs_desc1', 'lymphocytes_abs_1', 'lymphocytes_abs_2', 'lymphocytes_abs_3', 'neutrophils_abs_desc2', 'neutrophils_abs_desc1', 'neutrophils_abs_1', 'neutrophils_abs_2', 'neutrophils_abs_3', 'wbc_desc2', 'wbc_desc1', 'wbc_1', 'wbc_2', 'wbc_3', 'crp_desc2', 'crp_desc1', 'crp_1', 'crp_2', 'crp_3', 'input_desc2', 'input_desc1', 'input_1', 'input_2', 'input_3', 'output_desc2', 'output_desc1', 'output_1', 'output_2', 'output_3']


# ## Lasso筛选特征

# In[16]:


list1=[i for i in list1 if i in X_resampled.columns]


# In[19]:


from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV
alphas=np.logspace(-3,3,100)
model_lassoCV=LassoCV(alphas=alphas,cv=5,random_state=1).fit(X_resampled[list1], y_resampled)
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
MSEs=(model_lassoCV.mse_path_)
MSEs_mean=np.apply_along_axis(np.mean,1,MSEs)
MSEs_std=np.apply_along_axis(np.std,1,MSEs)

fig = plt.gcf()
fig.set_size_inches(12,4)
plt.errorbar(model_lassoCV.alphas_,MSEs_mean
            ,yerr=MSEs_std
            ,fmt="o"
            ,ms=3
            ,mfc="r"
            ,mec="r"
            ,ecolor="lightblue"
            ,elinewidth=2
            ,capsize=4
            ,capthick=1)
plt.semilogx()
plt.axvline(model_lassoCV.alpha_,color="black",ls="--")
plt.xlabel("Lambda")
plt.ylabel("MSE")
ax=plt.gca()
y_major_locator=MultipleLocator(0.05)
ax.yaxis.set_major_locator(y_major_locator)
plt.savefig("./%s/model_lassoCV.jpg"%name,dpi=600,bbox_inches = 'tight')
plt.show()
# print(Lambda)


# In[20]:


coefs=model_lassoCV.path(X_resampled[list1],y_resampled,alphas=alphas)[1].T
fig = plt.gcf()
fig.set_size_inches(15,6)
plt.semilogx(model_lassoCV.alphas_,coefs,"-")
plt.axvline(model_lassoCV.alpha_,color="black",ls="--")
plt.xlabel("Log Lambda")
plt.ylabel("Coefficients")
plt.savefig("./%s/model_lassoCV2.jpg"%name,dpi=600,bbox_inches = 'tight')
plt.show()


# In[21]:


# 获取特征选择结果
selected_features = model_lassoCV.coef_ != 0
lasso_selcet=[col for i,col in enumerate(list1) if selected_features[i]==True]


# In[22]:


lasso_selcet


# ## BILSTM提取动态特征

# In[23]:


from LSTM.lstm_classifier import LSTMClassifier 
# general parameters
nestimators = 100
lstmsize = 1024
lstmdropout = 0.0
lstmoptim = 'rmsprop'
lstmnepochs = 20
lstmbatchsize = 32
a1=X_resampled[[i for i in X_resampled.columns if 'desc2' in i]].values
a2=X_resampled[[i for i in X_resampled.columns if 'desc1' in i]].values
a3=X_resampled[[i for i in X_resampled.columns if '_1' in i]].values
a4=X_resampled[[i for i in X_resampled.columns if '_2' in i]].values
a5=X_resampled[[i for i in X_resampled.columns if '_3' in i]].values
dynamic_all=np.stack((a1,a2,a3,a4,a5), axis=1)
static_all=X_resampled[lasso_selcet].values
labels_all=y_resampled.values


# In[24]:


lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
model_pos, model_neg = lstmcl.train(dynamic_all,labels_all)


# In[25]:


model_pos.save(f'{name}/model_pos.h5')
model_neg.save(f'{name}/model_neg.h5')


# In[26]:


ratios_all_lstm = lstmcl.pos_neg_ratios(model_pos, model_neg, dynamic_all)


# In[27]:


X_resampled['ratios_all_lstm']=ratios_all_lstm


# ## 定义模型评估函数

# In[11]:


from sklearn.metrics import precision_score, recall_score, f1_score ,roc_curve, auc,confusion_matrix ,accuracy_score,roc_auc_score,auc,brier_score_loss
def Optimal_threshold(y_pred,labels_test):
    results=[]
    for i in range(1,1000):
        i=i/1000
        pred_t=y_pred >= i
        results.append([i,metrics.f1_score(labels_test.values.ravel(),pred_t)])
    results=pd.DataFrame(results)
    results.columns=['thresholds','f1-score']
    results['f1-score']=results['f1-score'].astype('float64')
    optimal_threshold = results.loc[results['f1-score'].idxmax()]['thresholds']
    return results,optimal_threshold
def try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2):
    print('Train:')

    precision = precision_score(y_train,y_pred_train1)
    recall = recall_score(y_train,y_pred_train1)
    f1score = f1_score(y_train, y_pred_train1)
    accuracy=accuracy_score(y_train, y_pred_train1)
    cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
    TP=cnf_matrix[1,1]  # 1-->1
    TN=cnf_matrix[0,0]  # 0-->0
    FP=cnf_matrix[0,1]  # 0-->1
    FN=cnf_matrix[1,0]  # 1-->0
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_train2)
    AUC = auc(fpr, tpr)
    print("AUC:      ", '%.4f'%float(AUC),"ACC: ", '%.4f'%float(accuracy),"F1：", '%.4f'%float(f1score),"PPV:", '%.4f'%float(TP/(TP+FP)),\
    "NPV:   ",'%.4f'%float(TN/(TN+FN)),"Sensitivity :   ",'%.4f'%float(TP/(TP+FN)),"Specificity:   ",'%.4f'%float(TN/(FP+TN)))
    print('Model Train Report: \n',metrics.classification_report(y_train,y_pred_train1,digits=4))
    print('*'*50)
    print('Test:')

    precision = precision_score(y_test,y_pred_test1)
    recall = recall_score(y_test,y_pred_test1)
    f1score = f1_score(y_test, y_pred_test1)
    accuracy=accuracy_score(y_test, y_pred_test1)
    cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
    TP=cnf_matrix[1,1]  # 1-->1
    TN=cnf_matrix[0,0]  # 0-->0
    FP=cnf_matrix[0,1]  # 0-->1
    FN=cnf_matrix[1,0]  # 1-->0
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test2)
    AUC = auc(fpr, tpr)
    print("AUC:      ", '%.4f'%float(AUC),"ACC: ", '%.4f'%float(accuracy),"F1：", '%.4f'%float(f1score),"PPV:", '%.4f'%float(TP/(TP+FP)),\
    "NPV:   ",'%.4f'%float(TN/(TN+FN)),"Sensitivity:   ",'%.4f'%float(TP/(TP+FN)),"Specificity:   ",'%.4f'%float(TN/(FP+TN)))
    print('Model Test Report: \n',metrics.classification_report(y_test,y_pred_test1,digits=4))
    print('*'*50)
    print('Valid:')

    precision = precision_score(y_val,y_pred_val1)
    recall = recall_score(y_val,y_pred_val1)
    f1score = f1_score(y_val, y_pred_val1)
    accuracy=accuracy_score(y_val, y_pred_val1)
    cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
    TP=cnf_matrix[1,1]  # 1-->1
    TN=cnf_matrix[0,0]  # 0-->0
    FP=cnf_matrix[0,1]  # 0-->1
    FN=cnf_matrix[1,0]  # 1-->0
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_val2)
    AUC = auc(fpr, tpr)
    print("AUC:      ", '%.4f'%float(AUC),"ACC: ", '%.4f'%float(accuracy),"F1：", '%.4f'%float(f1score),"PPV:", '%.4f'%float(TP/(TP+FP)),\
    "NPV:   ",'%.4f'%float(TN/(TN+FN)),"Sensitivity:   ",'%.4f'%float(TP/(TP+FN)),"Specificity:   ",'%.4f'%float(TN/(FP+TN)))
    print('Model Valid Report: \n',metrics.classification_report(y_val,y_pred_val1,digits=4))
import itertools
def plot_roc(k,y_pred_undersample_score,labels_test,classifiers,color,title):
    fpr, tpr, thresholds = metrics.roc_curve(labels_test.values.ravel(),y_pred_undersample_score)
    roc_auc = metrics.auc(fpr,tpr)
    plt.figure(figsize=(20,16))
    plt.figure(k)
    plt.title(title)
    plt.plot(fpr, tpr, 'b',color=color,label='%s AUC = %0.4f'% (classifiers,roc_auc))
    plt.legend(loc='lower right',fontsize=10)
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
    # plt.figure(figsize=(12,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",color="white" if cm[i, j] > thresh else "black",fontsize = 10,weight = 'heavy')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
from sklearn.model_selection import KFold
# 设置k-fold交叉验证
from sklearn import metrics
def model_cv_score1(model,X,y,random_state=3):
    kfold = KFold(n_splits=5,random_state=random_state,shuffle=True)
    # 对数据进行交叉验证
    test_acc_list=[]
    test_PPV_list=[]
    test_NPV_list=[]
    test_f1_list=[]
    test_auc_list=[]
    test_spe_list=[]
    test_sen_list=[]
    for i,(train_index, test_index) in enumerate(kfold.split(X)):
        # 获取训练集和测试集
        X_train1, X_test1 = X.loc[train_index], X.loc[test_index]
        y_train1, y_test1 = y.loc[train_index], y.loc[test_index]
        model.fit(X_train1, y_train1)                        #打乱标签
        # 在测试集上进行预测
        # y_pred_test1 = model.predict(X_test1)
        # y_pred_train1 = model.predict(X_train1)
        y_pred_test2 = model.predict_proba(X_test1)[:,1]
        results,optimal_threshold=Optimal_threshold(y_pred_test2,y_test1)
        y_pred_test1=[int(i>optimal_threshold) for i in y_pred_test2]
        cnf_matrix=metrics.confusion_matrix(y_test1,y_pred_test1)
        TP=cnf_matrix[1,1]  # 1-->1
        TN=cnf_matrix[0,0]  # 0-->0
        FP=cnf_matrix[0,1]  # 0-->1
        FN=cnf_matrix[1,0]  # 1-->0
        # 输出模型的准确率
        test_acc=metrics.accuracy_score(y_test1,y_pred_test1)
        test_acc_list.append(test_acc)
        # 输出模型的PPV
        test_PPV=TP/(TP+FP)
        test_PPV_list.append(test_PPV)    
        # 输出模型的NPV
        test_NPV=TN/(TN+FN)
        test_NPV_list.append(test_NPV)  
        # 输出模型的F1
        test_f1=metrics.f1_score(y_test1,y_pred_test1)
        test_f1_list.append(test_f1)   
        # 输出模型AUC
        test_auc=metrics.roc_auc_score(y_test1,y_pred_test2)
        test_auc_list.append(test_auc)   
        # 输出模型spe
        test_spe=TN/(FP+TN)
        test_spe_list.append(test_spe)  
        # 输出模型sen
        test_sen=TP/(TP+FN)
        test_sen_list.append(test_sen)  
        print('Fold %s'%(i+1),'*'*50)
        print("test ACC:", round(test_acc,4),
              "test PPV:", round(test_PPV,4),
              "test NPV:", round(test_NPV,4),
              "test F1:", round(test_f1,4),
              "test AUC:", round(test_auc,4),
              "test Specificity:", round(test_spe,4),
              "test Sensitivity:", round(test_sen,4))
        print('\n')
    print('CV Mean','*'*50)
    print('test ACC',round(np.array(test_acc_list).mean(),4),\
            'test PPV',round(np.array(test_PPV_list).mean(),4),\
            'test NPV',round(np.array(test_NPV_list).mean(),4),\
            'test F1',round(np.array(test_f1_list).mean(),4),\
            'test AUC',round(np.array(test_auc_list).mean(),4),\
            'test Specificity',round(np.array(test_spe_list).mean(),4),\
            'test Sensitivity',round(np.array(test_sen_list).mean(),4))
    print('\n')
    return np.array(test_acc_list),np.array(test_PPV_list),np.array(test_NPV_list),np.array(test_f1_list),np.array(test_auc_list),np.array(test_spe_list),np.array(test_sen_list)


# In[29]:


for i in ['stroke']:
    if i in lasso_selcet:
        lasso_selcet.remove(i)


# In[30]:


f_select=lasso_selcet+['ratios_all_lstm']


# In[31]:


f_select


# In[12]:


f_select=['sex',
 'admission_age',
 'race',
 'BMI',
 'atrial_fibrillation',
 'myocardial_infarct',
 'diabetes',
 'hypertension_disease',
 'glucose_Pre',
 'rbc_Pre',
 'platelets_Pre',
 'hemoglobin_Pre',
 'albumin_Post',
 'glucose_Post',
 'bun_Post',
 'ctnt_Post',
 'platelets_Post',
 'ratios_all_lstm']


# In[34]:


#保存数据
import joblib
joblib.dump(X_resampled,f'{name}/X_resampled.pkl')
joblib.dump(y_resampled,f'{name}/y_resampled.pkl')


# In[13]:


import joblib
X_resampled=joblib.load(f'{name}/X_resampled.pkl')
y_resampled=joblib.load(f'{name}/y_resampled.pkl')


# ## 相关性热力图

# In[15]:


tmp=pd.concat([X_resampled,y_resampled],axis=1)[f_select+['postprocedural_stroke']]


# In[16]:


import palettable
plt.figure(figsize=(12, 8),dpi=120)
# tmp_list=pd.DataFrame(tmp.corr()['postprocedural_atrial_fibrillation'].abs()).sort_values(by='postprocedural_atrial_fibrillation',ascending=False).head(15).index.tolist()
sns.heatmap(data=tmp.corr(),
            vmax=0.9, 
            cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
            annot=True,
            fmt=".2f",
            # annot_kws={'size':8,'weight':'normal', 'color':'#253D24'},
            mask=np.triu(np.ones_like(tmp.corr(),dtype=np.bool))#显示对脚线下面部分图
           )
plt.savefig(f"{name}/变量相关性图.jpg",dpi=600, bbox_inches = 'tight')
plt.show()


# In[47]:


from matplotlib import pyplot as plt
tmp.hist(bins=50,figsize=(20,12))
plt.savefig(f"{name}/变量分布图.jpg",dpi=600, bbox_inches = 'tight')
plt.show()


# In[14]:


# 数据分割
from sklearn.model_selection import train_test_split
X_train, _x, y_train, _y = train_test_split(X_resampled[f_select],y_resampled,test_size=0.3,random_state=1)
X_test, X_val, y_test, y_val = train_test_split(_x,_y,test_size=0.333,random_state=1)
print(X_train.shape,X_test.shape,X_val.shape)


# ## RF-BILSTM

# In[15]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(random_state=1,max_depth=1,n_estimators=10)
clf.fit(X_train,y_train)
threshold=0.5
# y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_train1=[int(i>threshold) for i in y_pred_train2]
# y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_test1=[int(i>threshold) for i in y_pred_test2]
# y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
y_pred_val1=[int(i>threshold) for i in y_pred_val2]
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# In[16]:


joblib.dump(clf,'./%s/clf_rf.pkl'%name)


# In[33]:


# 函数用来计算最佳阙值
def calculate_best_threshold(y, y_scores):
    fpr, tpr, thresholds = roc_curve(y, y_scores)
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][0],j_ordered[-1][1] # 返回最佳阙值
def bootstrap_auc(y, pred, classes, bootstraps = 1000, fold_size = 1000):
    statistics_auc = np.zeros((len(classes), bootstraps))
    statistics_acc = np.zeros((len(classes), bootstraps))
    statistics_f1 = np.zeros((len(classes), bootstraps))
    statistics_ppv= np.zeros((len(classes), bootstraps))
    statistics_npv = np.zeros((len(classes), bootstraps))
    statistics_sens = np.zeros((len(classes), bootstraps))
    statistics_spec = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        # df.
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            j_ordered,threshold = calculate_best_threshold(y_sample, pred_sample)          
            statistics_auc[c][i] = metrics.roc_auc_score(y_sample, pred_sample)
            y_pred=[int(i>threshold) for i in pred_sample]
            cnf_matrix=metrics.confusion_matrix(y_sample,y_pred)
            TP=cnf_matrix[1,1]  # 1-->1
            TN=cnf_matrix[0,0]  # 0-->0
            FP=cnf_matrix[0,1]  # 0-->1
            FN=cnf_matrix[1,0]  # 1-->0    
            statistics_acc[c][i] = metrics.accuracy_score(y_sample,y_pred)
            statistics_f1[c][i] = metrics.f1_score(y_sample,y_pred)
            statistics_npv[c][i] = TN/(TN+FN)
            statistics_ppv[c][i] = TP/(TP+FP)   
          
            statistics_sens[c][i]=TP/(TP+FN)
            statistics_spec[c][i]=TN/(FP+TN)       
    return statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec


# In[20]:


statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec = bootstrap_auc(y_train,clf.predict_proba(X_train)[:,1],[0,1])
list1=['AUC','ACC','F1','NPV','PPV','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("RF-BILSTM Train ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')

statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec = bootstrap_auc(y_test,clf.predict_proba(X_test)[:,1],[0,1])
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("RF-BILSTM Test ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')

statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_val,clf.predict_proba(X_val)[:,1],[0,1])
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("RF-BILSTM Valid ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')


# ### 模型评估

# In[21]:


plt.figure(figsize=(15,12), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
plot_confusion_matrix(cnf_matrix,[0,1],title='RF-BILSTM Train',cmap=plt.cm.Blues)
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
plot_confusion_matrix(cnf_matrix,[0,1],title='RF-BILSTM Test',cmap=plt.cm.Blues)
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
plot_confusion_matrix(cnf_matrix,[0,1],title='RF-BILSTM Valid',cmap=plt.cm.Blues)
plt.tight_layout()
plt.savefig('./%s/RF-BILSTM-confusion_matrix.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()


# ### 交叉验证

# In[17]:


clf=joblib.load('./%s/clf_rf.pkl'%name)
test_acc_list_rf,test_PPV_list_rf,test_NPV_list_rf,test_f1_list_rf,test_auc_list_rf,test_spe_list_rf,test_sen_list_rf=\
model_cv_score1(clf,X_resampled[f_select],y_resampled)


# ## Lightgbm-BILSTM

# In[18]:


import lightgbm as lgb
threshold=0.5
clf=lgb.LGBMClassifier(random_state=1,max_depth=1,n_estimators=25,learning_rate=0.1)
clf.fit(X_train,y_train)
# y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_train1=[int(i>threshold) for i in y_pred_train2]
# y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_test1=[int(i>threshold) for i in y_pred_test2]
# y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
y_pred_val1=[int(i>threshold) for i in y_pred_val2]
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# In[29]:


joblib.dump(clf,'./%s/clf_lgb.pkl'%name)


# ### 模型评估

# In[30]:


plt.figure(figsize=(15,12), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
plot_confusion_matrix(cnf_matrix,[0,1],title='LGB-BILSTM Train',cmap=plt.cm.Blues)
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
plot_confusion_matrix(cnf_matrix,[0,1],title='LGB-BILSTM Test',cmap=plt.cm.Blues)
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
plot_confusion_matrix(cnf_matrix,[0,1],title='LGB-BILSTM Valid',cmap=plt.cm.Blues)
plt.tight_layout()
plt.savefig('./%s/LGB-BILSTM-confusion_matrix1.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()


# ### 交叉验证

# In[19]:


clf=joblib.load('./%s/clf_lgb.pkl'%name)
test_acc_list_lgb,test_PPV_list_lgb,test_NPV_list_lgb,test_f1_list_lgb,test_auc_list_lgb,test_spe_list_lgb,test_sen_list_lgb=\
model_cv_score1(clf,X_resampled[f_select],y_resampled)


# In[32]:


clf=joblib.load('./%s/clf_lgb.pkl'%name)
statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec = bootstrap_auc(y_train,clf.predict_proba(X_train)[:,1],[0,1])
list1=['AUC','ACC','F1','NPV','PPV','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("LGB-BILSTM Train ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')

statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec = bootstrap_auc(y_test,clf.predict_proba(X_test)[:,1],[0,1])
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("LGB-BILSTM Test ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')

statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_val,clf.predict_proba(X_val)[:,1],[0,1])
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("LGB-BILSTM Valid ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')


# ## Xgboost-BILSTM

# In[20]:


import xgboost as xgb
threshold=0.5
clf=xgb.XGBClassifier(random_state=1,max_depth=1,n_estimators=30,learning_rate=0.09)
clf.fit(X_train,y_train)
# y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_train1=[int(i>threshold) for i in y_pred_train2]
# y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_test1=[int(i>threshold) for i in y_pred_test2]
# y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
y_pred_val1=[int(i>threshold) for i in y_pred_val2]
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# In[55]:


joblib.dump(clf,'./%s/clf_xgb.pkl'%name)


# ### 模型评估

# In[56]:


plt.figure(figsize=(15,12), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
plot_confusion_matrix(cnf_matrix,[0,1],title='XGB-BILSTM Train',cmap=plt.cm.Blues)
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
plot_confusion_matrix(cnf_matrix,[0,1],title='XGB-BILSTM Test',cmap=plt.cm.Blues)
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
plot_confusion_matrix(cnf_matrix,[0,1],title='XGB-BILSTM Valid',cmap=plt.cm.Blues)
plt.tight_layout()
plt.savefig('./%s/XGB-BILSTM-confusion_matrix1.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()


# ### 交叉验证

# In[21]:


clf=joblib.load('./%s/clf_xgb.pkl'%name)
test_acc_list_xgb,test_PPV_list_xgb,test_NPV_list_xgb,test_f1_list_xgb,test_auc_list_xgb,test_spe_list_xgb,test_sen_list_xgb=\
model_cv_score1(clf,X_resampled[f_select],y_resampled)


# In[58]:


clf=joblib.load('./%s/clf_xgb.pkl'%name)
statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec = bootstrap_auc(y_train,clf.predict_proba(X_train)[:,1],[0,1])
list1=['AUC','ACC','F1','NPV','PPV','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("XGB-BILSTM Train ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')

statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec = bootstrap_auc(y_test,clf.predict_proba(X_test)[:,1],[0,1])
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("XGB-BILSTM Test ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')

statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_val,clf.predict_proba(X_val)[:,1],[0,1])
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("XGB-BILSTM Valid ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')


# ## SVM-BILSTM

# In[25]:


from sklearnex import patch_sklearn
patch_sklearn() # 这个函数用于开启加速sklearn，出现如下语句就表示OK！


# In[28]:


from sklearn.svm import SVC
threshold=0.5
clf=SVC(random_state=1,probability=True,C=1e-3,max_iter=1,class_weight='balanced')
clf.fit(X_train,y_train)
# y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_train1=[int(i>threshold) for i in y_pred_train2]
# y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_test1=[int(i>threshold) for i in y_pred_test2]
# y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
y_pred_val1=[int(i>threshold) for i in y_pred_val2]
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# In[29]:


joblib.dump(clf,'./%s/clf_svm.pkl'%name)


# ### 模型评估

# In[30]:


plt.figure(figsize=(15,12), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
plot_confusion_matrix(cnf_matrix,[0,1],title='SVM-BILSTM Train',cmap=plt.cm.Blues)
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
plot_confusion_matrix(cnf_matrix,[0,1],title='SVM-BILSTM Test',cmap=plt.cm.Blues)
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
plot_confusion_matrix(cnf_matrix,[0,1],title='SVM-BILSTM Valid',cmap=plt.cm.Blues)
plt.tight_layout()
plt.savefig('./%s/SVM-BILSTM-confusion_matrix1.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()


# ### 交叉验证

# In[31]:


clf=joblib.load('./%s/clf_svm.pkl'%name)
test_acc_list_svm,test_PPV_list_svm,test_NPV_list_svm,test_f1_list_svm,test_auc_list_svm,test_spe_list_svm,test_sen_list_svm=\
model_cv_score1(clf,X_resampled[f_select],y_resampled)


# In[34]:


clf=joblib.load('./%s/clf_svm.pkl'%name)
statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec = bootstrap_auc(y_train,clf.predict_proba(X_train)[:,1],[0,1])
list1=['AUC','ACC','F1','NPV','PPV','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("SVM-BILSTM Train ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')

statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec = bootstrap_auc(y_test,clf.predict_proba(X_test)[:,1],[0,1])
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("SVM-BILSTM Test ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')

statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_val,clf.predict_proba(X_val)[:,1],[0,1])
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("SVM-BILSTM Valid ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')


# ## Catboost-BILSTM

# ### PSO调参

# In[35]:


import random
class PSO:
    def __init__(self, parameters):
        """
        particle swarm optimization
        parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
        """
        # initialization
        self.NGEN = parameters[0]    # Algebra of iteration
        self.pop_size = parameters[1]    # population size
        self.var_num = len(parameters[2])     # a variable quantity
        self.bound = []                 # Constraint range of variables
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])
 
        self.pop_x = np.zeros((self.pop_size, self.var_num))    # Position of all particles
        self.pop_v = np.zeros((self.pop_size, self.var_num))    # Velocity of all particles
        self.p_best = np.zeros((self.pop_size, self.var_num))   # The optimal position of each particle
        self.g_best = np.zeros((1, self.var_num))   # Globally optimal location
 
        # Initialize the 0th generation initial global optimal solution
        temp = -1
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.pop_v[i][j] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]      # Store the best individuals
            fit = self.fitness(self.p_best[i])
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit
 
    def fitness(self, ind_var):

        """
        Individual fitness calculation
        """
        depth = round(ind_var[0])
        iterations = round(ind_var[1])
        learning_rate = ind_var[3]
    
        print("depth:",round(depth),"iterations:",round(iterations),'learning_rate:',learning_rate)  
        test_score=get_test_score(depth,iterations,learning_rate)
        print("AUC = ",test_score) # AUC
        return  test_score
 
    def update_operator(self, pop_size):
        """
        Update operator : Updates the position and velocity at the next moment
        """
        c1 = 2     # Learning factor, generally 2
        c2 = 2
        w_start = 0.9;  # initial inertia weight
        w_end = 0.3; # The inertia weight of particle swarm at the maximum number of iterations
        for i in range(pop_size):
            w = w_start - (w_start - w_end) * i / NGEN;  # Update inertia weight
            # renewal speed
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * (
                        self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.pop_x[i])
            # types of regeneration site
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            # Cross-border protection
            for j in range(self.var_num):
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]
            # Update p _ best and g _ best
            if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.pop_x[i]
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]
 
    def main(self):
        popobj = []
        # self.ng_best = np.zeros((1, self.var_num))[0]
        for gen in range(self.NGEN):
            self.update_operator(self.pop_size)
            popobj.append(self.fitness(self.g_best))
            print('############ Generation {} ############'.format(str(gen + 1)))
            # if self.fitness(self.g_best) > self.fitness(self.ng_best):
            self.ng_best = self.g_best.copy()
            print('Best location：{}'.format(self.ng_best))
            print('Maximum function value：{}'.format(self.fitness(self.ng_best)))
        print("---- End of (successful) Searching ----")


# In[36]:


import catboost as cgb
def get_test_score(depth,iterations,learning_rate):
    clf=cgb.CatBoostClassifier(random_seed=1,depth=depth,iterations=iterations,learning_rate=learning_rate,verbose=0)
    clf.fit(X_train,y_train)
    # 评估测试数据R2
    y_pred_test=clf.predict_proba(X_test)[:,1]
    y_pred_val=clf.predict_proba(X_val)[:,1]
    return metrics.roc_auc_score(y_test,y_pred_test)+metrics.roc_auc_score(y_val,y_pred_val)


# In[37]:


NGEN = 2
popsize = 5
low = [1,1,3,0.001]
up = [3,50,5,0.08]
parameters = [NGEN, popsize, low, up]
pso = PSO(parameters)
pso.main()


# ### 最优参数模型

# In[38]:


import catboost as cgb
clf=cgb.CatBoostClassifier(random_seed=1,depth=3,iterations=44,learning_rate=0.08,verbose=0)
clf.fit(X_train,y_train)
threshold=0.5
# y_pred_train1=clf.predict(X_train)
y_pred_train2=clf.predict_proba(X_train)[:,1]
y_pred_train1=[int(i>threshold) for i in y_pred_train2]
# y_pred_test1=clf.predict(X_test)
y_pred_test2=clf.predict_proba(X_test)[:,1]
y_pred_test1=[int(i>threshold) for i in y_pred_test2]
# y_pred_val1=clf.predict(X_val)
y_pred_val2=clf.predict_proba(X_val)[:,1]
y_pred_val1=[int(i>threshold) for i in y_pred_val2]
try_different_method(y_pred_train1,y_pred_train2,y_pred_test1,y_pred_test2,y_pred_val1,y_pred_val2)


# In[39]:


joblib.dump(clf,'./%s/clf_cgb.pkl'%name)


# ### 模型评估

# In[40]:


plt.figure(figsize=(15,12), dpi=120)
plt.subplot(1, 3, 1)
#训练
cnf_matrix=metrics.confusion_matrix(y_train,y_pred_train1)
plot_confusion_matrix(cnf_matrix,[0,1],title='Catboost-BILSTM Train',cmap=plt.cm.Blues)
#测试
plt.subplot(1, 3, 2)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred_test1)
plot_confusion_matrix(cnf_matrix,[0,1],title='Catboost-BILSTM Test',cmap=plt.cm.Blues)
#验证
plt.subplot(1, 3, 3)
cnf_matrix=metrics.confusion_matrix(y_val,y_pred_val1)
plot_confusion_matrix(cnf_matrix,[0,1],title='Catboost-BILSTM Valid',cmap=plt.cm.Blues)
plt.tight_layout()
plt.savefig('./%s/Catboost-BILSTM-confusion_matrix1.jpg'%name,dpi=300,bbox_inches = 'tight')
plt.show()


# ### 交叉验证

# In[41]:


clf=joblib.load('./%s/clf_cgb.pkl'%name)
test_acc_list_cgb,test_PPV_list_cgb,test_NPV_list_cgb,test_f1_list_cgb,test_auc_list_cgb,test_spe_list_cgb,test_sen_list_cgb=\
model_cv_score1(clf,X_resampled[f_select],y_resampled)


# In[42]:


clf=joblib.load('./%s/clf_cgb.pkl'%name)
statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec = bootstrap_auc(y_train,clf.predict_proba(X_train)[:,1],[0,1])
list1=['AUC','ACC','F1','NPV','PPV','Sensitivity','Specificity']
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("Catboost-BILSTM Train ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')

statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec = bootstrap_auc(y_test,clf.predict_proba(X_test)[:,1],[0,1])
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_npv,statistics_ppv,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("Catboost-BILSTM Test ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')

statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec = bootstrap_auc(y_val,clf.predict_proba(X_val)[:,1],[0,1])
list2=[statistics_auc,statistics_acc,statistics_f1,statistics_precision,statistics_recall,statistics_sens,statistics_spec]
for i,j in zip(list1,list2):
    print("Catboost-BILSTM Valid ",i," (95% CI):",round(np.mean(j,axis=1)[1],4),'(',round(np.min(j,axis=1)[1],4),'-',round(np.max(j,axis=1)[1],4),')')


# ## ROC曲线

# In[43]:


color=['red','blue','green','tomato','darkred']
name_list=['RF-BILSTM','LGB-BILSTM','XGB-BILSTM','SVM-BILSTM','Catboost-BILSTM']
clf1=joblib.load('./%s/clf_rf.pkl'%name)
clf2=joblib.load('./%s/clf_lgb.pkl'%name)
clf3=joblib.load('./%s/clf_xgb.pkl'%name)
clf4=joblib.load('./%s/clf_svm.pkl'%name)
clf5=joblib.load('./%s/clf_cgb.pkl'%name)
model_list=[clf1,clf2,clf3,clf4,clf5]
# fig = plt.gcf()
# fig.set_size_inches(15,4)
plt.figure(figsize=(15,5), dpi=120)
plt.subplot(1,3,1)
for i,(name1,model) in enumerate(zip(name_list,model_list)):
    plot_roc(1,model.predict_proba(X_train)[:,1],y_train,name1,color[i],'Train  ROC curve')
plt.subplot(1,3,2)
for i,(name1,model) in enumerate(zip(name_list,model_list)):
    plot_roc(1,model.predict_proba(X_test)[:,1],y_test,name1,color[i],'Test  ROC curve')
plt.subplot(1,3,3)
for i,(name1,model) in enumerate(zip(name_list,model_list)):
    plot_roc(1,model.predict_proba(X_val)[:,1],y_val,name1,color[i],'Valid  ROC curve')
plt.tight_layout()
plt.savefig('./%s/ROC-Curve.jpg'%name,dpi=600,bbox_inches = 'tight')
plt.show()


# ## 模型对比

# In[44]:


columns=['ACC','PPV','NPV','F1','AUC','Specificity','Sensitivity']
tmp1=pd.DataFrame([[test_acc_list_rf.mean(),test_PPV_list_rf.mean(),test_NPV_list_rf.mean(),test_f1_list_rf.mean(),test_auc_list_rf.mean(),test_spe_list_rf.mean(),test_sen_list_rf.mean()],
[test_acc_list_lgb.mean(),test_PPV_list_lgb.mean(),test_NPV_list_lgb.mean(),test_f1_list_lgb.mean(),test_auc_list_lgb.mean(),test_spe_list_lgb.mean(),test_sen_list_lgb.mean()],
[test_acc_list_xgb.mean(),test_PPV_list_xgb.mean(),test_NPV_list_xgb.mean(),test_f1_list_xgb.mean(),test_auc_list_xgb.mean(),test_spe_list_xgb.mean(),test_sen_list_xgb.mean()],
[test_acc_list_svm.mean(),test_PPV_list_svm.mean(),test_NPV_list_svm.mean(),test_f1_list_svm.mean(),test_auc_list_svm.mean(),test_spe_list_svm.mean(),test_sen_list_svm.mean()],
[test_acc_list_cgb.mean(),test_PPV_list_cgb.mean(),test_NPV_list_cgb.mean(),test_f1_list_cgb.mean(),test_auc_list_cgb.mean(),test_spe_list_cgb.mean(),test_sen_list_cgb.mean()],
],columns=columns,index=name_list)
tmp1=tmp1.T
tmp1


# In[45]:


tmp1.to_excel('./%s/Model_comparison.xlsx'%name)


# In[47]:


plt.rcParams['axes.unicode_minus'] = False
# 构造数据
values1 = tmp1['RF-BILSTM'].values
values2 = tmp1['LGB-BILSTM'].values
values3 = tmp1['XGB-BILSTM'].values
values4 = tmp1['SVM-BILSTM'].values
values5 = tmp1['Catboost-BILSTM'].values
N = len(values1)
# 设置雷达图的角度，用于平分切开一个圆面
angles=np.linspace(0, 2*np.pi, N, endpoint=False)
# 为了使雷达图一圈封闭起来，需要下面的步骤
values1=np.concatenate((values1,[values1[0]]))
values2=np.concatenate((values2,[values2[0]]))
values3=np.concatenate((values3,[values3[0]]))
values4=np.concatenate((values4,[values4[0]]))
values5=np.concatenate((values5,[values5[0]]))
angles=np.concatenate((angles,[angles[0]]))
# 绘图
fig=plt.figure(figsize=(12,8),dpi=120)
ax = fig.add_subplot(111, polar=True)
# 绘制折线图
ax.plot(angles, values1, 'o-', linewidth=2, label = 'LR-BILSTM')
ax.fill(angles, values1, alpha=0.25)
# 绘制第二条折线图
ax.plot(angles, values2, 'o-', linewidth=2, label = 'LGB-BILSTM')
ax.fill(angles, values2, alpha=0.25)
# 绘制第三条折线图
ax.plot(angles, values3, 'o-', linewidth=2, label = 'XGB-BILSTM')
ax.fill(angles, values3, alpha=0.25)
# 绘制第四条折线图
ax.plot(angles, values4, 'o-', linewidth=2, label = 'SVM-BILSTM')
ax.fill(angles, values4, alpha=0.25)
# 绘制第五条折线图
ax.plot(angles, values5, 'o-', linewidth=2, label = 'Catboost-BILSTM')
ax.fill(angles, values5, alpha=0.25)
# 添加每个特征的标签
ax.set_thetagrids((angles * 180/np.pi)[:-1], tmp1.index.tolist())
# 设置雷达图的范围
ax.set_ylim(0.89,1)
# 添加标题
plt.title('Comparison of model cross-validation effect')
# 添加网格线
ax.grid(True)
plt.yticks(fontsize=12, color='k')
plt.xticks(fontsize=12, color='k')
# 设置图例
plt.legend(loc = 2)
plt.savefig('./%s/Model_comparison_Radar_map.jpg'%name,dpi=600,bbox_inches = 'tight')
# 显示图形
plt.show()


# ## 模型预测

# In[49]:


ss=joblib.load(f"{name}/ss.pkl")


# In[50]:


test_data={'sex': 1.0,
 'admission_age': 85.71411895751953,
 'race': 34.0,
 'BMI': 20.569628524780274,
 'atrial_fibrillation': 0.0,
 'valvular_disease': 0.0,
 'stroke': 1.0,
 'sleep_apnea': 0.0,
 'chronic_renal_failure': 0.0,
 'delirium': 0.0,
 'myocardial_infarct': 1.0,
 'congestive_heart_failure': 0.0,
 'peripheral_vascular_disease': 0.0,
 'diabetes': 0.0,
 'hypertension_disease': 0.4,
 'albumin_Pre': 2.4000000953674316,
 'creatinine_Pre': 1.7999999523162842,
 'glucose_Pre': 125.0,
 'bun_Pre': 36.0,
 'd_dimer_Pre': 337.4,
 'alt_Pre': 77.0,
 'ast_Pre': 162.0,
 'ck_mb_Pre': 54.0,
 'ctnt_Pre': 2.799999952316284,
 'rbc_Pre': 3.9100000858306885,
 'platelets_Pre': 179.0,
 'hemoglobin_Pre': 8.100000381469727,
 'albumin_Post': 3.799999952316284,
 'creatinine_Post': 1.5,
 'glucose_Post': 115.0,
 'bun_Post': 32.0,
 'alt_Post': 44.0,
 'ast_Post': 53.0,
 'ck_mb_Post': 41.0,
 'ctnt_Post': 0.3100000023841858,
 'rbc_Post': 4.429999828338623,
 'platelets_Post': 220.0,
 'hemoglobin_Post': 9.199999809265137,
 'sbp_desc2': 127.6,
 'sbp_desc1': 118.4,
 'sbp_1': 145.0,
 'sbp_2': 113.0,
 'sbp_3': 112.0,
 'dbp_desc2': 71.8,
 'dbp_desc1': 69.2,
 'dbp_1': 98.0,
 'dbp_2': 59.0,
 'dbp_3': 62.0,
 'mbp_desc2': 90.2,
 'mbp_desc1': 86.2,
 'mbp_1': 106.0,
 'mbp_2': 73.0,
 'mbp_3': 73.0,
 'spo2_desc2': 96.0,
 'spo2_desc1': 96.4,
 'spo2_1': 100.0,
 'spo2_2': 100.0,
 'spo2_3': 100.0,
 'calcium_desc2': 8.600000381469727,
 'calcium_desc1': 8.5,
 'calcium_1': 8.800000190734863,
 'calcium_2': 9.100000381469727,
 'calcium_3': 8.399999618530273,
 'chloride_desc2': 106.0,
 'chloride_desc1': 105.0,
 'chloride_1': 104.0,
 'chloride_2': 106.0,
 'chloride_3': 107.0,
 'sodium_desc2': 142.0,
 'sodium_desc1': 143.0,
 'sodium_1': 141.0,
 'sodium_2': 140.0,
 'sodium_3': 144.0,
 'potassium_desc2': 3.700000047683716,
 'potassium_desc1': 3.799999952316284,
 'potassium_1': 4.300000190734863,
 'potassium_2': 4.599999904632568,
 'potassium_3': 4.199999809265137,
 'lymphocytes_abs_desc2': 0.9575999975204468,
 'lymphocytes_abs_desc1': 1.5,
 'lymphocytes_abs_1': 1.559999942779541,
 'lymphocytes_abs_2': 1.0299999713897705,
 'lymphocytes_abs_3': 1.4299999475479126,
 'neutrophils_abs_desc2': 9.530400276184082,
 'neutrophils_abs_desc1': 4.25,
 'neutrophils_abs_1': 5.289999961853027,
 'neutrophils_abs_2': 7.309999942779541,
 'neutrophils_abs_3': 8.779999732971191,
 'wbc_desc2': 7.099999904632568,
 'wbc_desc1': 8.5,
 'wbc_1': 8.600000381469727,
 'wbc_2': 8.699999809265137,
 'wbc_3': 10.600000381469727,
 'crp_desc2': 7.209999942779541,
 'crp_desc1': 1.100000023841858,
 'crp_1': 50.04800066947937,
 'crp_2': 8.985999965667725,
 'crp_3': 26.500000262260436,
 'input_desc2': 195.85335235595704,
 'input_desc1': 663.0,
 'input_1': 1000.0,
 'input_2': 800.0,
 'input_3': 3.2146389484405518,
 'output_desc2': 129.0,
 'output_desc1': 487.8,
 'output_1': 155.0,
 'output_2': 55.0,
 'output_3': 40.0}
test_data=pd.DataFrame(ss.transform(pd.DataFrame([test_data])[ss.feature_names_in_]),columns=ss.feature_names_in_)
test_data


# In[51]:


#BILSTM提取特征
a1=test_data[[i for i in test_data.columns if 'desc2' in i]].values
a2=test_data[[i for i in test_data.columns if 'desc1' in i]].values
a3=test_data[[i for i in test_data.columns if '_1' in i]].values
a4=test_data[[i for i in test_data.columns if '_2' in i]].values
a5=test_data[[i for i in test_data.columns if '_3' in i]].values
dynamic_test_data=np.stack((a1,a2,a3,a4,a5), axis=1)
from LSTM.lstm_classifier import LSTMClassifier 
from keras.models import load_model
nestimators = 100
lstmsize = 1024
lstmdropout = 0.0
lstmoptim = 'rmsprop'
lstmnepochs = 20
lstmbatchsize = 32
lstmcl = LSTMClassifier(lstmsize, lstmdropout, lstmoptim, lstmnepochs, lstmbatchsize)
model_pos=load_model(f'{name}/model_pos.h5')
model_neg=load_model(f'{name}/model_neg.h5')
test_data['ratios_all_lstm']=lstmcl.pos_neg_ratios(model_pos, model_neg, dynamic_test_data)[0]


# In[52]:


clf1=joblib.load('./%s/clf_rf.pkl'%name)
clf2=joblib.load('./%s/clf_lgb.pkl'%name)
clf3=joblib.load('./%s/clf_xgb.pkl'%name)
clf4=joblib.load('./%s/clf_svm.pkl'%name)
clf5=joblib.load('./%s/clf_cgb.pkl'%name)
for i,(name1,model) in enumerate(zip(name_list,model_list)):
    print(name1,'Prediction class:',model.predict(test_data[f_select])[0],'Prediction probability:',model.predict_proba(test_data[f_select])[:,1][0])


# ## shap解释

# In[53]:


import shap
from shap import LinearExplainer, KernelExplainer, Explanation
shap.initjs()


# In[54]:


import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')


# In[55]:


clf5=joblib.load('./%s/clf_cgb.pkl'%name)
explainer = shap.TreeExplainer(clf5)
shap_values1 = explainer(X_test)
shap_values2 = explainer.shap_values(X_test)


# In[56]:


shap.plots.waterfall(shap_values1[0],show=False,max_display=40)
plt.savefig('./%s/waterfall.jpg'%name,dpi=600, bbox_inches = 'tight')
plt.show()


# In[57]:


shap.force_plot(np.around(explainer.expected_value, decimals=3), np.around(shap_values2[0,:], decimals=3), np.around(X_test.iloc[0,:], decimals=3),matplotlib=True,show=False)
plt.savefig('./%s/force_plot.jpg'%name,dpi=600, bbox_inches = 'tight')
plt.show()


# In[58]:


shap.summary_plot(shap_values2, X_test,plot_size=(12,8),show=False,max_display=40)
plt.savefig('./%s/summary_plot.jpg'%name,dpi=600, bbox_inches = 'tight')
plt.show()


# In[59]:


# shap.force_plot(explainer.expected_value, shap_values2, X_test)


# In[60]:


shap.plots.heatmap(shap_values1[:100],show=False,max_display=40)
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.savefig('./%s/heatmap.jpg'%name,dpi=600, bbox_inches = 'tight')
plt.show()


# In[61]:


shap.plots.bar(shap_values1,max_display=40,show=False)
plt.savefig('./%s/importance.jpg'%name,dpi=600, bbox_inches = 'tight')
plt.show()


# In[62]:


shap.decision_plot(explainer.expected_value, shap_values2[:100], 
                   X_test, feature_order='hclust',show=False)
plt.savefig('./%s/decision_plot.jpg'%name,dpi=600, bbox_inches = 'tight')
plt.show()


# In[63]:


pd.DataFrame({'fea':X_train.columns,'shap':abs(shap_values2).mean(axis=0)}).sort_values(by='shap',ascending=False).head(7).fea.tolist()


# In[64]:


list1=['race',
 'admission_age',
 'ratios_all_lstm',
 'hypertension_disease',
 'bun_Post',
 'albumin_Post',
 'atrial_fibrillation']
for i in range(len(list1)):
    for j in range(len(list1)):
        if i<j:
            shap.dependence_plot(list1[i], shap_values2, X_test, interaction_index=list1[j],show=False)
            fig = plt.gcf()
            fig.set_size_inches(8,4)
            plt.savefig('./%s/dependence_plot_%s_%s.jpg'%(name,list1[i],list1[j]),dpi=600, bbox_inches = 'tight')
            plt.show()

