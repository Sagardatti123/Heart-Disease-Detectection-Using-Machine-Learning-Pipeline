import numpy as np
import pandas as pd
import sklearn
import sys
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from log_code import setup_logging
logger=setup_logging('feature_select')
from sklearn.feature_selection import VarianceThreshold

def select_best_column(X_train,X_test,y_train,y_test):
    try:
        logger.info(f'before constant_tech X_train columns:{X_train.columns}:{X_train.shape}')
        logger.info(f'before constant_tech X_test columns:{X_test.columns}:{X_test.shape}')

        #constant technique

        constant_obj=VarianceThreshold(threshold=0.0)
        constant_obj.fit(X_train)
        logger.info(f'columns you might be remove are :{X_train.columns[~constant_obj.get_support()]}')
        X_train=X_train.drop(['fbs_yeo_trimming'],axis=1)
        X_test=X_test.drop(['fbs_yeo_trimming'],axis=1)

        #quesi constant

        qusi_constant_obj=VarianceThreshold(threshold=0.1)
        qusi_constant_obj.fit(X_train)
        logger.info(f'columns you might br remove are:{X_train.columns[~qusi_constant_obj.get_support()]}')
        X_train=X_train.drop(['trestbps_yeo_trimming', 'chol_yeo_trimming', 'exang_yeo_trimming','ca_yeo_trimming'],axis=1)
        X_test = X_test.drop(['trestbps_yeo_trimming', 'chol_yeo_trimming', 'exang_yeo_trimming', 'ca_yeo_trimming'],axis=1)
        logger.info(f'after qusi_constant_tech X_train columns:{X_train.columns}:{X_train.shape}')
        logger.info(f'after qusi_constant_tech X_test columns:{X_test.columns}:{X_test.shape}')



        #correlation with hypothises testing
        # coffiecient_pvalue=[]
        # for i in X_train.columns:
        #     values=pearsonr(X_train[i],y_train)
        #     coffiecient_pvalue.append(values)
        # coffiecient_pvalue=np.array(coffiecient_pvalue)
        # #logger.info(f'{coffiecient_pvalue}')
        # p_value=coffiecient_pvalue[:,1]
        # plt.figure(figsize=(5,3))
        # plt.title('hypothiese testing')
        # plt.xlabel('column names')
        # plt.ylabel('p values of independent columns')
        # plt.bar(X_train.columns,p_value)
        # plt.show()
        X_train = X_train.drop(['restecg_yeo_trimming'], axis=1)
        X_test = X_test.drop(['restecg_yeo_trimming'], axis=1)
        logger.info(f'after hyp_testing_tech X_train columns:{X_train.columns}:{X_train.shape}')
        logger.info(f'after hyp_testing_tech X_test columns:{X_test.columns}:{X_test.shape}')
        return X_train,X_test

    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')
