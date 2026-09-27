import numpy as np
import pandas as pd
import sklearn
import sys
import scipy
from scipy.stats import yeojohnson
from log_code import setup_logging
logger=setup_logging('variable_transformation_tech')


def variable_yeo_trimming(X_train,X_test):
    try:
        logger.info(f'before X_train columns:{X_train.columns}')
        logger.info(f'before X_test columns:{X_test.columns}')
        for i in X_train.columns:
            X_train[i+'_yeo'],lam=yeojohnson(X_train[i])
            X_test[i+'_yeo'],lam=yeojohnson(X_test[i])

            X_train=X_train.drop([i],axis=1)
            X_test=X_test.drop([i],axis=1)


            #outliers

            iqr=X_train[i+'_yeo'].quantile(0.75)-X_train[i+'_yeo'].quantile(0.25)
            upper=X_train[i+'_yeo'].quantile(0.75)+(1.5*iqr)
            lower=X_train[i+'_yeo'].quantile(0.25)-(1.5*iqr)

            X_train[i+'_yeo_trimming']=np.where(X_train[i+'_yeo']>upper,upper,np.where(X_train[i+'_yeo']<lower,lower,X_train[i+'_yeo']))
            X_test[i+'_yeo_trimming']=np.where(X_test[i+'_yeo']>upper,upper,np.where(X_test[i+'_yeo']<lower,lower,X_test[i+'_yeo']))

            X_train=X_train.drop([i+'_yeo'],axis=1)
            X_test=X_test.drop([i+'_yeo'],axis=1)

        logger.info(f'after X_train trimm column names:{X_train.columns},{X_train.shape}')
        logger.info(f'after X_test trimm columns names:{X_test.columns},{X_test.shape}')
        return X_train,X_test
    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')












    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')
