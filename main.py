''''heart disease prediction'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import  warnings

from fontTools.misc import transform
from narwhals.selectors import categorical

warnings.filterwarnings('ignore')
from log_code import setup_logging
logger=setup_logging('main')

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.naive_bayes import  GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from variable_transformation_tech import variable_yeo_trimming
from feature_select import select_best_column
from all_models import  model



class HEART_DISEASE:
        def __init__(self,path):
            try:
                self.path=path

                self.df=pd.read_csv(self.path)
                logger.info(f'no of rows and columns:{self.df.shape}')
                #logger.info(f'calculate null values : \n{self.df.isnull().sum()}')
                #logger.info(f'here there is no values there no need handling missing values tech')
                self.X=self.df.iloc[:,:-1]
                self.y=self.df.iloc[:,-1]
                self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)
                logger.info(f'training dataset shape:{self.X_train.shape}=={self.y_train.shape}')
                logger.info(f'testing dataset shape:{self.X_test.shape}=={self.y_test.shape}')






            except Exception as e:
                er_type,er_mess,er_line=sys.exc_info()
                logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')

                logger.info('===================================')


        def handling_missing_values(self):
            try:
                logger.info(f'calculate null values : \n{self.df.isnull().sum()}')
                logger.info(f'here there is no values there no need handling missing values tech')

            except Exception as e:
                er_type, er_mess, er_line = sys.exc_info()
                logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')
                logger.info('================================')
        def splitting_num_cat(self):
            try:
                self.X_train_numericaol=self.X_train.select_dtypes(exclude='object')
                self.X_train_categricol=self.X_train.select_dtypes(include='object')
                logger.info(f'shape of the X_train numericaol:{self.X_train_numericaol.shape}')
                logger.info(f'shape of the X_train categricol:{self.X_train_categricol.shape}')
                logger.info(f'there is no categricol columns so dont split the data')

            except Exception as e:
                er_type, er_mess, er_line = sys.exc_info()
                logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')

                logger.info('====================================')



        def varibale_trans(self):
            try:
                self.X_train,self.X_test=variable_yeo_trimming(self.X_train,self.X_test)
            except Exception as e:
                er_type, er_mess, er_line = sys.exc_info()
                logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')

                logger.info('========================================')


        def feature_selection(self):
            try:
                self.X_train,self.X_test=select_best_column(self.X_train,self.X_test,self.y_train,self.y_test)

            except Exception as e:
                er_type, er_mess, er_line = sys.exc_info()
                logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')

                logger.info('==================================')




        def data_balancing(self):
            try:
                logger.info(f'before balancing shape of y_train data:{self.y_train.shape}')
                logger.info(f'total rows  of attack {1} :class :{sum(self.y_train==1)}')
                logger.info(f'total rows  of normal {0} :class :{sum(self.y_train==0)}')
                bal_obj=SMOTE(random_state=42)
                self.X_train_balance,self.y_train_balance=bal_obj.fit_resample(self.X_train,self.y_train)
                logger.info(f'after balancing shape of y_train data:{self.y_train_balance.shape}')
                logger.info(f'total rows  of attack {1} :class :{sum(self.y_train_balance == 1)}')
                logger.info(f'total rows  of normal {0} :class :{sum(self.y_train_balance == 0)}')
            except Exception as e:
                er_type, er_mess, er_line = sys.exc_info()
                logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')

                logger.info('=============================')

        def feature_scaling(self):
            try:
                self.sc_obj=StandardScaler()
                self.sc_obj.fit(self.X_train_balance)
                self.X_train_balance_scale=self.sc_obj.transform(self.X_train_balance)
                self.X_test_scale=self.sc_obj.transform(self.X_test)
                with open('standerd_model.pkl','wb') as f:
                    pickle.dump(self.sc_obj,f)

            except Exception as e:
                er_type, er_mess, er_line = sys.exc_info()
                logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')




        def train_all_models(self):
            try:
                #model(self.X_train_balance_scale,self.y_train_balance,self.X_test_scale,self.y_test)
                #var_smoothing=np.float64(1e-10)
                navi_obj = GaussianNB(var_smoothing=np.float64(1e-10))
                navi_obj.fit(self.X_train_balance_scale, self.y_train_balance)
                navi_pred = navi_obj.predict(self.X_test_scale)
                logger.info(f'test accuracy of navi bays:{accuracy_score(self.y_test, navi_pred)}')
                logger.info(f'test confusion matrix:\n{confusion_matrix(self.y_test, navi_pred)}')
                logger.info(f'test classification report:\n{classification_report(self.y_test, navi_pred)}')

                #hyperparameter tuning
                # parameters={ "var_smoothing" : np.logspace(-10,-1,10)}
                # grid_obj=GridSearchCV(estimator=navi_obj,
                #                       param_grid=parameters,
                #                       cv=5,
                #                       scoring='accuracy',
                #                       n_jobs=-1)
                # grid_obj.fit(self.X_train_balance_scale,self.y_train_balance)
                # logger.info(f'parameters are:{grid_obj.best_params_}')
                # logger.info(f'accuracy:{grid_obj.best_score_}')

                #testing
                testing = np.array([[33, 1, 3, 169, 0.7, 0, 2]])
                self.sc_obj.transform(testing)
                logger.info(f'predictions of this model :{navi_obj.predict(testing)[0]}')
                logger.info(f'columns:{self.X_train_balance_scale.shape}')#columns


                #save the model
                with open('navi_bayes_model.pkl','wb') as b:
                    pickle.dump(navi_obj,b)

            except Exception as e:
                er_type, er_mess, er_line = sys.exc_info()
                logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')











            












if __name__=='__main__':
    obj=HEART_DISEASE('heart.csv')
    obj.handling_missing_values()
    obj.splitting_num_cat()
    obj.varibale_trans()
    obj.feature_selection()
    obj.data_balancing()
    obj.feature_scaling()
    obj.train_all_models()

