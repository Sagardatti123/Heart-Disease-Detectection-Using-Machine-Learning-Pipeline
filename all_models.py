import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sys
from log_code import setup_logging
logger=setup_logging('all_models')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import roc_curve


def knn_algorithm(X_train,y_train,X_test,y_test):
    try:
        #find best k value
        # for k in [1, 3, 5, 7, 11, 13]:
        #     obj = KNeighborsClassifier(n_neighbors=k)
        #     obj.fit(X_train, y_train)
        #     pred = obj.predict(X_test)
        #     accurcy = accuracy_score(y_test, pred)
        #     logger.info(f'k:{k},accuracy:{accurcy}')
        global knn_obj
        knn_obj=KNeighborsClassifier(n_neighbors=5)#k:5,accuracy:0.5737704918032787
        knn_obj.fit(X_train,y_train)
        knn_pred=knn_obj.predict(X_test)
        logger.info(f'test accuracy of k nearest :{accuracy_score(y_test,knn_pred)}')
        logger.info(f'test confusion matrix of k nearest:\n{confusion_matrix(y_test,knn_pred)}')
        logger.info(f'test classification report of k nearest:\n{classification_report(y_test,knn_pred)}')

    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')


def navi_algorithm(X_train,y_train,X_test,y_test):
    try:
        global navi_obj
        navi_obj=GaussianNB()
        navi_obj.fit(X_train,y_train)
        navi_pred=navi_obj.predict(X_test)
        logger.info(f'test accuracy of navi bays:{accuracy_score(y_test,navi_pred)}')
        logger.info(f'test confusion matrix:\n{confusion_matrix(y_test,navi_pred)}')
        logger.info(f'test classification report:\n{classification_report(y_test,navi_pred)}')
    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')



def logistic_algorithm(X_train,y_train,X_test,y_test):
    try:
        global logi_obj
        logi_obj=LogisticRegression()
        logi_obj.fit(X_train,y_train)
        logi_pred=logi_obj.predict(X_test)
        logger.info(f'test accuracy of logistic regression:{accuracy_score(y_test,logi_pred)}')
        logger.info(f'test confusion matrix:\n{confusion_matrix(y_test,logi_pred)}')
        logger.info(f'test classification report:\n{classification_report(y_test,logi_pred)}')

    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')


def desicion_algorithm(X_train,y_train,X_test,y_test):
    try:
        global tree_obj
        tree_obj=DecisionTreeClassifier(criterion='entropy')
        tree_obj.fit(X_train,y_train)
        tree_pred=tree_obj.predict(X_test)
        logger.info(f'test accuracy of desicion tree:{accuracy_score(y_test,tree_pred)}')
        logger.info(f'test confusion matric desicion tree:\n:{confusion_matrix(y_test,tree_pred)}')
        logger.info(f'test classification report desiscion tree:\n:{classification_report(y_test,tree_pred)}')

    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')


def random_algorithm(X_train,y_train,X_test,y_test):
    try:
        global random_obj
        random_obj=RandomForestClassifier(criterion='entropy',n_estimators=10)
        random_obj.fit(X_train,y_train)
        random_pred= random_obj.predict(X_test)
        logger.info(f'test accuracy of random forest:{accuracy_score(y_test,random_pred)}')
        logger.info(f'test confusion matric random forest :\n:{confusion_matrix(y_test,random_pred)}')
        logger.info(f'test classification report random forest:\n:{classification_report(y_test,random_pred)}')

    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')


def adaboost_algorithm(X_train,y_train,X_test,y_test):
    try:
        global ab_obj
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        ab_obj = AdaBoostClassifier(estimator=lr, n_estimators=10)
        ab_obj.fit(X_train, y_train)
        ab_pred=ab_obj.predict(X_test)
        logger.info(f'test accuracy of AB:{accuracy_score(y_test,ab_pred )}')
        logger.info(f'test confusion matric AB:\n:{confusion_matrix(y_test,ab_pred)}')
        logger.info(f'test classification report AB:\n:{classification_report(y_test, ab_pred)}')


    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')



def gradient_algorithm(X_train,y_train,X_test,y_test):
    try:
        global  gradient_obj
        gradient_obj = GradientBoostingClassifier(n_estimators=10)
        gradient_obj.fit(X_train, y_train)
        gradient_pred=gradient_obj.predict(X_test)
        logger.info(f'test accuracy of GB:{accuracy_score(y_test,gradient_pred )}')
        logger.info(f'test confusion matric GB:\n:{confusion_matrix(y_test,gradient_pred)}')
        logger.info(f'test classification report GB:\n:{classification_report(y_test, gradient_pred)}')

    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')



def xgb_algorithm(X_train,y_train,X_test,y_test):
    try:
        global xgb_obj
        xgb_obj=XGBClassifier()
        xgb_obj.fit(X_train,y_train)
        xgb_pred=xgb_obj.predict(X_test)
        logger.info(f'test accuracy of XGB:{accuracy_score(y_test, xgb_pred)}')
        logger.info(f'test confusion matric XGB:\n:{confusion_matrix(y_test,xgb_pred)}')
        logger.info(f'test classification report XGB:\n:{classification_report(y_test,xgb_pred)}')

    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')


def AUC_ROC(X_train,y_train,X_test,y_test):
    try:
        knn_pred = knn_obj.predict(X_test)
        navi_pred = navi_obj.predict(X_test)
        logi_pred=logi_obj.predict(X_test)
        tree_pred=tree_obj.predict(X_test)
        random_pred= random_obj.predict(X_test)
        ab_pred=ab_obj.predict(X_test)
        gradient_pred=gradient_obj.predict(X_test)
        xgb_pred=xgb_obj.predict(X_test)

        knn_fpr,knn_tpr,knn_thre=roc_curve(y_test,knn_pred)
        navi_fpr,navi_tpr,navi_thre=roc_curve(y_test,navi_pred)
        logi_fpr,logi_tpr,logi_thre=roc_curve(y_test,logi_pred)
        tree_fpr,tree_tpr,tree_thre=roc_curve(y_test,tree_pred)
        rf_fpr,rf_tpr,rf_thre=roc_curve(y_test,random_pred)
        ab_fpr,ab_tpr,ab_thre=roc_curve(y_test,ab_pred)
        gb_fpr,gb_tpr,gb_thre=roc_curve(y_test,gradient_pred)
        xgb_fpr,xgb_tpr,xgb_thre=roc_curve(y_test,xgb_pred)

        plt.figure(figsize=(5,3))
        plt.title('AUC AND ROC')
        plt.xlabel('flase positive rate')
        plt.ylabel('true positive rate')

        plt.plot(knn_fpr,knn_tpr,label='KNN')
        plt.plot(navi_fpr,navi_tpr,label='navi')
        plt.plot( logi_fpr,logi_tpr,label='logi')
        plt.plot( tree_fpr,tree_tpr,label='tree')
        plt.plot(rf_fpr,rf_tpr,label='rf')
        plt.plot(ab_fpr,ab_tpr,label='ab')
        plt.plot( gb_fpr,gb_tpr,label='gb')
        plt.plot( xgb_fpr,xgb_tpr,label='xgb')
        plt.legend(loc=0)
        plt.show()
        logger.info(f'navi bayes have better AUC and ROC curve ')

    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')






def model(X_train,y_train,X_test,y_test):
    try:
        logger.info('============KNN=================')
        knn_algorithm(X_train,y_train,X_test,y_test)

        logger.info('==============Navie_algorithm=======')
        navi_algorithm(X_train,y_train,X_test,y_test)

        logger.info('===============logistic regression=======')
        logistic_algorithm(X_train,y_train,X_test,y_test)

        logger.info('=============desicion tree================')
        desicion_algorithm(X_train,y_train,X_test,y_test)

        logger.info('===========random forest==================')
        random_algorithm(X_train,y_train,X_test,y_test)

        logger.info('===========adaboost=======================')
        adaboost_algorithm(X_train,y_train,X_test,y_test)

        logger.info('==============gradient boosting============')
        gradient_algorithm(X_train,y_train,X_test,y_test)

        logger.info('===============XGB=========================')
        xgb_algorithm(X_train,y_train,X_test,y_test)

        logger.info('================AUC AND ROC===================')
        AUC_ROC(X_train,y_train,X_test,y_test)



    except Exception as e:
        er_type, er_mess, er_line = sys.exc_info()
        logger.info(f'type of the error is :{er_type} due to the :{er_mess} in the line no :{er_line.tb_lineno}')
