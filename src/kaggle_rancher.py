#A program that takes a csv and trains models on it. Streamlined model selection.
#==============================================================================

#LazyPredict
import lazypredict
from lazypredict.Supervised import LazyRegressor
from lazypredict.Supervised import LazyClassifier
#Baysian Optimization
from bayes_opt import BayesianOptimization
#Pandas stack
import pandas as pd
import numpy as np
#FastAI
from fastai.tabular.all import *
from fastai.tabular.core import *
#Plots
import matplotlib.pyplot as plt
#System
import os
import sys
import traceback
#Fit an xgboost model
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
#Random
import random

#TabNet
from fast_tabnet.core import *

#For every _kaggle directory:
#   Open csv as a dataframe
#   Determine Target Variable
#   Run CrackPrediction Pipeline

#Project Variables
#===================================================================================================
project_name = ''

kaggle_directories = os.listdir('../kaggle_directories')
#shuffle directories
random.shuffle(kaggle_directories)

for kaggle_dir in kaggle_directories:
    try:
        if '_kaggle' not in kaggle_dir:
            continue
        kaggle_dir = f'../kaggle_directories/{kaggle_dir}'
        project_name = kaggle_dir.split('../kaggle_directories')[1].split('_kaggle')[0].replace(' ', '_').replace('/', '')
        param_dir = f'../param_files/{project_name}'
        print(f'param_dir: {param_dir}')
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        #move contents of kaggle_dir to param_dir
        for file in os.listdir(kaggle_dir):
            os.rename(f'{kaggle_dir}/{file}', f'{param_dir}/{file}')
        #rename any file in param_dir/file that ends with csv to data.csv
        for file in os.listdir(param_dir):
            if file.endswith('.csv'):
                if 'classification_results' not in file and 'regression_results' not in file:
                    os.rename(f'{param_dir}/{file}', f'{param_dir}/data.csv')
                #os.rename(f'{param_dir}/{file}', f'{param_dir}/data.csv')
        df = pd.read_csv(f'{param_dir}/data.csv')
        target = ''
        #The column closest to the end isPARAM_DIR the target variable that can be represented as a float is the target variable
        for i in range(len(df.columns)-1, 0, -1):
            try:
                df[df.columns[i]] = df[df.columns[i]].astype(float)
                target = df.columns[i]
            except:
                continue
            print(f'Target Variable: {target}')
            #Will be determined by the file name
            PROJECT_NAME = project_name
            PARAM_DIR = param_dir
            TARGET = target
            VARIABLE_FILES = False
            SAMPLE_COUNT = 1000

            FASTAI_LEARNING_RATE = 1e-3
            #Set to True automatically infer if variables are categorical or continuous
            ENABLE_BREAKPOINT = True
            #When trying to declare a column a continuous variable, if it fails, convert it to a categorical variable
            CONVERT_TO_CAT = False
            REGRESSOR = True
            SEP_DOLLAR = True
            SEP_PERCENT = True
            SHUFFLE_DATA = True


            #===================================================================================================

            #Create project config files if they don't exist.
            if not os.path.exists(PARAM_DIR):
                #create param_dir
                os.makedirs(PARAM_DIR)
            if not os.path.exists(f'{PARAM_DIR}/cats.txt'):
                #create param_dir
                with open(f'{PARAM_DIR}/cats.txt', 'w') as f:
                    f.write('')
            if not os.path.exists(f'{PARAM_DIR}/conts.txt'):
                #create param_dir
                with open(f'{PARAM_DIR}/conts.txt', 'w') as f:
                    f.write('')
            if not os.path.exists(f'{PARAM_DIR}/cols_to_delete.txt'):
                with open(f'{PARAM_DIR}/cols_to_delete.txt', 'w') as f:
                    f.write('')

            try:
                df = pd.read_csv(f'{PARAM_DIR}/data.csv', nrows=SAMPLE_COUNT)
            except:
                print('No data.csv file found. Please place a csv named data.csv in the project directory.')
                sys.exit()

            df = df.drop_duplicates()
            if SHUFFLE_DATA:
                df = df.sample(frac=1).reset_index(drop=True)

            # workaround for fastai/pytorch bug where bool is treated as object and thus erroring out.
            for n in df:
                if pd.api.types.is_bool_dtype(df[n]):
                    df[n] = df[n].astype('uint8')
            if SEP_DOLLAR:
                #For every column in df, if the column contains a $, make a new column with the value without the $
                for col in df.columns:
                    if '$' in df[col].to_string():
                        df[col + '_no_dollar'] = df[col].str.replace('$', '').str.replace(',', '')


            if SEP_PERCENT:
                #For every column in df, if the column contains a %, make a new column with the value without the %
                for col in df.columns:
                    if '%' in df[col].to_string():
                        df[col + '_no_percent'] = df[col].str.replace('%', '').str.replace(',', '')

            with open(f'{PARAM_DIR}/cols_to_delete.txt', 'r') as f:
                cols_to_delete = f.read().splitlines()
            for col in cols_to_delete:
                try:
                    del(df[col])
                except:
                    pass
            df = df.fillna(0)
            print(df.isna().sum().sort_values(ascending=False))
            #shrink df as much as possible
            df = df_shrink(df)


            #print types inside of df
            print(df.dtypes)


            #Auto detect categorical and continuous variables
            #==============================================================================
            likely_cat = {}
            for var in df.columns:
                likely_cat[var] = 1.*df[var].nunique()/df[var].count() < 0.05 #or some other threshold

            cats = [var for var in df.columns if likely_cat[var]]
            conts = [var for var in df.columns if not likely_cat[var]]

            #remove target from lists
            try:
                conts.remove(target)
                cats.remove(target)
            except:
                pass
            #Convert target to float
            df[target] = df[target].astype(float)

            print('CATS=====================')
            print(cats)
            print('CONTS=====================')
            print(conts)

            #==============================================================================

            #Populate categorical and continuous lists
            #==============================================================================

            if VARIABLE_FILES == True:
                with open(f'{PARAM_DIR}/cats.txt', 'r') as f:
                    cats = f.read().splitlines()

                with open(f'{PARAM_DIR}/conts.txt', 'r') as f:
                    conts = f.read().splitlines()

            #==============================================================================
            procs = [Categorify, FillMissing, Normalize]
            print(df.describe().T)
            df = df[0:SAMPLE_COUNT]
            splits = RandomSplitter()(range_of(df))

            print((len(cats)) + len(conts))
            #conts = []

            #Convert cont variables to floats
            #==============================================================================

            for var in conts:
                try:
                    df[var] = df[var].astype(float)
                except:
                    print(f'Could not convert {var} to float.')
                    pass

            #==============================================================================

            #Experimental logic to add columns one-by-one to find a breakpoint
            #==============================================================================
            if ENABLE_BREAKPOINT == True:
                temp_procs = [Categorify, FillMissing]
                print('Looping through continuous variables to find breakpoint')
                cont_list = []
                for cont in conts:
                    focus_cont = cont
                    cont_list.append(cont)
                    print(focus_cont)
                    try:
                        to = TabularPandas(df, procs=procs, cat_names=cats, cont_names=cont_list, y_names=target, y_block=RegressionBlock(), splits=splits)
                        del(to)
                    except:
                        print('Error with ', focus_cont)
                        #remove focus_cont from list
                        cont_list.remove(focus_cont)
                        if CONVERT_TO_CAT == True:
                            cats.append(focus_cont)
                        #traceback.print_exc()
                        continue
                #convert all continuous variables to floats
                for var in cont_list:
                    try:
                        df[var] = df[var].astype(float)
                    except:
                        print(f'Could not convert {var} to float.')
                        cont_list.remove(var)
                        if CONVERT_TO_CAT == True:
                            cats.append(var)
                        pass
                print(f'Continuous variables that made the cut : {cont_list}')
                print(f'Categorical variables that made the cut : {cats}')
                #shrink df as much as possible
                df = df_shrink(df)
                print(df.dtypes)

            #==============================================================================
            #Creating tabular object + quick preprocessing
            #==============================================================================
            to = None
            if REGRESSOR == True:
                try:
                    to = TabularPandas(df, procs, cats, conts, target, y_block=RegressionBlock(), splits=splits)
                except:
                    conts = []
                    to = TabularPandas(df, procs, cats, conts, target, y_block=RegressionBlock(), splits=splits)
            else:
                try:
                    to = TabularPandas(df, procs, cats, conts, target, splits=splits)
                except:
                    conts = []
                    to = TabularPandas(df, procs, cats, conts, target, splits=splits)

            #print(dir(to))
            #print(to.xs)
            dls = to.dataloaders()
            print(f'Tabular Object size: {len(to)}')
            try:
                dls.one_batch()
            except:
                print(f'problem with getting one batch of {PROJECT_NAME}')
            #==============================================================================

            #Extracting train and test sets from tabular object
            #==============================================================================

            X_train, y_train = to.train.xs, to.train.ys.values.ravel()
            X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()
            #create dataframe from X_train and y_train
            #export tabular object to csv
            pd.DataFrame(X_train).to_csv(f'{PARAM_DIR}/X_train_{target}.csv', index=False)
            pd.DataFrame(X_test).to_csv(f'{PARAM_DIR}/X_test_{target}.csv', index=False)
            pd.DataFrame(y_train).to_csv(f'{PARAM_DIR}/y_train_{target}.csv', index=False)
            pd.DataFrame(y_test).to_csv(f'{PARAM_DIR}/y_test_{target}.csv', index=False)

            #==============================================================================

            #Ready for model selection!

            if REGRESSOR == True:
                try:
                    reg = LazyRegressor(verbose=2, ignore_warnings=False, custom_metric=None)
                    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
                    print(f'Project: {PROJECT_NAME}')
                    print(PROJECT_NAME)
                    print(f'Target: {target}')
                    print(target)
                    target_std = y_train.std()
                    print(f'Target Standard Deviation: {target_std}')
                    print(models)
                    models['project'] = PROJECT_NAME
                    models['target'] = target
                    models['target_std'] = target_std
                    #rename index of 
                    models.to_csv(f'{PARAM_DIR}/regression_results_{target}.csv', mode='a', header=True, index=True)
                except:
                    print('Issue during lazypredict analysis')
            else:
                #TODO: remove this
                try:
                    clf = LazyClassifier(verbose=2, ignore_warnings=False, custom_metric=None)
                    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
                    print(f'Project: {PROJECT_NAME}')
                    print(PROJECT_NAME)
                    print(f'Target: {target}')
                    print(target)
                    print(f'Target Standard Deviation: {y_train.std()}')
                    print(models)
                    models.to_csv(f'{PARAM_DIR}/classification_results.csv', mode='a', header=False)
                except:
                    print('Issue during lazypredict analysis')

            model_name = 'tabnet'

            # FastAI + pre-trained TabNet
            #==============================================================================
            learn = None
            i = 0
            while True:
                try:
                    del learn
                except:
                    pass
                try:
                    learn = 0
                    model = TabNetModel(get_emb_sz(to), len(to.cont_names), dls.c, n_d=64, n_a=64, n_steps=5, virtual_batch_size=256)
                    # save the best model so far
                    cbs = [SaveModelCallback(monitor='_rmse', comp=np.less, fname=f'{PARAM_DIR}/{model_name}_{PROJECT_NAME}_{TARGET}_best'), EarlyStoppingCallback()]
                    learn = Learner(dls, model, loss_func=MSELossFlat(), metrics=rmse, cbs=cbs)
                    #learn = get_learner(to)
                    if(learn != 0):
                        break
                    if i > 50:
                        break
                except:
                    i += 1
                    print('Error in FastAI TabNet')
                    traceback.print_exc()
                    continue

            try:
                if i < 50:
                    learn.fit_one_cycle(7, FASTAI_LEARNING_RATE)
                    plt.figure(figsize=(10, 10))
                    try:
                        ax = learn.show_results()
                        plt.show(block=True)
                    except:
                        print('Could not show results')
                        pass
            except:
                print('Could not fit model')
                traceback.print_exc()
                pass

            #==============================================================================

            #fit an xgboost model
            #==============================================================================
            try:
                xgb = XGBRegressor()
                xgb.fit(X_train, y_train)

                y_pred = xgb.predict(X_test)
                print('XGBoost RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))
                #save feature importance plot to file
                plot_importance(xgb)
                plt.title(f'XGBoost Feature Importance for {PROJECT_NAME} | Target : {target}', wrap=True)
                plt.tight_layout()
                plt.savefig(f'{PARAM_DIR}/xgb_feature_importance_{target}.png')
                #create a dataframe of feature importance
                xgb_fi = pd.DataFrame(xgb.feature_importances_, index=X_train.columns, columns=['importance'])
                xgb_fi.to_csv(f'{PARAM_DIR}/xgb_feature_importance_{target}.csv')
                #print('XGBoost AUC: ', roc_auc_score(y_test, y_pred))
            except:
                traceback.print_exc()
                print('XGBoost failed')
                continue
    except:
        print(f'error with {project_name}')
        traceback.print_exc()

#==============================================================================

#Bayesian Optimization + FastAI
#==============================================================================
def fit_with(lr:float, wd:float, dp:float, n_layers:float, layer_1:float, layer_2:float, layer_3:float):

    print(lr, wd, dp)
    if round(n_layers) == 2:
        layers = [round(layer_1), round(layer_2)]
    elif int(n_layers) == 3:
        layers = [round(layer_1), round(layer_2), round(layer_3)]
    else:
        layers = [round(layer_1)]
    config = tabular_config(embed_p=float(dp),
                          ps=float(wd))
    learn = tabular_learner(dls, layers=layers, metrics=accuracy, config = config, cbs=CSVLogger(fname=f'{PARAM_DIR}/fastai_log_{target}.csv', append=True))

    with learn.no_bar() and learn.no_logging():
        learn.fit(5, lr=float(lr))

    acc = float(learn.validate()[1])

    return acc

hps = {'lr': (1e-05, 1e-01),
      'wd': (4e-4, 0.4),
      'dp': (0.01, 0.5),
       'n_layers': (1,3),
       'layer_1': (50, 200),
       'layer_2': (100, 1000),
       'layer_3': (200, 2000)}

optim = BayesianOptimization(
    f = fit_with, # our fit function
    pbounds = hps, # our hyper parameters to tune
    verbose = 2, # 1 prints out when a maximum is observed, 0 for silent
    random_state=1
)

#optim.maximize(n_iter=10)
#print(optim.max)
#==============================================================================


#More code to test models (Regression)
#Sourced from https://www.kaggle.com/code/anubhavgoyal10/house-rent-prediction-eda-10-models
#==============================================================================
'''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.special import inv_boxcox
import seaborn as sns

models = {
    'ridge' : Ridge(),
    'xgboost' : XGBRegressor(),
    'catboost' : CatBoostRegressor(verbose=0),
    'lightgbm' : LGBMRegressor(),
    'gradient boosting' : GradientBoostingRegressor(),
    'lasso' : Lasso(),
    'random forest' : RandomForestRegressor(),
    'bayesian ridge' : BayesianRidge(),
    'support vector': SVR(),
    'knn' : KNeighborsRegressor(n_neighbors = 4)
}


for name, model in models.items():
    model.fit(X_train, y_train)
    print(f'{name} trained')



results = {}
kf = KFold(n_splits= 10)

for name, model in models.items():
    result = np.mean(np.sqrt(-cross_val_score(model, X_train, y_train, scoring = 'neg_mean_squared_error', cv= kf)))
    results[name] = result

for name, result in results.items():
    print(f"{name} : {round(result, 3)}")


results_df = pd.DataFrame(results, index=range(0,1)).T.rename(columns={0: 'RMSE'}).sort_values('RMSE', ascending=False)
print(results_df.T)



plt.figure(figsize = (20, 6))
sns.barplot(x= results_df.index, y = results_df['RMSE'], palette = 'summer')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('RMSE of different models')


'''
#==============================================================================
