#Program to unzip all the files in the current directory and create directories based on each file name
#%%
import os
import sys
import shutil
import pandas as pd
import traceback
#==============================================================================
#Set download directory to Kaggle Ranch before running
#==============================================================================
# %%
os.listdir('.')
# %%
for file in os.listdir('../kaggle_downloads'):
    print(file)
    if file.endswith('.zip'):
        try:
            shutil.unpack_archive(f'../kaggle_downloads/{file}', '../kaggle_downloads')
        except:
            pass
#%%
for file in os.listdir('../kaggle_downloads'):
    if file.endswith('.csv'):
        #create a directory based on the file name
        filename_string = f'../kaggle_directories/{file[:-4]}_kaggle'
        try:
            os.mkdir(filename_string)
            #move the file to the directory
            print(f'file: {file}')
            print(f'filename_string: {filename_string}')
            shutil.move(f'../kaggle_downloads/{file}', filename_string)
        except:
            pass
# %%