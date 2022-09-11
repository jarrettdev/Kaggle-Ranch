# **Kaggle Ranch**

## Overview
- Scrape trending datasets from Kaggle and quickly train models.

## Features

- Organizes datasets with their model performances.

- For every dataset scraped:

    - Preprocessing

        - Detect continuous and categorical variables.

        - Normalize and impute data.

    - For every target variable in the dataset:
        - Compare performance on 27 models + a neural network.
        - Output model performance and processed data in CSV format for every dataset.
        - Save plots + CSVs of XGBoost Feature Importances.
        - Save best performing FastAI model.



- Makes use of FastAI for preprocessing and uses TabNet as a baseline model.

![Dataset Image](https://github.com/jarrettdev/Kaggle-Ranch/blob/main/resources/kaggle_dataset_page.png)

![Performance Picture](https://github.com/jarrettdev/Kaggle-Ranch/blob/main/resources/model_performance.png)

![Directory Image](https://github.com/jarrettdev/Kaggle-Ranch/blob/main/resources/Directory.png)
