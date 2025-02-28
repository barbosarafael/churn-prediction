# 0. Import libraries

from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def load_kaggle_credentials():
    
    # 1. Load environment variables
    
    """
    Carrega as credenciais do Kaggle da variável de ambiente local.
    
    Essa função carrega as credenciais do Kaggle armazenadas em um arquivo .env
    na pasta raiz do projeto. As credenciais s o necess rias para acessar a API do Kaggle.
    """
    
    load_dotenv()
    
    # 2. Define credentials
    
    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")
    

def clean_total_charges(df: pd.DataFrame, col_total_charges = 'TotalCharges') -> pd.DataFrame:
    
    df[col_total_charges] = np.where(df[col_total_charges] == " ", 0, df[col_total_charges])
    df[col_total_charges] = df[col_total_charges].astype(float)
    
    return df

def exclude_customer_id(df: pd.DataFrame, col_customer_id = 'customerID') -> pd.DataFrame:
    
    df = df.drop(col_customer_id, axis = 1)
    
    return df

def calculate_qtd_products(df: pd.DataFrame, service_cols: list = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']) -> pd.DataFrame:
    
    df['QtdProducts'] = (df[service_cols] == 'Yes').sum(axis = 1)
    
    return df

def mapping_churn_col(df: pd.DataFrame, col_churn = 'Churn') -> pd.DataFrame:
    
    df[col_churn] = df[col_churn].map({'Yes': 1, 'No': 0})
    
    return df

def clean_all_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = clean_total_charges(df = df)
    df = exclude_customer_id(df = df)
    df = calculate_qtd_products(df = df)
    df = mapping_churn_col(df = df)
    
    return df    