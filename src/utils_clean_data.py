# 0. Import libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean_total_charges(df: pd.DataFrame, col_total_charges = 'TotalCharges') -> pd.DataFrame:
    
    """
    This function cleans the column 'TotalCharges' by replacing empty strings with 0 and converting the column to float type.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column 'TotalCharges'.
    col_total_charges : str, optional
        The name of the column to be cleaned. The default is 'TotalCharges'.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the cleaned column 'TotalCharges'.
    """
    
    df[col_total_charges] = np.where(df[col_total_charges] == " ", 0, df[col_total_charges])
    df[col_total_charges] = df[col_total_charges].astype(float)
    
    return df

def exclude_customer_id(df: pd.DataFrame, col_customer_id = 'customerID') -> pd.DataFrame:
    
    """
    This function excludes the column 'customerID' from the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column 'customerID'.
    col_customer_id : str, optional
        The name of the column to be excluded. The default is 'customerID'.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the column 'customerID' excluded.
    """
    df = df.drop(col_customer_id, axis = 1)
    
    return df

def calculate_qtd_products(df: pd.DataFrame, service_cols: list = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']) -> pd.DataFrame:
    
    """
    This function calculates the number of products each customer has.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the columns of services.
    service_cols : list, optional
        The list of columns to be considered as services. The default is 
        ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'].

    Returns
    -------
    pd.DataFrame
        The DataFrame with the column 'QtdProducts' added.
    """
    df['QtdProducts'] = (df[service_cols] == 'Yes').sum(axis = 1)
    
    return df

def map_churn_column(df: pd.DataFrame, col_churn = 'Churn') -> pd.DataFrame:
    
    """
    Maps the 'Churn' column from categorical to numerical values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column to be mapped.
    col_churn : str, optional
        The name of the column to be mapped. The default is 'Churn'.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the mapped 'Churn' column.
    """

    df[col_churn] = df[col_churn].map({'Yes': 1, 'No': 0})
    
    return df

def clean_all_data(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Applies all cleaning steps to the given DataFrame.

    This function cleans the 'TotalCharges' column, excludes the 'customerID' column, 
    calculates the number of products each customer has, and maps the 'Churn' column 
    from categorical to numerical values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be cleaned.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    """
    df = clean_total_charges(df = df)
    df = exclude_customer_id(df = df)
    df = calculate_qtd_products(df = df)
    df = map_churn_column(df = df)
    
    return df

def load_and_clean_data(filepath):
    """
    Loads data from a CSV file, cleans it using predefined cleaning steps, and returns the cleaned DataFrame.

    This function reads data from the specified file path into a DataFrame, applies cleaning operations such as 
    cleaning the 'TotalCharges' column, excluding the 'customerID' column, calculating the number of products each 
    customer has, and mapping the 'Churn' column from categorical to numerical values.

    Parameters
    ----------
    filepath : str
        The path to the CSV file containing the data to be loaded and cleaned.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    """

    df = pd.read_csv(filepath)
    
    return clean_all_data(df)