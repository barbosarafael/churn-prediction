from utils import load_kaggle_credentials
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_dataset(dataset: str = 'muhammadshahidazeem/customer-churn-dataset', download_path: str = 'data/raw/'):
    
    """
    Download a dataset from Kaggle.

    Parameters
    ----------
    dataset : str
        The name of the dataset to download. The default is
        'muhammadshahidazeem/customer-churn-dataset'.
    download_path : str
        The path where the dataset will be downloaded. The default is
        'data/raw/'.

    Returns
    -------
    None
    """    
    
    # 1. Load Kaggle credentials
    
    load_kaggle_credentials()
    
    # 2. Authenticate in Kaggle
    
    api = KaggleApi()
    api.authenticate()
    
    # 3. Download dataset 
    
    print(f'Baixando dataset: {dataset}...')
    api.dataset_download_files(dataset, path = download_path, unzip = True)
    print(f'Download concluído! Arquivos salvos em: {download_path}')
    
if __name__ == '__main__':
    
    download_kaggle_dataset()