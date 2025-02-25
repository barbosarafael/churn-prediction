from utils import load_kaggle_credentials
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_dataset():
    
    """
    Baixa um dataset do Kaggle e salva em uma pasta local.

    Args:
        dataset (str): Nome do dataset a ser baixado.
        download_path (str, optional): Pasta local para salvar os arquivos do dataset. Defaults to 'data/raw/'.

    Returns:
        None
    """
    
    # 1. Load Kaggle credentials
    
    load_kaggle_credentials()
    
    # 2. Authenticate in Kaggle
    
    api = KaggleApi()
    api.authenticate()
    
    # 3. Download dataset 
    
    dataset = 'muhammadshahidazeem/customer-churn-dataset'
    download_path = 'data/raw/'
    
    print(f'Baixando dataset: {dataset}...')
    api.dataset_download_files(dataset, path = download_path, unzip = True)
    print(f'Download concluído! Arquivos salvos em: {download_path}')
    
if __name__ == '__main__':
    download_kaggle_dataset()