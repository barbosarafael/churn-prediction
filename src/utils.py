# 0. Import libraries

from dotenv import load_dotenv
import os

def load_kaggle_credentials():
    
    """
    Carrega as credenciais do Kaggle da variável de ambiente local.
    
    Essa função carrega as credenciais do Kaggle armazenadas em um arquivo .env
    na pasta raiz do projeto. As credenciais s o necess rias para acessar a API do Kaggle.
    """
    
    load_dotenv()
    
    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")