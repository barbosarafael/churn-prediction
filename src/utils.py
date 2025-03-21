# 0. Import libraries

from dotenv import load_dotenv
import os

def load_kaggle_credentials():
    
    """
    Loads Kaggle credentials from local environment variable.
    
    This function loads Kaggle credentials stored in a .env file in the project's root directory.
    The credentials are necessary to access the Kaggle API.
    """
    
    load_dotenv()
    
    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")
