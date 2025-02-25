# Customer Churn

## About

Machine learning project to practice customer churn analysis and explore topics such as class imbalance, optimal evaluation metrics and model calibration in classification context.

## Usage

### 1. Download raw data

This [Kaggle](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?select=customer_churn_dataset-training-master.csv) Customer Churn Dataset contains data to predict customer's retention. Follow the instructions:


1. Install the required libraries by running `pip install -r requirements.txt`
2. Create a `.env` file with your Kaggle credentials in root folder
3. Run the script by executing `python download_kaggle_dataset.py`
4. Two datasets will be downloaded and saved in the `data/raw/` directory
   1. train.csv
   2. test.csv