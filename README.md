# Customer Churn

## About

Machine learning project to practice customer churn analysis and explore topics such as class imbalance, optimal evaluation metrics and model calibration in classification context. This projects use MLFlow to track and compare experiments.

## Usage

### 1. Download raw data

This [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) Customer Churn Dataset contains data to predict customer's retention. Follow the instructions:

1. Install the required libraries by running `pip install -r requirements.txt`
2. Create a `.env` file with your Kaggle credentials (`username and key`) in root folder. You can find this credentials [here](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication) 
3. Run the script by executing `python src/data_loader.py`
4. The raw data will be downloaded and saved in the `data/raw/` directory

### 2. Split train and test data

Run the script `python src/split_train_test_data.py`

### 3. Exploratory Data Analysis (if needed)

Open and run the notebook `01-exploratory-data-analysis.ipynb` 

### 4. Train the model

This project uses classification models such as Logistic Regression, Decision Tree and XGBoost. Our metric to optimize is the **f1-score**, because I want to garantee a high recall and a good precision. 

In MLFlow we will track others metrics: 

- AUC (Area Under the Curve)
- AUC-Precision-Recall (AUC-PR)
- Kolmogorov-Smirnov score
- Confusion matrix metrics:
  - True positives
  - False positives
  - True negatives
  - False negatives
- Accuracy

In training, I use RandomSearch to find the best parameters for each model. By default, I'm using 20 iterations per model. If you want to change the number of iterations, you can change the `n_iter` parameter in the `parameters.yml`.

In `src/utils_modelling.py` you can find the functions and change the parameters of models and encoders. The functions are:

- `get_encoders`: dictionary of encoders. 
- `get_models_and_param_grids`: dictionary of models and parameter grids

To train the models, run the script `python src/train.py`. You will see the models running like "**Combinação 19/20 para XGBClassifier**"
- The best model will be saved in the `models/` directory

### 5. Model evaluation 

1. After training model, automatically you will see the metrics of the best model 
2. If you want to see detailed results by model, run `mlflow ui` to open MLFlow UI. Open `http://127.0.0.1:5000` in your browser
3. Click in the experiment "Churn Prediction Experiment v99" and explore the results, as shown in the image below

![alt text](images/image.png)

## 6. Model calibration

Our probabilities need to be reliable. For example, we need to guarantee that a customer has an X% probability to churn. To do this, we need to calibrate our model. I made a very basic approach to calibrate our model.

1. Open the script `src/calibrate_probabilities.py` 
2. Change id variable to the id of the model you want to calibrate
3. Run the script `python src/calibrate_probabilities.py` 
4. The results are the following:
   1. A new folder with **calibrated_models**: you can find the model's metadata and the calibration curve
   2. Brier score for each model
   3. A metadata containing the optimal threshold for each model
  
## 7. Model Deployment

The main goal of this project is to learn different ways to deploy a model in production, specifically by creating an API to serve predictions. I containerized the API using Docker and deployed it on AWS.

I faced some challenges deploying the model with AWS Lambda and API Gateway because my requirements were too large (XGBoost was the main issue). Although Lambda has a 10GB limit, I couldn't find a feasible way to work around this constraint.

To overcome this problem, I used the following AWS services:

- **S3**: To store the model, preprocessor artifacts, and Dockerfile.
- **CodeBuild**: To install dependencies and build the project.
- **Elastic Container Registry (ECR)**: To store the Docker image.
- **API Gateway**: To expose the API.
- **Lambda**: To serve the API.
- **IAM**: To manage permissions for Lambda and other services.
- **CloudWatch**: To monitor logs.

The files related to deployment are:

- `src/lambda_handler.py`
- `Dockerfile`

Finally, the goal was achieved! The image below shows the API in action.

![alt text](images/result.png)

Since this was a study-focused project, the API is no longer available, but everything is documented for a future round 🙃

## Steps

- [x] "Data exploration"
- [x] Feature engineering
- [x] Model training
- [x] Model evaluation
- [x] Model calibration
- [x] Translate 100% to english
- [x] Documentation all functions
- [x] Model deployment -> API + Docker + AWS
- [x] End project
- [ ] ~~Model monitoring~~
- [ ] ~~Deployment monitoring~~