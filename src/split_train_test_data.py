import pandas as pd
from sklearn.model_selection import train_test_split
import os

from typing import Tuple

def split_data_frame(
    data_frame: pd.DataFrame, test_ratio: float = 0.1, random_seed: int = 42,
    output_directory: str = 'data/processed/'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide the input DataFrame into a training set and a test set.

    Args:
        data_frame: The DataFrame to be split.
        test_ratio: The proportion of the test set. Defaults to 0.2.
        random_seed: The seed for the random number generator. Defaults to 42.
        output_directory: The directory where the split data will be saved. Defaults to 'data/processed/'.

    Returns:
        A tuple containing the training set and the test set.
    """

    training_set, test_set = train_test_split(
        data_frame,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=data_frame['Churn'] if 'Churn' in data_frame.columns else None
    )

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    training_set.to_csv(os.path.join(output_directory, 'train.csv'), index=False)
    test_set.to_csv(os.path.join(output_directory, 'test.csv'), index=False)
    
    print('Split data successfully!')
    
if __name__ == '__main__':
    
    dataset = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    split_data_frame(data_frame = dataset)