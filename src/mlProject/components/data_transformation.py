import os
from mlProject import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_data(self):
        # Load the data
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets with randomization (0.75, 0.25) split.
        train, test = train_test_split(data, test_size=0.25, random_state=42)

        # Apply StandardScaler
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)

        # Apply PCA (Principal Component Analysis)
        #pca = PCA(n_components=2)  # You can specify the number of components as needed
        #train_pca = pca.fit_transform(train_scaled)
        #test_pca = pca.transform(test_scaled)

        # Save the transformed data
        train_scaled = pd.DataFrame(train_scaled)
        test_scaled = pd.DataFrame(test_scaled)

        train_scaled.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test_scaled.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Transformed data with StandardScaler and PCA and saved to CSV files with randomization")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)

'''
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
'''