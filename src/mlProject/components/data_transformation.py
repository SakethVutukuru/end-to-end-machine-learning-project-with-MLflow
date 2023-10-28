import os
import pandas as pd
from mlProject import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_data(self):
        # Load the data
        data = pd.read_csv(self.config.data_path)

        # Shuffle the data
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Define the target variable
        target_variable = 'quality'

        # Split the data into features (X) and the target variable (y)
        X = data.drop(target_variable, axis=1)
        y = data[target_variable]

        # Split the data into training and test sets with randomization (0.75, 0.25) split.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Apply StandardScaler only to the features (X)
        scaler = StandardScaler()
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply PCA (Principal Component Analysis) if needed
        # pca = PCA(n_components=2)  # You can specify the number of components as needed
        # X_train_pca = pca.fit_transform(X_train_scaled)
        # X_test_pca = pca.transform(X_test_scaled)

        # Save the transformed data
        train_data = pd.DataFrame(X_train_scaled, columns=X.columns)
        test_data = pd.DataFrame(X_test_scaled, columns=X.columns)

        train_data[target_variable] = y_train
        test_data[target_variable] = y_test

        train_data.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test_data.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Transformed data with StandardScaler and saved to CSV files with randomization")
        logger.info(train_data.shape)
        logger.info(test_data.shape)

        print(train_data.shape)
        print(test_data.shape)