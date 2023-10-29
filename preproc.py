import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

class Dataset(object):
    def __init__(self, data: pd.DataFrame) -> None:
        # drop columns
        self.data = data.copy()

        # drop columns
        self.data = self.data.drop('pdeath',axis=1)
        self.data = self.data.drop('psych4',axis=1)
        self.data = self.data.drop('glucose',axis=1)
        self.data = self.data.drop('bloodchem4', axis=1)
        self.data = self.data.drop('urine',axis =1)
        self.data = self.data.drop('income',axis =1)

        # clean data
        self.data = self.data.apply(self.clean, axis=1)

        # replace missing data
        self.clean_fill_mean('psych2')
        self.clean_fill_mean('bloodchem3')
        self.replace_missing_with_knn('totalcost', n_neighbors=10)
        self.replace_missing_with_knn('confidence', n_neighbors=10)
        self.replace_missing_with_knn('bloodchem1', n_neighbors=10)
        self.replace_missing_with_knn('bloodchem2', n_neighbors=10)
        self.replace_missing_with_knn('blood', n_neighbors=10)
        self.replace_missing_with_knn('cost', n_neighbors=10)
        self.replace_missing_with_knn('sleep', n_neighbors=10)
        self.replace_missing_with_knn('bloodchem5', n_neighbors=10)
        self.replace_missing_with_mode('disability')
        self.replace_missing_with_knn('administratorcost')
        self.replace_missing_with_knn('diabetes')
        self.replace_missing_with_knn('bloodchem6')
        self.replace_missing_with_knn('education')
        self.replace_missing_with_knn('psych5')
        self.replace_missing_with_knn('psych6')
        self.replace_missing_with_knn('information')


        # one hot encode the data
        self.one_hot_encode_feature('cancer')
        self.one_hot_encode_feature('extraprimary')
        self.one_hot_encode_feature('dnr')
        self.one_hot_encode_feature('primary')
        self.one_hot_encode_feature('disability')


    
    def clean(self, row):
        row.sex = self.cleanSex(row.sex)
        row.race = self.cleanRace(row.race)
        row.cost = self.cleanCost(row.cost)
        
        return row
    
    # preproc for cost
    def cleanCost(self, val: float):
        if pd.isna(val) or val < 0:
            return np.nan
        
        return val


    # preproc logic for cleaning sex
    def cleanSex(self, val):
        val = val.lower()
        # 1: male
        if val in ['male', 'm', '1']:
            return 1
        # 0: female
        return 0

    # preproc logic for race
    def cleanRace(self, val):
        # unique values: ['white', 'black', 'hispanic', 'other', nan, 'asian']
        WHITE = 0
        BLACK = 1
        HISPANIC = 2
        OTHER = 3
        ASIAN = 4

        if(pd.isna(val)):
            return OTHER
        
        val = val.lower()

        if val == 'white':
            return WHITE
        elif val == 'black':
            return BLACK
        elif val == 'hispanic':
            return HISPANIC
        elif val == 'other':
            return OTHER
        elif val == 'asian':
            return ASIAN


        print('not possible')
        return -1
    

    def one_hot_encode_feature(self, feature_name):
        """
        One-hot encodes a specified feature from a DataFrame.

        Parameters:
        - dataframe: The input DataFrame.
        - feature_name: The name of the feature to be one-hot encoded.

        Returns:
        - one_hot_df: A DataFrame containing the one-hot encoded feature.
        """

        # Select the specified feature from the DataFrame
        feature_to_encode = self.data[feature_name]

        # Reshape the feature to have a 2D shape, required by OneHotEncoder
        feature_to_encode = feature_to_encode.values.reshape(-1, 1)

        # Create an instance of the OneHotEncoder
        encoder = OneHotEncoder(sparse=False)  # You can set sparse=True if you want a sparse matrix

        # Fit the encoder to the feature data
        encoder.fit(feature_to_encode)

        # Transform the feature data to one-hot encoded format
        one_hot_encoded = encoder.transform(feature_to_encode)

        # Convert the one-hot encoded data to a DataFrame for better visualization
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out([feature_name]))
        
        
        new_dataframe = self.data.drop(columns=[feature_name])  # Remove the target column
        new_dataframe[one_hot_df.columns] = one_hot_df  # Add the source columns to the target DataFrame
        
        self.data = new_dataframe

    # for glucose, psych2, 
    def clean_fill_mean(self, feature):
        mean_value = self.data[feature].mean()
        self.data[feature].fillna(mean_value, inplace=True)


    # replacing the outliers after replacing missing values
    def replace_outliers_with_mean(self, column_name, threshold=1.5):
        # Calculate lower and upper bounds for outliers
        Q1 = self.data[column_name].quantile(0.25)
        Q3 = self.data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Identify outliers in the specified column
        outliers = self.data[(self.data[column_name] < lower_bound) | (self.data[column_name] > upper_bound)]

        # Replace outliers with the mean of the column
        non_outliers_mean = self.data[([column_name] >= lower_bound) & (self.data[column_name] <= upper_bound)][column_name].mean()
        self.data.loc[outliers.index, column_name] = non_outliers_mean


    #replacing missing values with knn imputer
    # for totalcost
    def replace_missing_with_knn(self, column_name, n_neighbors=5):
        # Create a copy of the DataFrame to avoid modifying the original data
        df_imputed = self.data.copy()    
        # Extract the column with missing values for imputation
        column_to_impute = df_imputed[[column_name]]   
        # Initialize KNNImputer with the desired number of neighbors
        imputer = KNNImputer(n_neighbors=n_neighbors)   
        # Perform KNN imputation on the specified column
        column_imputed = imputer.fit_transform(column_to_impute)   
        # Replace the missing values in the original DataFrame with imputed values
        df_imputed[column_name] = column_imputed
        
        self.data = df_imputed

    
    def replace_missing_with_mode(self, categorical_feature):
        """
        Fill missing values in a categorical feature with the most frequent category.

        Parameters:
        - df: DataFrame containing the data.
        - categorical_feature: Name of the categorical feature/column with missing values.

        Returns:
        - Updated DataFrame with missing values filled in the specified feature.
        """
        # Find the most frequent category in the specified feature
        most_frequent_category = self.data[categorical_feature].mode()[0]
        
        # Fill missing values in the specified feature with the most frequent category
        self.data[categorical_feature].fillna(most_frequent_category, inplace=True)
        
    # Example Usage:
    # Assuming 'df' is your DataFrame and 'categorical_column' is the name of the categorical feature with missing values
    # df = fill_missing_categorical(df, 'categorical_column')



class Dataset2(object):
    def __init__(self, data: pd.DataFrame, is_test=False) -> None:
        # drop columns
        self.mean = {}
        self.mode = 0
        self.data = data.copy()
        self.is_test = is_test

        if self.is_test:
            self.data[self.data == 'nan'] = np.nan

        # drop columns
        self.data = self.data.drop('pdeath',axis=1)
        self.data = self.data.drop('psych4',axis=1)
        self.data = self.data.drop('glucose',axis=1)
        self.data = self.data.drop('bloodchem4', axis=1)
        self.data = self.data.drop('urine',axis =1)
        self.data = self.data.drop('income',axis =1)
        self.data = self.data.drop('disability', axis = 1)

        # clean data
        self.data = self.data.apply(self.clean, axis=1)

        # replace missing data
        self.clean_fill_mean('psych2')
        self.clean_fill_mean('bloodchem3')
        self.clean_fill_mean('totalcost')
        self.clean_fill_mean('confidence')
        self.clean_fill_mean('bloodchem1')
        self.clean_fill_mean('bloodchem2')
        self.clean_fill_mean('blood')
        self.clean_fill_mean('cost')
        self.clean_fill_mean('sleep')
        self.clean_fill_mean('bloodchem5')
#         self.replace_missing_with_mode('disability')
        self.clean_fill_mean('administratorcost')
        self.clean_fill_mean('diabetes')
        self.clean_fill_mean('bloodchem6')
        self.clean_fill_mean('education')
        self.clean_fill_mean('psych5')
        self.clean_fill_mean('psych6')
        self.clean_fill_mean('information')


        # one hot encode the data
        self.one_hot_encode_feature('cancer')
        self.one_hot_encode_feature('extraprimary')
        self.one_hot_encode_feature('dnr')
        self.one_hot_encode_feature('primary')
        self.one_hot_encode_feature('disability')
        
#         joblib.dump(self.knn_imputers, "knn_imputers.pk1")


        
    def clean(self, row):
        row.sex = self.cleanSex(row.sex)
        row.race = self.cleanRace(row.race)
        row.cost = self.cleanCost(row.cost)
        
        return row
    
    # preproc for cost
    def cleanCost(self, val: float):
        if pd.isna(val) or val < 0:
            return np.nan
        
        return val


    # preproc logic for cleaning sex
    def cleanSex(self, val):
        val = val.lower()
        # 1: male
        if val in ['male', 'm', '1']:
            return 1
        # 0: female
        return 0

    # preproc logic for race
    def cleanRace(self, val):
        # unique values: ['white', 'black', 'hispanic', 'other', nan, 'asian']
        WHITE = 0
        BLACK = 1
        HISPANIC = 2
        OTHER = 3
        ASIAN = 4

        if(pd.isna(val)):
            return OTHER
        
        val = val.lower()

        if val == 'white':
            return WHITE
        elif val == 'black':
            return BLACK
        elif val == 'hispanic':
            return HISPANIC
        elif val == 'other':
            return OTHER
        elif val == 'asian':
            return ASIAN


        print('not possible')
        return -1
    

    def one_hot_encode_feature(self, feature_name):
        """
        One-hot encodes a specified feature from a DataFrame.

        Parameters:
        - dataframe: The input DataFrame.
        - feature_name: The name of the feature to be one-hot encoded.

        Returns:
        - one_hot_df: A DataFrame containing the one-hot encoded feature.
        """

        # Select the specified feature from the DataFrame
        feature_to_encode = self.data[feature_name]

        # Reshape the feature to have a 2D shape, required by OneHotEncoder
        feature_to_encode = feature_to_encode.values.reshape(-1, 1)

        # Create an instance of the OneHotEncoder
        encoder = OneHotEncoder(sparse=False)  # You can set sparse=True if you want a sparse matrix

        # Fit the encoder to the feature data
        encoder.fit(feature_to_encode)

        # Transform the feature data to one-hot encoded format
        one_hot_encoded = encoder.transform(feature_to_encode)

        # Convert the one-hot encoded data to a DataFrame for better visualization
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out([feature_name]))
        
        
        new_dataframe = self.data.drop(columns=[feature_name])  # Remove the target column
        new_dataframe[one_hot_df.columns] = one_hot_df  # Add the source columns to the target DataFrame
        
        self.data = new_dataframe

    # for glucose, psych2, 
    def clean_fill_mean(self, feature):
        if self.is_test:
            self.data[feature].fillna(self.mean[feature], inplace=True)
        else:
            mean_value = self.data[feature].mean()
            self.mean[feature] = mean_value
            self.data[feature].fillna(mean_value, inplace=True)


    # replacing the outliers after replacing missing values
    def replace_outliers_with_mean(self, column_name, threshold=1.5):
        # Calculate lower and upper bounds for outliers
        Q1 = self.data[column_name].quantile(0.25)
        Q3 = self.data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Identify outliers in the specified column
        outliers = self.data[(self.data[column_name] < lower_bound) | (self.data[column_name] > upper_bound)]

        # Replace outliers with the mean of the column
        non_outliers_mean = self.data[([column_name] >= lower_bound) & (self.data[column_name] <= upper_bound)][column_name].mean()
        self.data.loc[outliers.index, column_name] = non_outliers_mean


    #replacing missing values with knn imputer
    # for totalcost
    def replace_missing_with_knn(self, column_name, n_neighbors=5):
        # Create a copy of the DataFrame to avoid modifying the original data
        df_imputed = self.data.copy()    
        # Extract the column with missing values for imputation
        column_to_impute = df_imputed[[column_name]]   
        # Initialize KNNImputer with the desired number of neighbors
        imputer = KNNImputer(n_neighbors=n_neighbors)   
        # Perform KNN imputation on the specified column
        column_imputed = imputer.fit_transform(column_to_impute) 
        
        self.knn_imputers[column_name] = imputer
        # Replace the missing values in the original DataFrame with imputed values
        df_imputed[column_name] = column_imputed
        
        self.data = df_imputed

    
    def replace_missing_with_mode(self, categorical_feature):
        """
        Fill missing values in a categorical feature with the most frequent category.

        Parameters:
        - df: DataFrame containing the data.
        - categorical_feature: Name of the categorical feature/column with missing values.

        Returns:
        - Updated DataFrame with missing values filled in the specified feature.
        """
        if self.is_test:
            self.data[categorical_feature].fillna(self.mode, inplace=True)
        else:
            # Find the most frequent category in the specified feature
            most_frequent_category = self.data[categorical_feature].mode()[0]
            self.mode = most_frequent_category
        
            # Fill missing values in the specified feature with the most frequent category
            self.data[categorical_feature].fillna(most_frequent_category, inplace=True)
        
    # Example Usage:
    # Assuming 'df' is your DataFrame and 'categorical_column' is the name of the categorical feature with missing values
    # df = fill_missing_categorical(df, 'categorical_column')
    
if __name__ == 'main':
    data = pd.read_csv('./TD_HOSPITAL_TRAIN.csv')

    cleaned = Dataset(data)

    print(cleaned.shape)