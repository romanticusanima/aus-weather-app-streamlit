import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, Any, List

def drop_na_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drop rows with NA values in the specified columns.

    Args:
        df (pd.DataFrame): The raw dataframe.
        columns (list): List of columns to check for NA values.

    Returns:
        pd.DataFrame: DataFrame with NA values dropped.
    """
    return df.dropna(subset=columns)

def split_data_by_year(df: pd.DataFrame, year_col: str) -> Dict[str, pd.DataFrame]:
    """
    Split the dataframe into training, validation, and test sets
    based on the year.

    Args:
        df (pd.DataFrame): The raw dataframe.
        year_col (str): The column containing year information.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing the train,
                                 validation, and test dataframes.
    """
    year = pd.to_datetime(df[year_col]).dt.year
    train_df = df[year < 2015]
    val_df = df[year == 2015]
    test_df = df[year > 2015]
    return {'train': train_df, 'val': val_df, 'test': test_df}


def create_inputs_targets(df_dict: Dict[str, pd.DataFrame], input_cols: list, target_col: str) -> Dict[str, Any]:
    """
    Create inputs and targets for training and validation sets.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing the train and validation dataframes.
        input_cols (list): List of input columns.
        target_col (str): Target column.

    Returns:
        Dict[str, Any]: Dictionary containing inputs and targets for train and val sets.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data

def impute_missing_values(data: Dict[str, Any], numeric_cols: list) -> Any:
    """
    Impute missing numerical values using the mean strategy.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets
                               for train, validation, and test sets.
        numeric_cols (list): List of numerical columns.

    Returns:
        Imputer.
    """
    imputer = (SimpleImputer(strategy='mean').fit(data['train_inputs'][numeric_cols]))
    for split in ['train', 'val', 'test']:
        data[f'{split}_inputs'][numeric_cols] = imputer.transform(data[f'{split}_inputs'][numeric_cols])
    return imputer


def scale_numeric_features(data: Dict[str, Any], numeric_cols: list) -> Any:
    """
    Scale numeric features using MinMaxScaler.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train and val sets.
        numeric_cols (list): List of numerical columns.

    Returns:
        Scaler.
    """
    scaler = MinMaxScaler().fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val', 'test']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(data[f'{split}_inputs'][numeric_cols])
    return scaler


def encode_categorical_features(data: Dict[str, Any], categorical_cols: list) -> Dict[str, Any]:
    """
    One-hot encode categorical features.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train and val sets.
        categorical_cols (list): List of categorical columns.

    Returns:
        Dict[str, Any]: Dictionary containing encoder and encoded_cols.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data['train_inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    for split in ['train', 'val', 'test']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        data[f'{split}_inputs'] = pd.concat([data[f'{split}_inputs'], pd.DataFrame(encoded, columns=encoded_cols, index=data[f'{split}_inputs'].index)], axis=1)
        data[f'{split}_inputs'].drop(columns=categorical_cols, inplace=True)
    return {
        'encoder': encoder,
        'encoded_cols': encoded_cols,
    }


def preprocess_data(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Preprocess the raw dataframe.
    This function:
      - Drops rows with missing values in the 'Exited' column.
      - Drops unnecessary columns such as 'CustomerId' and 'Surname'.
      - Splits the data into training and validation sets.
      - Extracts numeric and categorical columns.
      - Applies Imputer for missing numerical values.
      - Applies MinMax scaling to numeric features.
      - Applies one-hot encoding to categorical features.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - train_inputs, val_inputs: processed feature sets
            - train_targets, val_targets: target labels
            - input_cols: list of input feature columns
            - numeric_cols: list of numeric feature names
            - categorical_cols: list of categorical feature names
            - imputer: fitted imputer
            - scaler: fitted scaler
            - encoder: fitted OneHotEncoder
            - encoded_cols: names of encoded categorical columns
    """
    raw_df = drop_na_values(raw_df, ['RainToday', 'RainTomorrow'])
    split_dfs = split_data_by_year(raw_df, 'Date')

    input_cols = list(raw_df.columns)[1:-1] # ignore Date, RainTomorrow
    target_col = 'RainTomorrow'
    
    data = create_inputs_targets(split_dfs, input_cols, target_col)

    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes('object').columns.tolist()

    data['input_cols'] = input_cols
    data['target_col'] = target_col
    data['numeric_cols'] = numeric_cols
    data['categorical_cols'] = categorical_cols

    data['imputer'] = impute_missing_values(data, numeric_cols)
    data['scaler'] = scale_numeric_features(data, numeric_cols)

    encoded_data = encode_categorical_features(data, categorical_cols)
    data['encoder'] = encoded_data['encoder']
    data['encoded_cols'] = encoded_data['encoded_cols']

    return data


def preprocess_new_data(input_data: Dict[str, Any], model_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess a new dataset using the fitted imputer, scaler and encoder.

    Args:
        input_data (Dict[str, Any]): Dictionary containing the raw innput data.
        model_dict (Dict[str, Any]): Dictionary with trained model and preprocessing
                                     objects (imputer, scaler, encoder, etc.).

    Returns:
        pd.DataFrame: A DataFrame containing the transformed numeric and encoded categorical features.
    """

    input_df = pd.DataFrame([input_data])
    
    numeric_cols = model_dict['numeric_cols']
    categorical_cols = model_dict['categorical_cols']
    
    imputer = model_dict['imputer']
    scaler = model_dict['scaler']
    encoder = model_dict['encoder']

    preprocessed_data = input_df[model_dict['input_cols']].copy()

    preprocessed_data[numeric_cols] = imputer.transform(preprocessed_data[numeric_cols])
    preprocessed_data[numeric_cols] = scaler.transform(preprocessed_data[numeric_cols])

    encoded = encoder.transform(preprocessed_data[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=preprocessed_data.index
    )

    preprocessed_data = pd.concat(
        [preprocessed_data.drop(columns=categorical_cols), encoded_df],
        axis=1
    )

    return preprocessed_data