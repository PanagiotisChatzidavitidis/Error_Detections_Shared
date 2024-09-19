import pandas as pd



def encode_target(y: pd.Series) -> pd.Series:
    """Encode the target variable."""
    if y.dtype == 'object':
        y_encoded = pd.Series(pd.factorize(y)[0], index=y.index)  # Ensuring the Series retains the original index
        return y_encoded
    return y


def encode_categories(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Encode categorical features in the dataframe."""
    encoded_df = X.copy()
    encoding_maps = {}

    for col in encoded_df.select_dtypes(include=['object']).columns:
        encoded_df[col], mapping = pd.factorize(encoded_df[col])
        encoding_maps[col] = dict(enumerate(mapping))

    return encoded_df, encoding_maps
