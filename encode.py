import os
import pandas as pd

def encode_categories(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Encode categorical (object, boolean) features and the target in the dataframe, except for true numeric columns.

    Args:
        df (pd.DataFrame): The DataFrame containing features and the target.
        target_column (str): The name of the target column.

    Returns:
        pd.DataFrame: Encoded DataFrame.
    """
    # Ensure the 'encode' folder exists
    os.makedirs('encode', exist_ok=True)

    encoded_df = df.copy()
    encoding_maps = {}

    # Open the CSV file for saving the mappings
    with open('encode/encoding_mapping.csv', 'w') as f:
        # Encode only categorical (object) and boolean columns
        for col in encoded_df.columns:
            if col == target_column:
                continue  # Skip the target column for now

            # Check if the column should be encoded (categorical or binary)
            if pd.api.types.is_object_dtype(encoded_df[col]) or pd.api.types.is_bool_dtype(encoded_df[col]):
                # Factorize the column if it's of type object or boolean
                encoded_df[col], mapping = pd.factorize(encoded_df[col])
                encoding_maps[col] = dict(enumerate(mapping))

                # Write the column mapping to CSV
                f.write(f"\n{col} Mapping:\n")
                f.write("Original Value,Encoded Value\n")
                for encoded_value, original_value in encoding_maps[col].items():
                    f.write(f"{original_value},{encoded_value}\n")
                    
    # Encode the target column only if it is categorical or boolean
    if pd.api.types.is_object_dtype(df[target_column]) or pd.api.types.is_bool_dtype(df[target_column]):
        encoded_df[target_column], target_mapping = pd.factorize(df[target_column])
        encoding_maps[target_column] = dict(enumerate(target_mapping))

        # Write the target column mapping to CSV
        with open('encode/encoding_mapping.csv', 'a') as f:
            f.write(f"\n{target_column} Mapping:\n")
            f.write("Original Value,Encoded Value\n")
            for encoded_value, original_value in encoding_maps[target_column].items():
                f.write(f"{original_value},{encoded_value}\n")

    # Save the encoded DataFrame to a new CSV file in the 'encode' folder
    encoded_df.to_csv('encode/encoded_data.csv', index=False)
    print("Encoded data has been saved to 'encode/encoded_data.csv'.")
    print("Encoding mapping table has been saved to 'encode/encoding_mapping.csv'.")

    return encoded_df
