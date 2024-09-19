import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def log_errors(error_type, df):
    df_with_index = df.copy()
    df_with_index['Index'] = df_with_index.index  # Add the index as a column
    try:
        with open("data_errors.csv", "a") as f:
            f.write(f"{error_type}:\n")
            df_with_index.to_csv(f, index=False, header=False)  # Append without headers
    except Exception as e:
        print(f"Error while logging data quality errors: {e}")




def data_quality_check(df, generate_diagrams):
    # 1. Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values in each column:\n", missing_values)
    
    if missing_values.sum() > 0:
        print("Rows with missing values:")
        missing_indices = df[df.isnull().any(axis=1)].index.tolist()
        for idx in missing_indices:
            print(f"Row {idx} with missing values:\n{df.loc[idx]}")
        log_errors("Missing Values", df[df.isnull().any(axis=1)])
    
    # 2. Check for duplicates
    duplicates = df[df.duplicated(keep=False)]
    unique_duplicates = duplicates.drop_duplicates()
    num_duplicates = unique_duplicates.shape[0]
    print(f"Number of duplicate rows: {num_duplicates}")

    if num_duplicates > 0:
        print("Duplicate rows and their indices:")
        duplicate_indices_list = []
        for _, row in unique_duplicates.iterrows():
            duplicate_indices = df[df.eq(row).all(axis=1)].index.tolist()
            duplicate_indices_list.append(','.join(map(str, duplicate_indices)))
            print(f"Duplicate group values: {tuple(row)}\nIndices: {duplicate_indices}")

        unique_duplicates_copy = unique_duplicates.copy()  # Create a copy of unique_duplicates before modifying it
        unique_duplicates_copy['Indices'] = duplicate_indices_list  # Safely modify the copy
        log_errors("Duplicate Rows", unique_duplicates_copy)  # Log with indices

    # 3. Check data types
    print("Data types of each column:\n", df.dtypes)

    # 4. Descriptive statistics
    print("Descriptive statistics:\n", df.describe())

    # 5. Value counts for categorical variables
    for column in df.select_dtypes(include=['object']).columns:
        value_counts = df[column].str.strip().str.lower().value_counts()
        print(f"Value counts for {column}:\n", value_counts)

    # 6. Range checks
    for column in df.select_dtypes(include=[np.number]).columns:
        print(f"Range of {column}: {df[column].min()} to {df[column].max()}")

    # 7. Outlier detection
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_df = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        if not outliers_df.empty:
            outliers[column] = outliers_df
            log_errors(f"Outliers in {column}", outliers_df)
    
    for column, outlier_df in outliers.items():
        if not outlier_df.empty:
            print(f"Outliers in {column}:\n", outlier_df)

    # Generate diagrams only if requested
    if generate_diagrams:
        # Generate box plots
        fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(10, 3 * len(numeric_columns)))
        for idx, column in enumerate(numeric_columns):
            sns.boxplot(x=df[column], ax=axes[idx])
            axes[idx].set_title(f"Boxplot of {column}")
        plt.subplots_adjust(hspace=0.9)
        plt.show()

        # Generate histograms
        fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(10, 3 * len(numeric_columns)))
        for idx, column in enumerate(numeric_columns):
            sns.histplot(df[column], kde=True, ax=axes[idx])
            axes[idx].set_title(f"Distribution of {column}")
        plt.subplots_adjust(hspace=0.9)
        plt.show()

        # Correlation matrix
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

    # 8. Constant columns
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    print("Constant columns:", constant_columns)
    if constant_columns:
        # Convert constant columns to a DataFrame with a single column for logging
        constant_columns_df = pd.DataFrame(constant_columns, columns=["Constant Columns"])
        log_errors("Constant Columns", constant_columns_df)
