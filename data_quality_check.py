import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def log_errors(error_type, df):
    df_with_index = df.copy()
    df_with_index['Index'] = df_with_index.index
    try:
        with open("Detected Errors/quality_errors.csv", "a") as f:
            f.write(f"{error_type}:\n")
            df_with_index.to_csv(f, index=False, header=False)
    except Exception as e:
        print(f"Error while logging data quality errors: {e}")

def data_quality_check(df, generate_diagrams):
    print("\n--- Data Quality Check ---\n")

    # 1. Check for missing values
    missing_values = df.isnull().sum()
    print("\n1. Missing Values:")
    print("Columns with missing values:")
    print(missing_values[missing_values > 0].to_string(), "\n")

    if missing_values.sum() > 0:
        print("Rows with missing values:")
        missing_rows = df[df.isnull().any(axis=1)]
        for idx, row in missing_rows.iterrows():
            print(f"Row {idx}: {row.to_dict()}")
        log_errors("Missing Values", missing_rows)

    # 2. Check for duplicates
    duplicates = df[df.duplicated(keep=False)]
    num_duplicates = len(duplicates.drop_duplicates())
    print("\n2. Duplicate Rows:")
    print(f"Total number of duplicate rows: {num_duplicates}")
    
    if num_duplicates > 0:
        print("Duplicate rows and their indices:")
        unique_duplicates = duplicates.drop_duplicates()
        duplicate_info = []
        for idx, row in unique_duplicates.iterrows():
            indices = df[df.eq(row).all(axis=1)].index.tolist()
            duplicate_info.append((row.to_dict(), indices))
            print(f"Values: {row.to_dict()}, Indices: {indices}")
        unique_duplicates['Indices'] = [','.join(map(str, idx_list)) for _, idx_list in duplicate_info]
        log_errors("Duplicate Rows", unique_duplicates)

    # 3. Check data types
    print("\n3. Data Types:")
    print(df.dtypes.to_string(), "\n")

    # 4. Descriptive statistics
    print("\n4. Descriptive Statistics:")
    print(df.describe().to_string(), "\n")

    # 5. Value counts for categorical variables
    print("\n5. Value Counts for Categorical Variables:")
    for column in df.select_dtypes(include=['object']).columns:
        print(f"\n{column}:")
        print(df[column].value_counts().to_string())

    # 6. Range checks for numeric columns
    print("\n6. Range of Numeric Columns:")
    for column in df.select_dtypes(include=[np.number]).columns:
        min_val, max_val = df[column].min(), df[column].max()
        print(f"{column}: {min_val} to {max_val}")

    # 7. Outlier detection
    print("\n7. Outlier Detection:")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for column in numeric_columns:
        Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers_df = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        if not outliers_df.empty:
            outliers[column] = outliers_df
            print(f"Outliers in {column}: {outliers_df.index.tolist()}")
            log_errors(f"Outliers in {column}", outliers_df)

    # Generate diagrams only if requested
    if generate_diagrams:
        # Generate box plots
        fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(10, 3 * len(numeric_columns)))
        if len(numeric_columns) == 1:
            axes = [axes]
        for idx, column in enumerate(numeric_columns):
            sns.boxplot(x=df[column], ax=axes[idx])
            axes[idx].set_title(f"Boxplot of {column}")
        plt.subplots_adjust(hspace=0.9)
        plt.show()

        # Generate histograms
        fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(10, 3 * len(numeric_columns)))
        if len(numeric_columns) == 1:
            axes = [axes]
        for idx, column in enumerate(numeric_columns):
            sns.histplot(df[column], kde=True, ax=axes[idx])
            axes[idx].set_title(f"Distribution of {column}")
        plt.subplots_adjust(hspace=0.9)
        plt.show()

        # Correlation matrix
        if len(numeric_columns) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numeric_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
            plt.title("Correlation Matrix")
            plt.show()
        else:
            print("\nCorrelation matrix requires at least two numeric columns.\n")

    # 8. Constant columns
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    print("\n8. Constant Columns:")
    if constant_columns:
        print(f"Constant columns found: {constant_columns}")
        log_errors("Constant Columns", pd.DataFrame(constant_columns, columns=["Constant Columns"]))
    else:
        print("No constant columns found.")

    print("\n--- End of Data Quality Check ---\n")
