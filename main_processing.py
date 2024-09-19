import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import tkinter as tk
from tkinter import filedialog, ttk
import warnings
from data_quality_check import data_quality_check
from encode import encode_categories, encode_target  # Import custom encoding functions

# Suppress warnings
warnings.filterwarnings('ignore')


def clear_csv_file(file_path: str) -> None:
    if os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write("")  # Clear the filε
        print(f"Cleared {file_path}.")
    else:
        print(f"{file_path} does not exist.")
        
#Ending program when X is clicked
def on_closing(root):
    print("Ending the program...")
    root.quit()    # Exit the `mainloop()`
    root.destroy()  # Destroy the window and end the program
    os._exit(0)    # Ensure the program terminates completely
    
#Loading Data
def load_dataset(file_path: str) -> pd.DataFrame:
    print("Loading the dataset...")
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    return df

#Select Target column for error Detection
def select_target_column(columns: list[str]) -> str:
    def on_submit():
        selected_column.set(dropdown.get())
        root.quit()
        root.destroy()

    root = tk.Tk()
    selected_column = tk.StringVar()

    root.title("Choose Target Column")
    root.geometry("400x200")
    root.resizable(False, False)

    main_frame = ttk.Frame(root, padding="20 20 20 20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(main_frame, text="Select the column to predict:").grid(row=0, column=0, padx=10, pady=10)
    
    dropdown = ttk.Combobox(main_frame, textvariable=selected_column, values=list(columns), state="readonly")
    dropdown.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

    submit_button = ttk.Button(main_frame, text="Submit", command=on_submit)
    submit_button.grid(row=2, column=0, padx=10, pady=20)

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=1)

    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
    
    root.mainloop()

    return selected_column.get()

def ask_options() -> dict[str, tk.BooleanVar]:
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
    options = {
        "data_quality": tk.BooleanVar(),
        "plot_diagrams": tk.BooleanVar(),
        "ml_error_detection": tk.BooleanVar(),
        "cross_validation": tk.BooleanVar()
    }

    # Function to show/hide "Plot Diagrams" checkbox based on "Data Quality Check" selection
    def update_plot_diagrams_visibility():
        if options["data_quality"].get():
            plot_diagrams_checkbox.grid(row=1, column=1, padx=10, pady=5)
        else:
            options["plot_diagrams"].set(False)  # Automatically uncheck "Plot Diagrams"
            plot_diagrams_checkbox.grid_forget()

    # Function to show/hide "Cross-Validation" checkbox based on "ML Error Detection" selection
    def update_cross_validation_visibility():
        if options["ml_error_detection"].get():
            cross_validation_checkbox.grid(row=3, column=1, padx=10, pady=5)
        else:
            options["cross_validation"].set(False)  # Automatically uncheck "Cross-Validation"
            cross_validation_checkbox.grid_forget()

    def on_submit():
        root.quit()
        root.destroy()

    root.title("Choose Options")
    window_width = 350
    window_height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 5 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    options_frame = ttk.Frame(root)
    options_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    tk.Label(options_frame, text="Select options for analysis:").grid(row=0, column=0, columnspan=2, pady=10)

    # Data Quality Check Checkbox
    data_quality_checkbox = tk.Checkbutton(options_frame, text="Data Quality Check", variable=options["data_quality"], command=update_plot_diagrams_visibility)
    data_quality_checkbox.grid(row=1, column=0, sticky='w', padx=10)

    # Plot Diagrams Checkbox (hidden initially)
    plot_diagrams_checkbox = tk.Checkbutton(options_frame, text="Plot Diagrams", variable=options["plot_diagrams"])

    # Machine Learning Error Detection Checkbox
    ml_error_detection_checkbox = tk.Checkbutton(options_frame, text="Machine Learning Error Detection", variable=options["ml_error_detection"], command=update_cross_validation_visibility)
    ml_error_detection_checkbox.grid(row=3, column=0, sticky='w', padx=10)

    # Cross-Validation Checkbox (hidden initially)
    cross_validation_checkbox = tk.Checkbutton(options_frame, text="Cross-Validation", variable=options["cross_validation"])

    #Quit button
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
    
    # Submit button
    submit_button = ttk.Button(options_frame, text="Submit", command=on_submit)
    submit_button.grid(row=4, column=0, columnspan=2, pady=10)

    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))

    root.mainloop()

    return options

#File selection
def select_file() -> str:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
    root.destroy()
    return file_path

#Dynamic number of folds based of number of records
def get_dynamic_splits(n_samples: int) -> int:
    if n_samples < 100:
        return 3
    elif 100 <= n_samples < 1000:
        return 5
    else:
        return 10

#Cross Validation Funtion
def evaluate_model_with_cross_validation(model, X: pd.DataFrame, y: pd.Series, is_target_integer: bool) -> float:
    print("\nRunning cross-validation...")
    scoring_metric = 'accuracy' if is_target_integer else 'neg_mean_squared_error'
    kf = KFold(n_splits=get_dynamic_splits(len(X)), shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring_metric)
    
    cv_mean_score = np.mean(cv_scores)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Average Cross-Validation Score: {cv_mean_score:.4f}")
    return cv_mean_score


def check_data_quality(X: pd.DataFrame, y: pd.Series) -> None:
    if X.isnull().values.any():
        print("Warning: Missing values found in features after imputation!")
    if y.isnull().values.any():
        print("Warning: Missing values found in target variable!")

    zero_variance_features = X.columns[X.var() == 0]
    if len(zero_variance_features) > 0:
        print(f"Warning: These features have zero variance and might cause issues: {zero_variance_features}")

def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    is_target_integer: bool,
    perform_cross_validation: bool = False
) -> tuple[object, str, dict[str, float]]:
    models = {}
    
    target_binary = y_train.nunique() == 2
    target_categorical = y_train.nunique() > 2 and is_target_integer

    if target_binary:
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=200),
            'KNN': KNeighborsClassifier(),
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
            'XGBoostClassifier': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss') 
        }
    elif target_categorical:
        models = {
            'KNN': KNeighborsClassifier(),
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
            'XGBoostClassifier': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss') 
        }
    else:
        models = {
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
            'RandomForestRegressor': RandomForestRegressor(random_state=42),
            'XGBoostRegressor': xgb.XGBRegressor(eval_metric='rmse') 
        }

    performance = {}
    cv_performance = {}

    # Loop through each model in the 'models' dictionary
    for name, model in models.items():
        # Train the model using the training data
        model.fit(X_train, y_train)
        
        # Predict the target values for the test set
        y_pred = model.predict(X_test)
        
        # If the target is an integer (classification task)
        if is_target_integer:
            # Calculate accuracy for classification
            accuracy = accuracy_score(y_test, y_pred)
            # Store the model's accuracy in the 'performance' dictionary
            performance[name] = accuracy
            # Print the model's accuracy
            print(f'{name} Accuracy: {accuracy:.4f}')
        else:
            # If it's a regression task, calculate the Mean Squared Error (MSE)
            mse = mean_squared_error(y_test, y_pred)
            # Store the model's MSE in the 'performance' dictionary
            performance[name] = mse
            # Print the model's MSE
            print(f'{name} Mean Squared Error: {mse}')

        # If cross-validation is picked
        if perform_cross_validation:
            # Perform cross-validation on the model and get the average score
            cv_mean_score = evaluate_model_with_cross_validation(model, X_train, y_train, is_target_integer)
            # Store the cross-validation performance score in the 'cv_performance' dictionary
            cv_performance[name] = cv_mean_score

    # If cross-validation was performed
    if perform_cross_validation:
        # For classification, find the model with the highest average cross-validation score
        # For regression, find the model with the lowest cross-validation score (lower MSE is better)
        best_model_name = max(cv_performance, key=cv_performance.get) if is_target_integer else min(cv_performance, key=cv_performance.get)
        # Get the best model's cross-validation score
        best_cv_score = cv_performance[best_model_name]
        # Print the best model and its average cross-validation score
        print(f'\nBest Model from Cross-Validation: {best_model_name} with Average CV Score: {best_cv_score}')
    else:
        # If cross-validation was not performed, find the best model based on the performance metrics
        # For regression, select the model with the lowest MSE & For classification, select the model with the highest accuracy
        best_model_name = max(performance, key=performance.get) if is_target_integer else min(performance, key=performance.get)

    # Retrieve the best model from the 'models' dictionary
    best_model = models[best_model_name]
    # Print the top model's name and its performance metric (accuracy or MSE)
    print(f'\nTop Model: {best_model_name} with {"Accuracy" if is_target_integer else "MSE"}: {performance[best_model_name]}')

    # Return the best model, its name, and the performance dictionary
    return best_model, best_model_name, performance

def detect_errors(model, X_test: pd.DataFrame, y_test: pd.Series, is_target_integer: bool, original_target_dtype, target_column: str) -> None:
    y_pred = model.predict(X_test)
    
    # Adjust predictions and actual values for integer targets
    if is_target_integer:
        y_pred = np.round(y_pred).astype(int)
        y_test = y_test.astype(int)
    
    # Prepare a DataFrame to show errors
    errors_df = pd.DataFrame({
        'Index': y_test.index,
        'Actual': y_test,
        'Predicted': y_pred
    })

    # Add the 'Difference' column only for numeric targets
    if pd.api.types.is_integer_dtype(original_target_dtype) or pd.api.types.is_float_dtype(original_target_dtype):
        absolute_diff = np.abs(y_test - y_pred)
        errors_df['Difference'] = absolute_diff
        
        # Set a threshold to identify significant errors
        error_threshold = np.mean(absolute_diff) + 2 * np.std(absolute_diff)
        significant_errors_df = errors_df[absolute_diff > error_threshold]
    else:
        significant_errors_df = errors_df[y_test != y_pred]

    if significant_errors_df.empty:
        print(f"No errors found for {target_column}! Your data seems to be good!")
    else:
        print(f"Detected Significant Errors for {target_column}:")
        print(significant_errors_df)
        
        # Save errors to possible_errors_ml.csv
        significant_errors_df.to_csv('possible_errors_ml.csv', index=False)
        print(f"Errors have been saved to possible_errors_ml.csv for {target_column}")



def select_best_model_window(best_model_name: str, available_models: list[str], target_type: str) -> str:
    def on_submit():
        selected_model.set(dropdown.get())
        root.quit()
        root.destroy()

    def on_closing():
        # Handle the case when the user clicks "X" (window close)
        print("Window closed without submission.")
        selected_model.set("")  # Set to an empty string to indicate no selection
        root.quit()
        root.destroy()

    root = tk.Tk()
    selected_model = tk.StringVar(value=best_model_name)

    # Update model_options to include XGBoost models
    model_options = {
        'int': ['KNN', 'RandomForestClassifier', 'DecisionTreeClassifier', 'XGBoostClassifier'],  # Multiclass classification
        'binary': ['LogisticRegression', 'KNN', 'RandomForestClassifier', 'DecisionTreeClassifier', 'XGBoostClassifier'],  # Binary classification
        'float': ['DecisionTreeRegressor', 'RandomForestRegressor', 'XGBoostRegressor'],  # Regression for continuous targets
        'object': ['KNN', 'RandomForestClassifier', 'DecisionTreeClassifier', 'XGBoostClassifier']  # Categorical classification
    }

    # Adjust based on target_type (int, float, binary, etc.)
    fitting_models = model_options.get(target_type, available_models)

    root.title("Choose the Best Model")
    root.geometry("400x200")
    root.resizable(False, False)

    main_frame = ttk.Frame(root, padding="20 20 20 20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(main_frame, text=f"The best model right now is: {best_model_name}").grid(row=0, column=0, padx=10, pady=10)

    dropdown = ttk.Combobox(main_frame, textvariable=selected_model, values=fitting_models, state="readonly")
    dropdown.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

    submit_button = ttk.Button(main_frame, text="Submit", command=on_submit)
    submit_button.grid(row=2, column=0, padx=10, pady=20)

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=1)

    # Handle the window close (X) event
    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()

    return selected_model.get()

def main() -> None:
    # Clear the possible_errors_ml.csv and data_errors.csv files at the start
    clear_csv_file('possible_errors_ml.csv')
    clear_csv_file('data_errors.csv')  # Add this line to clear data_errors.csv

    file_path = select_file()
    df = load_dataset(file_path)
    options = ask_options()

    # Run data quality check if selected
    if options['data_quality'].get():
        data_quality_check(df, options['plot_diagrams'].get())  # Pass the option to plot diagrams
    
    # Only proceed with ML-related tasks if 'ml_error_detection' option is selected
    if options['ml_error_detection'].get():
        # Drop the 'Id' column if it exists before ML processing
        if 'Id' in df.columns:
            df.drop(columns=['Id'], inplace=True)
            print("Dropped 'Id' column before ML error detection.")

        target_column = select_target_column(df.columns)  # Only select target column when ML is needed
        
        y = encode_target(df[target_column])  # Encode target variable
        original_target_dtype = df[target_column].dtype
        X = df.drop(columns=[target_column])
        
        is_target_integer = pd.api.types.is_integer_dtype(y)
        is_target_float = pd.api.types.is_float_dtype(y)
        is_target_object = pd.api.types.is_object_dtype(y)
        is_target_boolean = pd.api.types.is_bool_dtype(y)
        
        if is_target_integer:
            target_type = 'int'
        elif is_target_float:
            target_type = 'float'
        elif is_target_object:
            target_type = 'string'
        elif is_target_boolean:
            target_type = 'binary'

        X, _ = encode_categories(X)  # Encode feature columns

        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        check_data_quality(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model, model_name, performance = train_and_evaluate_models(
            X_train,
            y_train,
            X_test,
            y_test,
            is_target_integer,
            options['cross_validation'].get()
        )

        available_models = [
            'LogisticRegression', 'KNN', 'RandomForestClassifier', 'DecisionTreeClassifier',
            'DecisionTreeRegressor', 'RandomForestRegressor',
            'XGBoostClassifier', 'XGBoostRegressor'
        ]

        if is_target_integer:
            available_models = [m for m in available_models if m in performance]

        selected_model_name = select_best_model_window(model_name, available_models, target_type)

        # Check if a model was selected
        if not selected_model_name:
            print("No model was selected. Exiting.")
            return  # Exit the main function if no model was selected

        # Optionally, re-evaluate the selected model if it was changed
        if selected_model_name != model_name:
            model_classes = {
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=200),
                'KNN': KNeighborsClassifier(),
                'RandomForestClassifier': RandomForestClassifier(random_state=42),
                'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
                'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
                'RandomForestRegressor': RandomForestRegressor(random_state=42),
                'XGBoostClassifier': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                'XGBoostRegressor': xgb.XGBRegressor(eval_metric='rmse')
            }
            model = model_classes.get(selected_model_name)
            if model is not None:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if is_target_integer:
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f'Selected model is {selected_model_name} with Accuracy: {accuracy:.4f}')
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    print(f'{selected_model_name} Mean Squared Error: {mse}')

        # If error detection is selected, run the detection logic
        if options['ml_error_detection'].get():
            detect_errors(model, X_test, y_test, is_target_integer, original_target_dtype, target_column)

if __name__ == "__main__":
    main()
