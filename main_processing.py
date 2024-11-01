import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer
import xgboost as xgb 
from sklearn.svm import SVR
import tkinter as tk
from tkinter import filedialog, ttk
import warnings
from data_quality_check import data_quality_check
from encode import encode_categories  # Import custom my encoding functions

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
    root.quit()    # Exit the mainloop()
    root.destroy()  # Destroy the window and end the program
    os._exit(0)    # Ensure the program terminates completely
    
#Loading Data
def load_dataset(file_path: str) -> pd.DataFrame:
    print("Loading the dataset...")
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    return df

def center_window(root, width, height):
    # Calculate position to center the window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - height / 2)
    position_right = int(screen_width / 2 - width / 2)
    root.geometry(f'{width}x{height}+{position_right}+{position_top}')

def select_target_column(columns: list[str]) -> str:
    def on_submit():
        selected_column.set(dropdown.get())
        root.quit()
        root.destroy()

    root = tk.Tk()
    selected_column = tk.StringVar()

    root.title("Choose Target Column")
    center_window(root, 500, 300)  # Set larger, centered window
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

    def update_plot_diagrams_visibility():
        if options["data_quality"].get():
            plot_diagrams_checkbox.grid(row=1, column=1, padx=10, pady=5)
        else:
            options["plot_diagrams"].set(False)
            plot_diagrams_checkbox.grid_forget()

    def update_cross_validation_visibility():
        if options["ml_error_detection"].get():
            cross_validation_checkbox.grid(row=3, column=1, padx=10, pady=5)
        else:
            options["cross_validation"].set(False)
            cross_validation_checkbox.grid_forget()

    def on_submit():
        root.quit()
        root.destroy()

    root.title("Choose Options")
    center_window(root, 500, 300)  # Set larger, centered window

    options_frame = ttk.Frame(root, padding="10 10 10 10")
    options_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    tk.Label(options_frame, text="Select options for analysis:").grid(row=0, column=0, columnspan=2, pady=10)

    data_quality_checkbox = tk.Checkbutton(options_frame, text="Data Quality Check", variable=options["data_quality"], command=update_plot_diagrams_visibility)
    data_quality_checkbox.grid(row=1, column=0, sticky='w', padx=10)

    plot_diagrams_checkbox = tk.Checkbutton(options_frame, text="Plot Diagrams", variable=options["plot_diagrams"])

    ml_error_detection_checkbox = tk.Checkbutton(options_frame, text="Machine Learning Error Detection", variable=options["ml_error_detection"], command=update_cross_validation_visibility)
    ml_error_detection_checkbox.grid(row=3, column=0, sticky='w', padx=10)

    cross_validation_checkbox = tk.Checkbutton(options_frame, text="Cross-Validation", variable=options["cross_validation"])

    submit_button = ttk.Button(options_frame, text="Submit", command=on_submit)
    submit_button.grid(row=4, column=0, columnspan=2, pady=10)

    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
    root.mainloop()
    return options

def select_best_model_window(best_model_name: str, available_models: list[str], target_type: str, best_cv_model: str, perform_cross_validation: bool) -> str:
    def on_submit():
        selected_model.set(dropdown.get())
        root.quit()
        root.destroy()

    def on_closing():
        print("Window closed without submission.")
        selected_model.set("")
        root.quit()
        root.destroy()

    root = tk.Tk()
    selected_model = tk.StringVar(value=best_model_name)

    model_options = {
        'int': ['KNN', 'RandomForestClassifier', 'DecisionTreeClassifier', 'XGBoostClassifier'],
        'binary': ['LogisticRegression', 'KNN', 'RandomForestClassifier', 'DecisionTreeClassifier', 'XGBoostClassifier'],
        'float': ['DecisionTreeRegressor', 'RandomForestRegressor', 'XGBoostRegressor', 'GradientBoostingRegressor', 'SVR'],
        'object': ['KNN', 'RandomForestClassifier', 'DecisionTreeClassifier', 'XGBoostClassifier']
    }

    fitting_models = model_options.get(target_type, available_models)

    root.title("Choose the Best Model")
    center_window(root, 500, 300)  # Set larger, centered window

    main_frame = ttk.Frame(root, padding="20 20 20 20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Display best model message depending on cross-validation selection
    if perform_cross_validation:
        ttk.Label(main_frame, text=f"Best model from cross-validation: {best_cv_model}").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    else:
        ttk.Label(main_frame, text=f"Best model: {best_model_name}").grid(row=0, column=0, padx=10, pady=5, sticky="w")

    dropdown = ttk.Combobox(main_frame, textvariable=selected_model, values=fitting_models, state="readonly")
    dropdown.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

    submit_button = ttk.Button(main_frame, text="Submit", command=on_submit)
    submit_button.grid(row=2, column=0, padx=10, pady=20)

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=1)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

    return selected_model.get()

def ask_options() -> dict[str, tk.BooleanVar]:
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
    options = {
        "data_quality": tk.BooleanVar(),
        "plot_diagrams": tk.BooleanVar(),
        "ml_error_detection": tk.BooleanVar(),
        "cross_validation": tk.BooleanVar()
    }

    def update_plot_diagrams_visibility():
        if options["data_quality"].get():
            plot_diagrams_checkbox.grid(row=1, column=1, padx=10, pady=5)
        else:
            options["plot_diagrams"].set(False)
            plot_diagrams_checkbox.grid_forget()

    def update_cross_validation_visibility():
        if options["ml_error_detection"].get():
            cross_validation_checkbox.grid(row=3, column=1, padx=10, pady=5)
        else:
            options["cross_validation"].set(False)
            cross_validation_checkbox.grid_forget()

    def on_submit():
        root.quit()
        root.destroy()

    root.title("Choose Options")
    center_window(root, 500, 300)  # Centered and resized window

    options_frame = ttk.Frame(root, padding="10 10 10 10")
    options_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    tk.Label(options_frame, text="Select options for analysis:").grid(row=0, column=0, columnspan=2, pady=10)

    data_quality_checkbox = tk.Checkbutton(options_frame, text="Data Quality Check", variable=options["data_quality"], command=update_plot_diagrams_visibility)
    data_quality_checkbox.grid(row=1, column=0, sticky='w', padx=10)

    plot_diagrams_checkbox = tk.Checkbutton(options_frame, text="Plot Diagrams", variable=options["plot_diagrams"])

    ml_error_detection_checkbox = tk.Checkbutton(options_frame, text="Machine Learning Error Detection", variable=options["ml_error_detection"], command=update_cross_validation_visibility)
    ml_error_detection_checkbox.grid(row=3, column=0, sticky='w', padx=10)

    cross_validation_checkbox = tk.Checkbutton(options_frame, text="Cross-Validation", variable=options["cross_validation"])

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




# Define parameter grids for models
param_grids = {
    'LogisticRegression': {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    'RandomForestRegressor': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10]
    },
    'XGBoostClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'XGBoostRegressor': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    },
   
    'DecisionTreeClassifier': {
        'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
    },
    'DecisionTreeRegressor': {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10]
    }
    }


#Train_and_evaluate_models function also applies cross-validation and grid search only when selected
def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_type: str,
    perform_cross_validation: bool = False
) -> tuple[object, str, dict[str, float]]:

    binary_models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=200),
        'KNN': KNeighborsClassifier(),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'XGBoostClassifier': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    multiclass_models = {
        'KNN': KNeighborsClassifier(),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'XGBoostClassifier': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }
    
    regression_models = {
        'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
        'RandomForestRegressor': RandomForestRegressor(random_state=42),
        'XGBoostRegressor': xgb.XGBRegressor(eval_metric='rmse'),
        'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
        'SVR': SVR()
    }

    selected_models = (
        binary_models if target_type == 'binary' 
        else multiclass_models if target_type == 'multiclass' 
        else regression_models
    )
    
    performance = {}
    cv_performance = {}
    best_fitted_model = None
    best_model_name = None
    best_score = -np.inf if target_type in ['binary', 'multiclass'] else np.inf

    for name, model in selected_models.items():
        if perform_cross_validation and name in param_grids:
            print(f"\nPerforming cross-validation and Grid Search for {name}...")
            grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name],
                                       scoring='accuracy' if target_type in ['binary', 'multiclass'] else 'neg_mean_squared_error', 
                                       cv=5)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"{name} - Grid Search Best Parameters: {grid_search.best_params_}")
            
            scoring_metric = 'accuracy' if target_type in ['binary', 'multiclass'] else 'neg_mean_squared_error'
            kf = KFold(n_splits=get_dynamic_splits(len(X_train)), shuffle=True, random_state=42)
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=kf, scoring=scoring_metric)
            cv_mean_score = np.mean(cv_scores)
            cv_performance[name] = cv_mean_score
            print(f"{name} Cross-Validation Scores: {cv_scores}")
            print(f"{name} Cross-Validation Average Score: {cv_mean_score:.4f}")

            if (target_type in ['binary', 'multiclass'] and cv_mean_score > best_score) or (target_type == 'regression' and cv_mean_score < best_score):
                best_score = cv_mean_score
                best_fitted_model = best_model
                best_model_name = name

        elif not perform_cross_validation:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if target_type in ['binary', 'multiclass']:
                accuracy = accuracy_score(y_test, y_pred)
                performance[name] = accuracy
                print(f"{name} Accuracy: {accuracy:.4f}")
                score = accuracy
            else:
                mse = mean_squared_error(y_test, y_pred)
                performance[name] = mse
                print(f"{name} Mean Squared Error: {mse:.4f}")
                score = mse

            if (target_type in ['binary', 'multiclass'] and score > best_score) or (target_type == 'regression' and score < best_score):
                best_score = score
                best_fitted_model = model
                best_model_name = name

    if perform_cross_validation:
        print(f'\nBest Model from Cross-Validation: {best_model_name} with Average Score: {best_score:.4f}')
    else:
        print(f'\nBest Model: {best_model_name} with {"Accuracy" if target_type in ["binary", "multiclass"] else "MSE"}: {best_score:.4f}')

    return best_fitted_model, best_model_name, cv_performance if perform_cross_validation else performance

def detect_errors(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    is_target_integer: bool,
    original_target_dtype,
    target_column: str,
    original_df: pd.DataFrame
) -> None:
    # Predict the target values for the entire dataset
    y_pred = model.predict(X)

    # Adjust predictions for integer targets by rounding
    if is_target_integer:
        y_pred = np.round(y_pred).astype(int)
        y = y.astype(int)

    # Prepare a DataFrame to show actual and predicted values
    errors_df = pd.DataFrame({
        'Actual': y.values,
        'Predicted': y_pred
    })

    # Add a 'Difference' column for numeric targets to calculate the absolute differences
    if pd.api.types.is_integer_dtype(original_target_dtype) or pd.api.types.is_float_dtype(original_target_dtype):
        absolute_diff = np.abs(errors_df['Actual'] - errors_df['Predicted'])
        errors_df['Difference'] = absolute_diff

        # Calculate the error threshold for both float and integer targets
        error_threshold = np.mean(absolute_diff) + 2 * np.std(absolute_diff)
        significant_errors_df = errors_df[absolute_diff > error_threshold]
    else:
        # For categorical targets, use exact mismatches
        significant_errors_df = errors_df[errors_df['Actual'] != errors_df['Predicted']]

    # Check if any significant errors were found
    if significant_errors_df.empty:
        print(f"No significant errors found for {target_column}! Your data seems to be good!")
    else:
        # Get the indexes of significant errors
        error_indexes = significant_errors_df.index
        print(f"Indexes of significant errors for {target_column}: {error_indexes.tolist()}")

        # Merge the significant errors with the original data for easier identification
        significant_errors_df = significant_errors_df.merge(original_df, left_index=True, right_index=True, how='left')

        # Add the adjusted index as the first column, with a +2 offset
        significant_errors_df.insert(0, 'Index', significant_errors_df.index + 2)

        print(f"Detected Significant Errors for {target_column}:")
        print(significant_errors_df)

        # Save errors to 'Detected Errors/possible_errors_ml.csv' including the adjusted index column
        significant_errors_df.to_csv('Detected Errors/possible_errors_ml.csv', index=False)
        print(f"Errors have been saved to 'Detected Errors/possible_errors_ml.csv' for {target_column}")


def main() -> None:
    # Clear the Detected Errors/possible_errors_ml.csv and Detected Errors/quality_errors.csv files at the start
    clear_csv_file('Detected Errors/possible_errors_ml.csv')
    clear_csv_file('Detected Errors/quality_errors.csv')

    file_path = select_file()
    df = load_dataset(file_path)
    options = ask_options()

    if 'Id' in df.columns:
            df.drop(columns=['Id'], inplace=True)
    
    if options['data_quality'].get():
        data_quality_check(df, options['plot_diagrams'].get())  
    
    # Only proceed with ML-related tasks if 'ml_error_detection' option is selected
    if options['ml_error_detection'].get():
        target_column = select_target_column(df.columns)
        
        # Combine features and target for encoding
        combined_df = encode_categories(df, target_column=target_column)

        # Separate the encoded target column and feature columns
        y = combined_df[target_column]
        X = combined_df.drop(columns=[target_column])

        # Determine target type with integer threshold condition
        if pd.api.types.is_integer_dtype(df[target_column]) or pd.api.types.is_float_dtype(df[target_column]):
            target_type = 'regression'
        elif pd.api.types.is_object_dtype(df[target_column]) or pd.api.types.is_bool_dtype(df[target_column]):
            target_type = 'binary' if df[target_column].nunique() == 2 else 'multiclass'

        # Define model options based on target type
        model_options = {
            'regression': ['DecisionTreeRegressor', 'RandomForestRegressor', 'XGBoostRegressor', 'GradientBoostingRegressor', 'SVR'],
            'binary': ['LogisticRegression', 'KNN', 'RandomForestClassifier', 'DecisionTreeClassifier', 'XGBoostClassifier'],
            'multiclass': [ 'KNN', 'RandomForestClassifier', 'DecisionTreeClassifier', 'XGBoostClassifier']
        }

        fitting_models = model_options.get(target_type, [])

        # Impute and scale the feature data
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split data into training and testing sets for initial evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and evaluate models with the determined target type
        model, model_name, performance = train_and_evaluate_models(
            X_train,
            y_train,
            X_test,
            y_test,
            target_type,  # Pass the determined target type
            options['cross_validation'].get()
        )

        best_cv_model = model_name

        selected_model_name = select_best_model_window(
            model_name, fitting_models, target_type, best_cv_model, options['cross_validation'].get()
        )

        if not selected_model_name:
            print("No model was selected. Exiting.")
            return  

        # Run error detection on the entire dataset using the trained model
        print(f"Using the trained model: {selected_model_name}")
        
        if options['ml_error_detection'].get():
            detect_errors(model, X, y, pd.api.types.is_integer_dtype(y), df[target_column].dtype, target_column, df)


if __name__ == "__main__":
    main()
