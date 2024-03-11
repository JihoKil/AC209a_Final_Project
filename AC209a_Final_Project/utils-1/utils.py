from sklearn.model_selection import GridSearchCV
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# Performs hyperparameter tuning using GridSearchCV.
def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5, scoring='neg_mean_squared_error'):

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("Best Hyperparameters:", grid_search.best_params_)

    return grid_search, grid_search.best_params_

#  Fits a model to the training data and evaluates its performance.
def fit_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()  # Start time
    model.fit(X_train, y_train)
    end_time = time.time()  # End time

    # Calculate training time
    training_time = end_time - start_time

    # calculate r^2
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    # Generate predictions and calculate MSE
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_mse = mean_squared_error(y_train, train_preds)
    test_mse = mean_squared_error(y_test, test_preds)

    return model, train_r2, test_r2, train_mse, test_mse, training_time


# Display results as a pandas dataframe
def display_results(model_name, train_r2, test_r2, train_mse, test_mse, training_time, results_df = None):
  if results_df is None:
    results_df = pd.DataFrame(columns=['Model', 'Train_R^2', 'Test_R^2', 'Train_MSE', 'Test_MSE', 'Training Time'])

  cur_results = {
      'Model': model_name,
      'Train_R^2':  train_r2,
      'Test_R^2': test_r2,
      'Train_MSE': train_mse,
      'Test_MSE': test_mse,
      'Training Time': training_time
  }

  # Adding the results to the dataframe
  cur_results_df = pd.DataFrame.from_dict([cur_results])
  results_df = pd.concat([results_df, cur_results_df], ignore_index=True)
  return results_df




# Helper functions for plotting results

# Plots the top `num_features_to_display` feature importances
def plot_top_feature_importance(model_name, feature_importance, num_features_to_display, feature_names):
    # Create a dataframe for the feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })

    # Sort the dataframe by importance and select the top features
    top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(num_features_to_display)

    # Plot using seaborn
    plt.figure(figsize=(6, 5))
    sns.barplot(x='Importance', y='Feature', data=top_features, palette="Reds_d")
    plt.title(f'Top {num_features_to_display} Feature Importances in {model_name}')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.show()


# Plots the training and test residuals for a given model.
def plot_residuals(model, model_name, X_train, y_train, X_test, y_test):

    #  get predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Calculate residuals
    train_residuals = y_train - train_predictions
    test_residuals = y_test - test_predictions

    # Plotting
    plt.figure(figsize=(6, 5))

    # Training residuals plot
    sns.residplot(x=train_predictions, y=train_residuals, lowess=True, color="blue", line_kws={'color': 'darkblue', 'lw': 1}, label='Training Residuals')

    # Test residuals plot
    sns.residplot(x=test_predictions, y=test_residuals, lowess=True, color="red", line_kws={'color': 'darkred', 'lw': 1}, label='Test Residuals')

    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted Values - {model_name}')
    plt.legend()
    plt.show()




# Plots predicted values against actual values for the testing sets
def plot_predictions_vs_actual(model, model_name, X_test, y_test):
    predictions = model.predict(X_test)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values - {model_name}')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.show()







# The pipline that standardized numerical predictors and one-hot encoded categorical predictors
def data_preprocess(df, X_train, X_test):
  # Identify numerical and categorical features
  numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
  categorical_cols = df.select_dtypes(include=['object']).columns

  # Standardize numerical predictors
  scaler = StandardScaler()
  X_train_num = scaler.fit_transform(X_train[numerical_cols])
  X_test_num = scaler.transform(X_test[numerical_cols])

  # One-hot encode categorical predictors
  encoder = OneHotEncoder(handle_unknown='ignore', drop='first')
  encoder.fit(df[categorical_cols])
  X_train_cat = encoder.transform(X_train[categorical_cols]).toarray()
  X_test_cat = encoder.transform(X_test[categorical_cols]).toarray()

  # Combine scaled numerical data and encoded categorical data
  X_train_prep = np.hstack([X_train_num, X_train_cat])
  X_test_prep = np.hstack([X_test_num, X_test_cat])

  return X_train_prep, X_test_prep, encoder, categorical_cols, numerical_cols


# Helper function for PCA and PC reverse engineering

def apply_pca(X_train_prep, X_test_prep):
    # Apply PCA
    pca = PCA()
    pca.fit(X_train_prep)
    X_train_pca = pca.transform(X_train_prep)
    X_test_pca = pca.transform(X_test_prep)
    # Calculate the cumulative variance ratio
    cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()
    return X_train_pca, X_test_pca, cumulative_variance_ratio

def get_components(cumulative_variance_ratio, threshold, X_pca):
    n_components = (cumulative_variance_ratio < threshold).sum() + 1
    print(f'The variance of the first {n_components} Principal Components explained {threshold*100}% of Data.')
    X_pca_reduced = X_pca[:, :n_components]
    return n_components, X_pca_reduced

# Reconstruct feature names after one-hot encoding
def feature_name_reconstruct(encoder, numerical_cols, categorical_cols, categories):
    categories = encoder.categories_
    new_categorical_features = [f"{col}_{cat}" for col, cats in zip(categorical_cols, categories) for cat in cats[1:]]
    all_features = list(numerical_cols) + new_categorical_features
    return all_features

def reverse_pca(pca, all_features, reg_coefficients):
    # Extract PCA loadings
    loadings = pd.DataFrame(pca.components_[:51].T, columns=[f'PC{i+1}' for i in range(51)], index=all_features)

    # Map back to original predictors
    mapped_coefficients = loadings.dot(reg_coefficients)




# revert back to original features importance from principle components
def get_original_feature_importances(model, pca, original_features, n_components):

    # Get feature importances from the model (based on PCA components)
    try:
        importances_pca = model.feature_importances_
    except:
        importances_pca = model.coef_

    # Get PCA loadings (contributions of each original feature to each PCA component)
    loadings = pca.components_[:n_components]

    # Calculate the contribution of each original feature to the importance of each PCA component
    feature_contributions = loadings.T * importances_pca

    # Aggregate these contributions across all PCA components for each original feature
    original_feature_importances = np.sum(feature_contributions, axis=1)

    # Create a dictionary mapping original feature names to their approximated importances
    feature_importance_dict = dict(zip(original_features, original_feature_importances))

    # Sort the dictionary by importance
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    return sorted_feature_importance
