import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import ttest_1samp


class XGBModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.predictions = []
        self.realised_values = []

    def load_data(self, fold):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(script_dir))
        panel_dir = os.path.join(parent_dir, self.data_path, "panel")
        features_to_drop = ["month", "initialSpreadToSwap", "coupon", "maturityTerm", "issuerType", "issueSize",
                            "zSpread", "zSpreadTrailing", "MOVE", "rating", "issueSizeLog"]
        X_train = pd.read_csv(os.path.join(panel_dir, f"fold{fold}", "X_train.csv"))
        X_val = pd.read_csv(os.path.join(panel_dir, f"fold{fold}", "X_val.csv"))
        X_test = pd.read_csv(os.path.join(panel_dir, "test", "X_test.csv"))
        X_train = X_train.drop(features_to_drop, axis=1)
        X_val = X_val.drop(features_to_drop, axis=1)
        X_test = X_test.drop(features_to_drop, axis=1)
        y_train = pd.read_csv(os.path.join(panel_dir, f"fold{fold}", "y_train.csv"))
        y_val = pd.read_csv(os.path.join(panel_dir, f"fold{fold}", "y_val.csv"))
        y_test = pd.read_csv(os.path.join(panel_dir, "test", "y_test.csv"))

        # Scale the input features
        feature_columns_to_scale = [col for col in X_train.columns if col not in ['issuerType', 'paymentRank', 'rating']]
        X_scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[feature_columns_to_scale] = X_scaler.fit_transform(X_train_scaled[feature_columns_to_scale])
        X_val_scaled[feature_columns_to_scale] = X_scaler.transform(X_val_scaled[feature_columns_to_scale])
        X_test_scaled[feature_columns_to_scale] = X_scaler.transform(X_test_scaled[feature_columns_to_scale])

        # Scale the target variable
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)
        y_test_scaled = y_scaler.transform(y_test)

        return (
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            y_train_scaled,
            y_val_scaled,
            y_test_scaled,
            y_train,
            y_val,
            y_test,
            y_scaler,
        )

    def train(self, X_train, y_train, learning_rate, n_estimators, max_depth):
        model = XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model

    def evaluate(self, model, X, y):
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, r2

    def unscale_predictions(self, y_pred_scaled, y_scaler):
        y_pred_scaled_2d = y_pred_scaled.reshape(-1, 1)
        y_pred_unscaled = y_scaler.inverse_transform(y_pred_scaled_2d)
        return y_pred_unscaled

    def plot_predictions(self, y, y_pred, dataset_name):
        plt.scatter(y, y_pred, alpha=0.5)
        plt.xlabel("Realized NIP")
        plt.ylabel("Predicted NIP")
        plt.title(f"Predicted vs. Realized NIP ({dataset_name})")
        plt.show()

    def run(self):

        # Perform cross-validation
        num_folds = 5
        fold_results = []

        for fold in range(1, num_folds + 1):
            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                y_train_unscaled,
                y_val_unscaled,
                y_test_unscaled,
                y_scaler,
            ) = self.load_data(fold)

            # Define the parameter grid for hyperparameter tuning
            param_grid = {
                'learning_rate': [0.01, 0.1, 1.0],
                'n_estimators': [50, 100, 200],
                'max_depth': [1, 3, 5]
            }

            # Create a GBRT model
            model = XGBRegressor(objective='reg:pseudohubererror', huber_slope=0.7, random_state=42)

            # Create a GridSearchCV object with XGBRegressor and parameter grid
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error')

            # Fit the GridSearchCV object to the training data
            grid_search.fit(X_train, y_train.ravel())

            # Get the best hyperparameter values and the best model
            best_learning_rate = grid_search.best_params_['learning_rate']
            best_n_estimators = grid_search.best_params_['n_estimators']
            best_max_depth = grid_search.best_params_['max_depth']
            final_model = grid_search.best_estimator_

            print(f"Fold: {fold}")
            print("Best Learning Rate:", best_learning_rate)
            print("Best n_estimators:", best_n_estimators)
            print("Best max_depth:", best_max_depth)

            # Train the final model with the best hyperparameter values
            final_model.fit(X_train, y_train.ravel())

            # Evaluate on the validation set
            mse_val, r2_val = self.evaluate(final_model, X_val, y_val)
            print("Mean Squared Error (Validation):", mse_val)
            print("R-squared (Validation):", r2_val)

            # Evaluate on the test set
            mse_test, r2_test = self.evaluate(final_model, X_test, y_test)
            print("Mean Squared Error (Test):", mse_test)
            print("R-squared (Test):", r2_test)

            # Unscale predictions
            y_pred_test_scaled = final_model.predict(X_test)
            y_pred_test_unscaled = self.unscale_predictions(y_pred_test_scaled, y_scaler)

            # Store the predictions for the current fold
            self.predictions.append(y_pred_test_unscaled)
            self.realised_values.append(y_test_unscaled)

            # Store fold results
            fold_result = {
                'fold': fold,
                'best_learning_rate': best_learning_rate,
                'best_n_estimators': best_n_estimators,
                'best_max_depth': best_max_depth,
                'mse_val': mse_val,
                'r2_val': r2_val,
                'mse_test': mse_test,
                'r2_test': r2_test,
                'y_test': y_test_unscaled,
                'y_pred_test': y_pred_test_unscaled,
            }
            fold_results.append(fold_result)

        # Calculate average metrics across folds
        mse_vals = [fold['mse_val'] for fold in fold_results]
        r2_vals = [fold['r2_val'] for fold in fold_results]
        mse_tests = [fold['mse_test'] for fold in fold_results]
        r2_tests = [fold['r2_test'] for fold in fold_results]

        avg_mse_val = np.mean(mse_vals)
        avg_r2_val = np.mean(r2_vals)
        avg_mse_test = np.mean(mse_tests)
        avg_r2_test = np.mean(r2_tests)

        print("Average Mean Squared Error (Validation):", avg_mse_val)
        print("Average R-squared (Validation):", avg_r2_val)
        print("Average Mean Squared Error (Test):", avg_mse_test)
        print("Average R-squared (Test):", avg_r2_test)

        # Perform the one-sample t-test with a theoretical baseline value
        baseline_r2 = 0.306471954379447
        t_stat, p_value = ttest_1samp(r2_tests, baseline_r2)

        # Print the p-value
        print("P-value:", p_value)

        # Plot predictions for each fold
        # for fold_result in fold_results:
        #     fold = fold_result['fold']
        #     y_test = fold_result['y_test']
        #     y_pred_test = fold_result['y_pred_test']
        #     self.plot_predictions(y_test, y_pred_test, f"Test Fold {fold}")


if __name__ == "__main__":
    data_path = "data/folds"
    model = XGBModel(data_path)
    model.run()
