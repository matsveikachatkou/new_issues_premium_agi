import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold


class GBRTModel:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self, fold):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(script_dir))
        panel1_dir = os.path.join(parent_dir, self.data_path, "panel1")
        features_to_drop = ["initialSpreadToSwap", "coupon", "maturityTerm", "issuerType", "issueSize", "zSpread", "zSpreadTrailing", "MOVE"]
        X_train = pd.read_csv(os.path.join(panel1_dir, f"fold{fold}", "X_train.csv"))
        X_val = pd.read_csv(os.path.join(panel1_dir, f"fold{fold}", "X_val.csv"))
        X_test = pd.read_csv(os.path.join(panel1_dir, "test", "X_test.csv"))
        X_train = X_train.drop(features_to_drop, axis=1)
        X_val = X_val.drop(features_to_drop, axis=1)
        X_test = X_test.drop(features_to_drop, axis=1)
        y_train = pd.read_csv(os.path.join(panel1_dir, f"fold{fold}", "y_train.csv"))
        y_val = pd.read_csv(os.path.join(panel1_dir, f"fold{fold}", "y_val.csv"))
        y_test = pd.read_csv(os.path.join(panel1_dir, "test", "y_test.csv"))

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )

    def train(self, X_train, y_train, learning_rate, n_estimators, max_depth):
        model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train.values.ravel())
        return model

    def evaluate(self, model, X, y):
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        return mse, r2, mae

    def calculate_percentage(self, y, y_pred, threshold):
        total_count = len(y)
        correct_count = np.sum(y_pred >= threshold)
        percentage = (correct_count / total_count) * 100
        return percentage

    def plot_predictions(self, y, y_pred, dataset_name):
        plt.scatter(y, y_pred, alpha=0.5)
        plt.xlabel("Realized NIP")
        plt.ylabel("Predicted NIP")
        plt.title(f"Predicted vs. Realized NIP ({dataset_name})")
        plt.show()

    def run(self):
        panel1_dir = os.path.join(self.data_path, "panel1")

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
            ) = self.load_data(fold)

            # Define the parameter grid for hyperparameter tuning
            param_grid = {
                'learning_rate': [0.01, 0.1, 1.0],
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7]
            }

            # Create a GBRT model
            model = GradientBoostingRegressor(random_state=42)

            # Create a GridSearchCV object with GBRT model and parameter grid
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error')

            # Fit the GridSearchCV object to the training data
            grid_search.fit(X_train, y_train.values.ravel())

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
            final_model.fit(X_train, y_train.values.ravel())

            # Evaluate on the validation set
            mse_val, r2_val, mae_val = self.evaluate(final_model, X_val, y_val)
            print("Mean Squared Error (Validation):", mse_val)
            print("R-squared (Validation):", r2_val)
            print("Mean Absolute Error (Validation):", mae_val)

            # Evaluate on the test set
            mse_test, r2_test, mae_test = self.evaluate(final_model, X_test, y_test)
            print("Mean Squared Error (Test):", mse_test)
            print("R-squared (Test):", r2_test)
            print("Mean Absolute Error (Test):", mae_test)

            # Calculate percentage of NIP >= 3 predicted correctly
            percentage_nip_3 = self.calculate_percentage(y_test.values.ravel(), final_model.predict(X_test), threshold=4.7)
            print("Percentage of NIP >= 3 Correctly Predicted:", percentage_nip_3)

            # Calculate percentage of NIP >= 10 predicted correctly
            percentage_nip_10 = self.calculate_percentage(y_test.values.ravel(), final_model.predict(X_test), threshold=10)
            print("Percentage of NIP >= 10 Correctly Predicted:", percentage_nip_10)

            # Store fold results
            fold_result = {
                'fold': fold,
                'best_learning_rate': best_learning_rate,
                'best_n_estimators': best_n_estimators,
                'best_max_depth': best_max_depth,
                'mse_val': mse_val,
                'r2_val': r2_val,
                'mae_val': mae_val,
                'mse_test': mse_test,
                'r2_test': r2_test,
                'mae_test': mae_test,
                'percentage_nip_3': percentage_nip_3,
                'percentage_nip_10': percentage_nip_10,
                'y_test': y_test.values.ravel(),
                'y_pred_test': final_model.predict(X_test),
            }
            fold_results.append(fold_result)

        # Calculate average metrics across folds
        mse_vals = [fold['mse_val'] for fold in fold_results]
        r2_vals = [fold['r2_val'] for fold in fold_results]
        mae_vals = [fold['mae_val'] for fold in fold_results]
        mse_tests = [fold['mse_test'] for fold in fold_results]
        r2_tests = [fold['r2_test'] for fold in fold_results]
        mae_tests = [fold['mae_test'] for fold in fold_results]

        avg_mse_val = np.mean(mse_vals)
        avg_r2_val = np.mean(r2_vals)
        avg_mae_val = np.mean(mae_vals)
        avg_mse_test = np.mean(mse_tests)
        avg_r2_test = np.mean(r2_tests)
        avg_mae_test = np.mean(mae_tests)

        print("Average Mean Squared Error (Validation):", avg_mse_val)
        print("Average R-squared (Validation):", avg_r2_val)
        print("Average Mean Absolute Error (Validation):", avg_mae_val)
        print("Average Mean Squared Error (Test):", avg_mse_test)
        print("Average R-squared (Test):", avg_r2_test)
        print("Average Mean Absolute Error (Test):", avg_mae_test)

        # Plot predictions for each fold
        for fold_result in fold_results:
            fold = fold_result['fold']
            y_test = fold_result['y_test']
            y_pred_test = fold_result['y_pred_test']
            self.plot_predictions(y_test, y_pred_test, f"Test Fold {fold}")


if __name__ == "__main__":
    data_path = "data/folds"
    model = GBRTModel(data_path)
    model.run()
