import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.preprocessing import Binarizer


class LinearRegressionModel:
    def __init__(self, data_path):
        self.data_path = data_path

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
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def evaluate(self, model, X, y):
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, mae, r2

    def unscale_predictions(self, y_pred):
        return y_pred

    def plot_predictions(self, y, y_pred, dataset_name):
        plt.scatter(y, y_pred, alpha=0.5)
        plt.xlabel("Realized NIP")
        plt.ylabel("Predicted NIP")
        plt.title(f"Predicted vs. Realized NIP ({dataset_name})")
        plt.show()

    def calculate_confusion_matrix(self, y, y_pred, threshold):
        binarizer = Binarizer(threshold=threshold)
        y_binary = binarizer.transform(y)
        y_pred_binary = binarizer.transform(y_pred)
        cm = confusion_matrix(y_binary, y_pred_binary)
        return cm

    def calculate_metrics(self, y, y_pred, threshold):
        binarizer = Binarizer(threshold=threshold)
        y_binary = binarizer.transform(y)
        y_pred_binary = binarizer.transform(y_pred)
        accuracy = accuracy_score(y_binary, y_pred_binary)
        precision = precision_score(y_binary, y_pred_binary)
        recall = recall_score(y_binary, y_pred_binary)
        f1 = f1_score(y_binary, y_pred_binary)
        return accuracy, precision, recall, f1

    def plot_average_feature_importance(self, model, X_train):
        feature_importance = np.abs(model.coef_)
        feature_names = X_train.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance.flatten()})
        feature_importance_df['Relative Importance'] = feature_importance_df['Importance'] / feature_importance_df['Importance'].sum()

        # Sort features by relative importance in descending order
        feature_importance_df.sort_values(by='Relative Importance', inplace=True)

        plt.figure(figsize=(8, 4))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Relative Importance'], color='lightgrey')
        plt.xlabel('Relative Importance')
        plt.ylabel('Features')
        plt.title('OLS')
        plt.tight_layout()
        plt.savefig('average_realtive_feature_importance_linear_regression.png', facecolor='w', dpi=300, bbox_inches='tight')
        plt.show()

        # Print feature importances with values
        print("Average Feature Importance:")
        print(feature_importance_df[['Feature', 'Relative Importance']].sort_values(by='Relative Importance', ascending=False))

    def run(self):
        panel1_dir = os.path.join(self.data_path, "panel1")

        # Perform cross-validation
        num_folds = 5
        fold_results = []
        cm_sum = np.zeros((2, 2))
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for fold in range(1, num_folds + 1):
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_data(fold)

            # Create a Linear Regression model
            model = LinearRegression()

            # Train the model
            model.fit(X_train, y_train)

            # Evaluate on the validation set
            mse_val, mae_val, r2_val = self.evaluate(model, X_val, y_val)
            print("Mean Squared Error (Validation):", mse_val)
            print("Mean Absolute Error (Validation):", mae_val)
            print("R-squared (Validation):", r2_val)

            # Evaluate on the test set
            mse_test, mae_test, r2_test = self.evaluate(model, X_test, y_test)
            print("Mean Squared Error (Test):", mse_test)
            print("Mean Absolute Error (Test):", mae_test)
            print("R-squared (Test):", r2_test)

            # Unscale predictions
            y_pred_test = model.predict(X_test)

            # Calculate confusion matrix
            threshold = 3
            cm = self.calculate_confusion_matrix(y_test, y_pred_test, threshold)
            cm_sum += cm

            # Calculate metrics
            accuracy, precision, recall, f1 = self.calculate_metrics(y_test, y_pred_test, threshold)
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            # Store fold results
            fold_result = {
                'fold': fold,
                'mse_val': mse_val,
                'mae_val': mae_val,
                'r2_val': r2_val,
                'mse_test': mse_test,
                'mae_test': mae_test,
                'r2_test': r2_test,
                'y_test': y_test,
                'y_pred_test': y_pred_test,
                'final_model': model,  # Added to store the final model for each fold
            }
            fold_results.append(fold_result)

        # Calculate average metrics across folds
        mse_vals = [fold['mse_val'] for fold in fold_results]
        mae_vals = [fold['mae_val'] for fold in fold_results]
        r2_vals = [fold['r2_val'] for fold in fold_results]
        mse_tests = [fold['mse_test'] for fold in fold_results]
        mae_tests = [fold['mae_test'] for fold in fold_results]
        r2_tests = [fold['r2_test'] for fold in fold_results]

        avg_mse_val = np.mean(mse_vals)
        avg_mae_val = np.mean(mae_vals)
        avg_r2_val = np.mean(r2_vals)
        avg_mse_test = np.mean(mse_tests)
        avg_mae_test = np.mean(mae_tests)
        avg_r2_test = np.mean(r2_tests)

        print("Average Mean Squared Error (Validation):", avg_mse_val)
        print("Average Mean Absolute Error (Validation):", avg_mae_val)
        print("Average R-squared (Validation):", avg_r2_val)
        print("Average Mean Squared Error (Test):", avg_mse_test)
        print("Average Mean Absolute Error (Test):", avg_mae_test)
        print("Average R-squared (Test):", avg_r2_test)

        # Calculate average confusion matrix
        cm_avg = cm_sum / num_folds

        print("Average Confusion Matrix:")
        print(cm_avg)

        # Calculate average metrics
        avg_accuracy = np.mean(accuracy_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)

        print("Average Accuracy:", avg_accuracy)
        print("Average Precision:", avg_precision)
        print("Average Recall:", avg_recall)
        print("Average F1-score:", avg_f1)

        # Calculate the percentage of values above 3 in the test sample
        threshold = 3
        num_values_above_threshold = (y_test > threshold).sum()
        percentage_above_threshold = (num_values_above_threshold / len(y_test)) * 100

        print("Percentage of values above 3 in the test sample:", percentage_above_threshold)

        # Plot and save the average feature importance as a picture
        self.plot_average_feature_importance(model, X_train)


if __name__ == "__main__":
    data_path = "data/folds"
    model = LinearRegressionModel(data_path)
    model.run()
