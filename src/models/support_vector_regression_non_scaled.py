import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import Binarizer


class SVMModel:
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

    def train(self, X_train, y_train, C, epsilon):
        model = SVR(C=C, epsilon=epsilon)
        model.fit(X_train, y_train.ravel())
        return model

    def evaluate(self, model, X, y):
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        return mse, r2, mae

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
        y_binary = binarizer.transform(y.values.reshape(-1, 1))
        y_pred_binary = binarizer.transform(y_pred.reshape(-1, 1))
        cm = confusion_matrix(y_binary, y_pred_binary)
        return cm

    def calculate_metrics(self, y, y_pred, threshold):
        binarizer = Binarizer(threshold=threshold)
        y_binary = binarizer.transform(y.values.reshape(-1, 1))
        y_pred_binary = binarizer.transform(y_pred.reshape(-1, 1))
        accuracy = accuracy_score(y_binary, y_pred_binary)
        precision = precision_score(y_binary, y_pred_binary)
        recall = recall_score(y_binary, y_pred_binary)
        f1 = f1_score(y_binary, y_pred_binary)
        return accuracy, precision, recall, f1

    def run(self):

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

            # Define the parameter grid for hyperparameter tuning
            param_grid = {'kernel': ['linear', 'poly', 'rbf'],
                          'C': [0.01, 0.1, 1, 10, 100],
                          'epsilon': [0.01, 0.1, 1.0]}

            # Set the random seed for NumPy
            np.random.seed(42)

            # Create a SVR model
            model = SVR()

            # Create a GridSearchCV object with SVR and parameter grid
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error')

            # Fit the GridSearchCV object to the training data
            grid_search.fit(X_train, y_train.values.ravel())

            # Get the best hyperparameter values and the best model
            best_C = grid_search.best_params_['C']
            best_epsilon = grid_search.best_params_['epsilon']
            final_model = grid_search.best_estimator_

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

            # Unscale predictions
            y_pred_test = self.unscale_predictions(final_model.predict(X_test))

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
                'r2_val': r2_val,
                'mae_val': mae_val,
                'mse_test': mse_test,
                'r2_test': r2_test,
                'mae_test': mae_test,
                'confusion_matrix': cm,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_test': y_test.values.ravel(),
                'y_pred_test': y_pred_test,
            }
            fold_results.append(fold_result)

        # Calculate average metrics across folds
        mse_vals = [fold['mse_val'] for fold in fold_results]
        r2_vals = [fold['r2_val'] for fold in fold_results]
        mae_vals = [fold['mae_val'] for fold in fold_results]
        mse_tests = [fold['mse_test'] for fold in fold_results]
        r2_tests = [fold['r2_test'] for fold in fold_results]
        mae_tests = [fold['mae_test'] for fold in fold_results]
        accuracy_scores = [fold['accuracy'] for fold in fold_results]
        precision_scores = [fold['precision'] for fold in fold_results]
        recall_scores = [fold['recall'] for fold in fold_results]
        f1_scores = [fold['f1'] for fold in fold_results]

        avg_mse_val = np.mean(mse_vals)
        avg_r2_val = np.mean(r2_vals)
        avg_mae_val = np.mean(mae_vals)
        avg_mse_test = np.mean(mse_tests)
        avg_r2_test = np.mean(r2_tests)
        avg_mae_test = np.mean(mae_tests)
        avg_accuracy = np.mean(accuracy_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)

        print("Average Mean Squared Error (Validation):", avg_mse_val)
        print("Average R-squared (Validation):", avg_r2_val)
        print("Average Mean Absolute Error (Validation):", avg_mae_val)
        print("Average Mean Squared Error (Test):", avg_mse_test)
        print("Average R-squared (Test):", avg_r2_test)
        print("Average Mean Absolute Error (Test):", avg_mae_test)
        print("Average Accuracy:", avg_accuracy)
        print("Average Precision:", avg_precision)
        print("Average Recall:", avg_recall)
        print("Average F1-score:", avg_f1)

        # Calculate the percentage of values above 3 in the test sample
        threshold = 3
        num_values_above_threshold = (y_test > threshold).sum()
        percentage_above_threshold = (num_values_above_threshold / len(y_test)) * 100

        print("Percentage of values above 3 in the test sample:", percentage_above_threshold)

        # Calculate average confusion matrix
        cm_avg = cm_sum / num_folds

        print("Average Confusion Matrix:")
        print(cm_avg)

        # Plot predictions for each fold
        # for fold_result in fold_results:
        #     fold = fold_result['fold']
        #     y_test = fold_result['y_test']
        #     y_pred_test = fold_result['y_pred_test']
        #     self.plot_predictions(y_test, y_pred_test, f"Test Fold {fold}")

if __name__ == "__main__":
    data_path = "data/folds"
    model = SVMModel(data_path)
    model.run()
