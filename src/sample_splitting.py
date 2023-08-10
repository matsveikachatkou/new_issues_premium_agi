import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold


class DataSampleSplitting:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(self.script_dir)

    def load_data(self):
        panel = pd.read_csv(os.path.join(self.parent_dir, self.data_path, "panel.csv"))
        return panel

    def split_data(self, data, output_dir):
        X = data.drop(["date", "NIP"], axis=1)
        y = data["NIP"]

        # Split data into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Further split train and validation sets using cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(kf.split(X_train_val))

        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            X_train, X_val = X_train_val.iloc[train_indices], X_train_val.iloc[val_indices]
            y_train, y_val = y_train_val.iloc[train_indices], y_train_val.iloc[val_indices]

            output_dir_fold = os.path.join(self.parent_dir, output_dir, f"fold{fold_idx + 1}")
            os.makedirs(output_dir_fold, exist_ok=True)

            pd.DataFrame(X_train).to_csv(os.path.join(output_dir_fold, "X_train.csv"), index=False)
            pd.DataFrame(X_val).to_csv(os.path.join(output_dir_fold, "X_val.csv"), index=False)
            pd.DataFrame(y_train, columns=["NIP"]).to_csv(os.path.join(output_dir_fold, "y_train.csv"), index=False)
            pd.DataFrame(y_val, columns=["NIP"]).to_csv(os.path.join(output_dir_fold, "y_val.csv"), index=False)

        # Save test set
        output_dir_test = os.path.join(self.parent_dir, output_dir, "test")
        os.makedirs(output_dir_test, exist_ok=True)
        pd.DataFrame(X_test).to_csv(os.path.join(output_dir_test, "X_test.csv"), index=False)
        pd.DataFrame(y_test, columns=["NIP"]).to_csv(os.path.join(output_dir_test, "y_test.csv"), index=False)

    def prepare_data(self):
        panel = self.load_data()

        # Create output directories for panel1 and panel2
        output_dir_panel = os.path.join(self.parent_dir, self.output_path, "panel")
        os.makedirs(output_dir_panel, exist_ok=True)
        os.makedirs(output_dir_panel, exist_ok=True)

        # Split panel1 data into folds and create test set
        self.split_data(panel, output_dir_panel)


data_prep = DataSampleSplitting(data_path="data/processed", output_path="data/folds")
data_prep.prepare_data()
