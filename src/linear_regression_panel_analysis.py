import pandas as pd
import numpy as np
import statsmodels.api as sm
import os


class LinearRegressionPanelAnalysis:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path

    def load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        panel = pd.read_csv(os.path.join(parent_dir, self.data_path, "panel.csv"))
        return panel

    def run_simple_regression(self, panel, output_file):
        X = panel.drop(["NIP", "date", "month", "initialSpreadToSwap", "coupon", "maturityTerm", "issuerType", "issueSize",
                        "zSpread", "zSpreadTrailing", "MOVE", "rating", "issueSizeLog"], axis=1)
        y = panel["NIP"]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        # Save model summary as text file
        with open(output_file, 'w') as fh:
            fh.write(model.summary().as_text())

    def huber_loss(self, y_true, y_pred, epsilon):
        residual = y_true - y_pred
        condition = np.abs(residual) <= epsilon
        loss = np.where(condition, 0.5 * residual**2, epsilon * (np.abs(residual) - 0.5 * epsilon))
        return loss

    def run_huber_regression(self, panel, output_file, epsilon):
        X = panel.drop(["NIP", "date", "month", "initialSpreadToSwap", "coupon", "maturityTerm", "issuerType", "issueSize",
                        "zSpread", "zSpreadTrailing", "MOVE", "rating", "issueSizeLog"], axis=1)
        y = panel["NIP"]
        X = sm.add_constant(X)

        # Create the regression model with Huber loss
        model = sm.OLS(y, X)
        results = model.fit(cov_type='HC3', loss=self.huber_loss, epsilon=epsilon)

        # Save model summary as text file
        with open(output_file, 'w') as fh:
            fh.write(results.summary().as_text())

    def run_analysis(self):
        panel = self.load_data()

        # Run simple regression for panel
        self.run_simple_regression(
            panel,
            os.path.join(self.output_path, "simple_regression_results.txt")
        )

        # Run Huber regression for panel
        self.run_huber_regression(
            panel,
            os.path.join(self.output_path, "huber_regression_results.txt"), epsilon=1.35
        )


analysis = LinearRegressionPanelAnalysis(data_path="data/processed", output_path="results")
analysis.run_analysis()
