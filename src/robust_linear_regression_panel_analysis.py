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

    def run_robust_regression(self, panel, output_file, cluster_var):
        X = panel.drop(["NIP", "date", "month", "initialSpreadToSwap", "coupon", "maturityTerm", "issuerType", "issueSize",
                        "zSpread", "zSpreadTrailing", "MOVE", "rating", "issueSizeLog"], axis=1)
        y = panel["NIP"]
        X = sm.add_constant(X)

        # Create the OLS model
        model = sm.OLS(y, X)

        # Calculate the cluster-robust standard errors
        results = model.fit(cov_type='cluster', cov_kwds={'groups': panel[cluster_var]})

        # Save model summary as text file
        with open(output_file, 'w') as fh:
            fh.write(results.summary().as_text())

    def run_analysis(self):
        panel = self.load_data()

        # Run robust regression for panel2
        self.run_robust_regression(
            panel,
            os.path.join(self.output_path, "robust_regression_results.txt"),
            cluster_var='month'
        )


analysis = LinearRegressionPanelAnalysis(data_path="data/processed", output_path="results")
analysis.run_analysis()
