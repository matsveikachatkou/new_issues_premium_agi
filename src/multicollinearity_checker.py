import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


class MulticollinearityChecker:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path

    def load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        panel = pd.read_csv(os.path.join(parent_dir, self.data_path, "panel.csv"))
        return panel

    def compute_correlation_matrix(self, data):
        correlation_matrix = data.corr()
        return correlation_matrix

    def visualize_correlation_heatmap(self, correlation_matrix, output_file):
        plt.figure(figsize=(12, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()

    def check_multicollinearity(self):
        panel = self.load_data()

        # Drop the target variable and irrelevant columns in Panel 1
        features_panel = panel.drop(["NIP", "date", "MOVE", "zSpreadTrailing", "rating", "issueSize", "maturityTerm",
                                       "coupon", "initialSpreadToSwap"], axis=1)

        # Compute the correlation matrix for panel1
        correlation_matrix_panel = self.compute_correlation_matrix(features_panel)

        # Save the correlation heatmap for panel1 as an image file
        output_file_panel = os.path.join(self.output_path, "correlation_heatmap.png")
        self.visualize_correlation_heatmap(correlation_matrix_panel, output_file_panel)


checker = MulticollinearityChecker(data_path="data/processed", output_path="results")
checker.check_multicollinearity()
