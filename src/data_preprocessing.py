import pandas as pd
import os
import shutil


class DataPreprocessing:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path

    def load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        df = pd.read_csv(os.path.join(parent_dir, self.data_path, "new_issues_preprocessed.csv"))
        zspread = pd.read_csv(os.path.join(parent_dir, self.data_path, "zSpread.csv"))
        zspread['zSpread'] = zspread['zSpread'].shift(1)
        zspread_trailing30 = pd.read_csv(os.path.join(parent_dir, self.data_path, "zSpreadTrailing30.csv"))
        zspread_trailing30['zSpreadTrailing'] = zspread_trailing30['zSpreadTrailing'].shift(1)
        move = pd.read_csv(os.path.join(parent_dir, self.data_path, "MOVE.csv"))
        move['MOVE'] = move['MOVE'].shift(1)
        move_trailing30 = pd.read_csv(os.path.join(parent_dir, self.data_path, "MOVETrailing30.csv"))
        move_trailing30['MOVETrailing'] = move_trailing30['MOVETrailing'].shift(1)
        rets_trailing30 = pd.read_csv(os.path.join(parent_dir, self.data_path, "retsTrailing30.csv"))
        rets_trailing30['iboxxRetsTrailing'] = rets_trailing30['iboxxRetsTrailing'].shift(1)
        move_rets_trailing30 = pd.read_csv(os.path.join(parent_dir, self.data_path, "moveRetsTrailing30.csv"))
        move_rets_trailing30['MOVERetsTrailing'] = move_rets_trailing30['MOVERetsTrailing'].shift(1)

        return df, zspread, zspread_trailing30, move, move_trailing30, rets_trailing30, move_rets_trailing30

    def merge_datasets(self, df, zspread, zspread_trailing30, move, move_trailing30, rets_trailing30,
                       move_rets_trailing30):
        merged_zspread = zspread.merge(zspread_trailing30, on="date")
        merged_move = move.merge(move_trailing30, on="date").merge(move_rets_trailing30, on="date")
        merged_rets = rets_trailing30
        panel = df.merge(merged_zspread, on="date").merge(merged_move, on="date").merge(merged_rets, on="date")
        return panel

    def encode_issuer_type(self, df):
        df['issuerType'] = df['issuerType'].apply(lambda x: 1 if x == 'Corporate' else 0)
        return df

    def assign_rating_values(self, df):
        rating_mapping = {'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4, 'A+': 5, 'A': 6,
                          'A-': 7, 'BBB+': 8, 'BBB': 9, 'BBB-': 10, 'BB+': 11, 'BB': 12,
                          'BB-': 13, 'B+': 14, 'B': 15, 'B-': 16, 'NR': None}
        max_rating = max([value for value in rating_mapping.values() if value is not None])
        rating_mapping['NR'] = (1 + max_rating) / 2

        df['rating'] = df['rating'].map(rating_mapping)
        return df

    def assign_payment_rank_values(self, df):
        payment_rank_mapping = {'Secured': 1, 'Sr Preferred': 2, 'Sr Non Preferred': 3, 'Sr Unsecured': 4}
        df['paymentRank'] = df['paymentRank'].map(payment_rank_mapping)
        return df

    def save_data(self, panel):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, self.output_path)
        os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
        panel.to_csv(os.path.join(output_dir, "panel.csv"), index=False)

    def process_data(self):
        df, zspread, zspread_trailing30, move, move_trailing30, rets_trailing30, move_rets_trailing30 = self.load_data()

        # Perform modifications on the original DataFrame
        df = self.encode_issuer_type(df)
        df = self.assign_rating_values(df)
        df = self.assign_payment_rank_values(df)

        # Drop unnecessary columns
        df = df.drop(['isin', 'ipt', 'guidance'], axis=1)

        # Create a new 'month' column based on the 'date' column
        df['month'] = pd.to_datetime(df['date']).dt.month

        # Convert columns to numeric (except date and month)
        numeric_columns = df.columns.drop(['date', 'month'])
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Merge datasets and create panels
        panel = self.merge_datasets(df, zspread, zspread_trailing30, move, move_trailing30, rets_trailing30, move_rets_trailing30)

        # Save the processed panels to CSV files
        self.save_data(panel)


dp = DataPreprocessing(data_path="data/preprocessed", output_path="data/processed")
dp.process_data()
