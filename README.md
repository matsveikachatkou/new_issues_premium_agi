# New Issues Premium in the European Corporate Bond Market: From Empirical Evidence to Forecasting

Welcome to the README for the "New Issues Premium in the European Corporate Bond Market: From Empirical Evidence to Forecasting" project. This project is a part of a master's thesis focused on analyzing and forecasting the new issues premium in the European corporate bond market.

The new issues premium is a critical metric that reflects the pricing dynamics of newly issued corporate bonds compared to their secondary market counterparts. Through empirical analysis and forecasting techniques, this project aims to provide valuable insights into the behavior of the new issues premium and its potential predictability.

## Features

- In-depth empirical analysis of the European corporate bond market
- Utilization of forecasting models to predict new issues premium behavior
- Data visualization to illustrate pricing dynamics

## Project Structure

The project is organized as follows:

- `data/`: This directory contains the raw and processed data used in the analysis.
  - `raw/`: Raw data.
    - `new_issues.csv`: New issues data (AllianzGI Trading Desk/Refinitive).
    - `new_issues_characteristics.csv`: New issues characteristics (Refinitive).
    - `new_issues_prices_askyield_refinitive.csv`: New issues ask YTM (Refinitive).
    - `new_issues_prices_bidyield_refinitive.csv`: New issues bid YTM (Refinitive).
    - `new_issues_initial_price_bb.csv`: New issues initial YTM (Bloomberg).
    - `new_issues_initial_price_refinitive.csv`: New issues initial YTM (Refinitive).
    - `comparable_bonds_peers_duration_iboxx_*.csv`: Comparable bonds data (iBoxx).
    - `comparable_bonds_same_iboxx_*.csv`: Comparable bonds (only same issuer) data (iBoxx).
    - `comparable_bonds_hourly_prices.csv`: Comparable bonds tick YTM (Refinitive).
    - `comparable_bonds_filtered_yield.csv`: Subset of comparable bonds bid YTM (Refinitive).
    - `swap_rates.csv`: Swap rates (Refinitive).
    - `iboxx_indices.csv`: Indices (iBoxx).
    - `move_index.csv`: MOVE index (Refinitive).
  - `preprocessed/`: Engineered features.
    - `nip_peers_settlement.csv`: Calculated NIP.
    - `spread_to_peers.csv`: Calculated Initial spread discount.
    - `swap_rates.csv`: Swap rates (Refinitive).
    - `iboxx_overall.csv`: iBoxx € Overall ex-Subordinated (iBoxx).
    - `new_issues.csv`: New issues data (AllianzGI Trading Desk/Refinitive).
    - `new_issues_characteristics.csv`: New issues characteristics (Refinitive).
    - `move_index.csv`: MOVE index (Refinitive).
    - `feature_engineering.ipynb`: Notebook detailing the feature engineering steps.
      - `new_issues_preprocessed.csv`: New issues' NIP and features dataset.
      - `move.csv`: MOVE index divided by 100 (for regression).
      - `moveTrailing30.csv`: Trailing 30-day MOVE index.
      - `moveRetsTrailing30.csv`: Trailing 30-day MOVE returns.
      - `zSpread.csv`: Z-spread.
      - `zSpreadTrailing30.csv`: Trailing 30-day Z-spread.
  - `processed/`: Panel.csv is a combined dataset of NIP values and drivers.
  - `folds/`: 5-fold cross-validation datasets.
- `notebooks/`: Jupyter notebooks related to the analysis process.
  - `nip_analysis_settlement_comparable_peers_ms.ipynb`: NIP analysis, comparable bonds approach. Source for `nip_peers_settlement.csv`.
  - `nip_analysis_initial_spread_to_peers.ipynb`: Analysis of Initial spread discount. Source for `spread_to_peers.csv`.
  - `nip_analysis_prev_comparable_peers_ms.ipynb`: NIP analysis, spread discount approach.
  - `nip_analysis_settlement_index_ms_interpolated..ipynb`: NIP analysis, index-adjusted approach.
  - `nip_analysis_convergence_peers.ipynb`: Spread convergence analysis.
  - `nip_price_movement.ipynb`: Pre- and post-issue price movement study case.
  - `data_analysis.ipynb.ipynb`: New issues and comparable bonds data analysis.
- `src/`: Python scripts for regression and ML models.
  - `data_preprocessing.py`: Creates panel.csv in `data/processed/`.  
  - `sample_splitting.py`: Creates 5-folds cross-validation from panel.csv in `data/folds/`.
  - `multicollinearity_checker.py`: Creates correlation heatmap, saves it in `src/results/`. 
  - `linear_regression_panel_analysis.py`: Simple OLS and Huber Loss regression of the whole dataset, saves results in `src/results/`.
  - `linear_regression_panel_analysis_rating.py`: Simple OLS and Huber Loss regression, rating subgroups, saves results in `src/results/`.
  - `robust_linear_regression_panel_analysis.py`: Robust clustered regression of the whole dataset, saves results in `src/results/`.
  - `results/`: Results of linear regressions and multicollinearity check.
  - `models/`: ML models, 5-fold cross-validation. Results of the ML models printed out in console.
    - `linear_regression.py`: OLS regression, scaled data.
    - `linear_regression_non_scaled.py`: OLS regression, non-scaled data. Saves features importance figure in `src/models/`.
    - `lasso_regression.py`: Lasso regression, scaled data.
    - `ridge_regression.py`: Ridge regression, scaled data.
    - `support_vector_regression.py`: SVR, scaled data.
    - `support_vector_regression_non_scaled.py`: SVR, non-scaled data.
    - `random_forest.py`: Random Forest model, scaled data.
    - `gbrt.py`: GBRT model, scaled data.
    - `xgbm.py`: XGB model, scaled data.
    - `cat_boost.py`: CAT model, scaled data.
    - `cat_boost_non_scaled.py`: CAT model, non-scaled data. Saves features importance figure in `src/models/`.
    - `model_comparison.py`: Diebold-Mariano test of the models. Saves results as dm_test_results.txt and p_values.txt in `src/models/`.
- `requirements.txt`: Requirments for python environment.
- `README.md`: The document you're currently reading.

## Reproduce results

To reproduce the analysis and findings of this thesis project, follow these steps:

1. Clone the repository:
   ```bash
   $ git clone https://github.com/matsveikachatkou/new_issues_premium_agi.git
   $ cd new_issues_premium_agi
   $ pip install -r requirements.txt
2. For NIP analysis, explore Jupyter notebooks in notebooks/.
3. For feature selection analysis, explore `feature_engineering.ipynb` in `data/preprocessed`.
4. For multicollinearity analysis, run `multicollinearity_checker.py`. Results stored in `src/results/`.
5. For regression analysis, run `linear_regression_panel_analysis.py`, `linear_regression_panel_analysis_rating.py`, `robust_linear_regression_panel_analysis.py` in `src/`. Results stored in `src/results/`. To run regression for different rating subgroup, change panel_filtered = panel[panel['rating'].between(8, 16)]. The assigned rating values can be found in `data_preprocessing.py`.
6. 
