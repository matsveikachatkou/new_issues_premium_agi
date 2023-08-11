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
  
    - `nip_peers_settlement.csv`: Calculated NIP.
    - `move_index.csv`: Calculated initial spread discount.
  
- `notebooks/`: Jupyter notebooks related to the analysis and forecasting process.
  - `analyze.ipynb`: Notebook detailing the empirical analysis steps.
  - `forecast.ipynb`: Notebook explaining forecasting techniques.
- `scripts/`: Python scripts used to perform specific tasks.
  - `analyze.py`: Script for running the empirical analysis.
  - `utils.py`: Utility functions used across the project.
- `results/`: Output directory for storing generated figures and analysis results.
- `LICENSE`: The license information for this project.
- `README.md`: The document you're currently reading.

Feel free to explore and adapt the project structure as needed.
