from dieboldmariano import dm_test
from linear_regression import LinearRegressionModel
from ridge_regression import RidgeRegressionModel
from lasso_regression import LassoRegressionModel
from support_vector_regression import SVRModel
from random_forest import RandomForestModel
from gbrt import GBRTModel
from xgbm import XGBModel
from cat_boost import CatBoostModel
import numpy as np

def calculate_dm_statistics(realised_values, predictions1, predictions2):
    dm_statistics = []
    for fold in range(len(realised_values)):
        realized_values_float = np.array(realised_values[fold]).astype(float)
        dm_statistic, _ = dm_test(realized_values_float, predictions1[fold], predictions2[fold], one_sided=True)
        dm_statistics.append(dm_statistic)
    mean_dm_statistic = np.mean(dm_statistics)
    return mean_dm_statistic

def calculate_p_values(realised_values, predictions1, predictions2):
    p_values = []
    for fold in range(len(realised_values)):
        realized_values_float = np.array(realised_values[fold]).astype(float)
        _, p_value = dm_test(realized_values_float, predictions1[fold], predictions2[fold], one_sided=True)
        p_values.append(p_value)
    mean_p_value = np.mean(p_values)
    return mean_p_value

if __name__ == "__main__":
    data_path = "data/folds"

    # Instantiate and run all the models
    linear_model = LinearRegressionModel(data_path)
    linear_model.run()
    linear_predictions = linear_model.predictions

    ridge_model = RidgeRegressionModel(data_path)
    ridge_model.run()
    ridge_predictions = ridge_model.predictions

    lasso_model = LassoRegressionModel(data_path)
    lasso_model.run()
    lasso_predictions = lasso_model.predictions

    svr_model = SVRModel(data_path)
    svr_model.run()
    svr_predictions = svr_model.predictions

    rf_model = RandomForestModel(data_path)
    rf_model.run()
    rf_predictions = rf_model.predictions

    gbrt_model = GBRTModel(data_path)
    gbrt_model.run()
    gbrt_predictions = gbrt_model.predictions

    xgbm_model = XGBModel(data_path)
    xgbm_model.run()
    xgbm_predictions = xgbm_model.predictions

    catboost_model = CatBoostModel(data_path)
    catboost_model.run()
    catboost_predictions = catboost_model.predictions

    # Get the realized values for each fold
    linear_realised_values = linear_model.realised_values
    ridge_realised_values = ridge_model.realised_values
    lasso_realised_values = lasso_model.realised_values
    svr_realised_values = svr_model.realised_values
    rf_realised_values = rf_model.realised_values
    gbrt_realised_values = gbrt_model.realised_values
    xgbm_realised_values = xgbm_model.realised_values
    catboost_realised_values = catboost_model.realised_values

    # Calculate Diebold-Mariano test statistics and p-values for each pair of models
    model_names = ['LinearRegressionModel', 'RidgeRegressionModel', 'LassoRegressionModel', 'SVRModel',
                   'RandomForestModel', 'GBRTModel', 'XGBModel', 'CatBoost']

    dm_table_data = []
    pval_table_data = []

    for i, model1 in enumerate([linear_predictions, ridge_predictions, lasso_predictions,
                                svr_predictions, rf_predictions, gbrt_predictions,
                                xgbm_predictions, catboost_predictions]):
        dm_row = []
        pval_row = []
        for j, model2 in enumerate([linear_predictions, ridge_predictions, lasso_predictions,
                                    svr_predictions, rf_predictions, gbrt_predictions,
                                    xgbm_predictions, catboost_predictions]):
            if i == j:
                dm_row.append(np.nan)
                pval_row.append(np.nan)
            else:
                dm_statistic = calculate_dm_statistics([linear_realised_values, ridge_realised_values, lasso_realised_values,
                                                        svr_realised_values, rf_realised_values, gbrt_realised_values,
                                                        xgbm_realised_values, catboost_realised_values][i], model1, model2)
                p_value = calculate_p_values([linear_realised_values, ridge_realised_values, lasso_realised_values,
                                              svr_realised_values, rf_realised_values, gbrt_realised_values,
                                              xgbm_realised_values, catboost_realised_values][i], model1, model2)
                dm_row.append(dm_statistic)
                pval_row.append(p_value)

        dm_table_data.append(dm_row)
        pval_table_data.append(pval_row)

    # Write the tables to the text files
    with open('dm_test_results.txt', 'w') as f:
        f.write('\t' + '\t'.join(model_names) + '\n')
        for i, model_name in enumerate(model_names):
            f.write(model_name + '\t' + '\t'.join([str(val) for val in dm_table_data[i]]) + '\n')

    with open('p_values.txt', 'w') as f:
        f.write('\t' + '\t'.join(model_names) + '\n')
        for i, model_name in enumerate(model_names):
            f.write(model_name + '\t' + '\t'.join([str(val) for val in pval_table_data[i]]) + '\n')
