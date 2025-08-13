# my_krml_25176165/models/performance.py

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def print_regressor_scores(y_preds, y_actuals, set_name="Set"):
    """ Print RMSE and MAE for regression models. """
    rmse = np.sqrt(mean_squared_error(y_actuals, y_preds))
    mae = mean_absolute_error(y_actuals, y_preds)
    print(f"{set_name} RMSE: {rmse:.2f}")
    print(f"{set_name} MAE: {mae:.2f}")
