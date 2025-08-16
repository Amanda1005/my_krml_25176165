from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def print_regressor_scores(y_preds, y_actuals, set_name="Set"):
    """Print RMSE and MAE for regression predictions."""
    rmse = np.sqrt(mean_squared_error(y_actuals, y_preds))
    mae = mean_absolute_error(y_actuals, y_preds)
    print(f"RMSE {set_name}: {rmse}")
    print(f"MAE {set_name}: {mae}")
