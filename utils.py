import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
# 调用示例
# force_model, load_model, scaler_X, scaler_y_force, scaler_y_load = load_models_and_scalers("models/DNN_Optimized")
def load_models_and_scalers(model_dir):
    force_model = tf.keras.models.load_model(f"{model_dir}/forming_force_model.keras")
    load_model = tf.keras.models.load_model(f"{model_dir}/die_load_model.keras")
    scaler_X = joblib.load(f"{model_dir}/scaler_X.pkl")
    scaler_y_force = joblib.load(f"{model_dir}/scaler_y_force.pkl")
    scaler_y_load = joblib.load(f"{model_dir}/scaler_y_load.pkl")
    return force_model, load_model, scaler_X, scaler_y_force, scaler_y_load

# 调用示例
# force, load = predict_properties(population, force_model, load_model, scaler_X, scaler_y_force, scaler_y_load)
def predict_properties(input_array, force_model, load_model, scaler_X, scaler_y_force, scaler_y_load):
    
    # 给列名
    columns = scaler_X.feature_names_in_
    input_df = pd.DataFrame(input_array, columns=columns)
    
    input_scaled = scaler_X.transform(input_df)
    force_pred_scaled = force_model.predict(input_scaled, verbose=0)
    load_pred_scaled = load_model.predict(input_scaled, verbose=0)

    force_pred = scaler_y_force.inverse_transform(force_pred_scaled).ravel()
    load_pred = scaler_y_load.inverse_transform(load_pred_scaled).ravel()
    return force_pred, load_pred


# 适应度函数
def fitness_function(params_array, force_model, load_model, scaler_X, scaler_y_force, scaler_y_load, target_force=None):
    # params_array: shape (n_samples, 4)
    force, load = predict_properties(params_array, force_model, load_model, scaler_X, scaler_y_force, scaler_y_load)
    
    # 例如：我们希望最小化载荷，同时让成形力接近 target_force
    penalty = 0
    if target_force is not None:
        penalty = np.abs(force - target_force)
    
    # 这里定义适应度为“载荷 + 罚分”
    fitness = load + penalty
    return fitness



