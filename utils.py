import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
# 调用示例
# stdv_model, load_model, scaler_X, scaler_y_stdv, scaler_y_load = load_models_and_scalers("models/DNN_Optimized")
def load_models_and_scalers(model_dir):
    # 加载多输出模型
    multi_output_model = tf.keras.models.load_model(f"{model_dir}/multi_output_model.keras")
    scaler_X = joblib.load(f"{model_dir}/scaler_X.pkl")
    scaler_y_stdv = joblib.load(f"{model_dir}/scaler_y_stdv.pkl")
    scaler_y_load = joblib.load(f"{model_dir}/scaler_y_load.pkl")
    return multi_output_model, scaler_X, scaler_y_stdv, scaler_y_load

# 调用示例
# stdv, load = predict_properties(population, stdv_model, load_model, scaler_X, scaler_y_stdv, scaler_y_load)
def predict_properties(input_array, multi_output_model, scaler_X, scaler_y_stdv, scaler_y_load):
    # 给列名
    columns = scaler_X.feature_names_in_
    input_df = pd.DataFrame(input_array, columns=columns)
    
    input_scaled = scaler_X.transform(input_df)
    # 多输出模型预测返回两个数组
    predictions = multi_output_model.predict(input_scaled, verbose=0)
    stdv_pred_scaled, load_pred_scaled = predictions  # 分别获取两个输出
    
    stdv_pred = scaler_y_stdv.inverse_transform(stdv_pred_scaled).ravel()
    load_pred = scaler_y_load.inverse_transform(load_pred_scaled).ravel()
    return stdv_pred, load_pred


# # 适应度函数
# def fitness_function(params_array, stdv_model, load_model, scaler_X, scaler_y_stdv, scaler_y_load, target_stdv=None):
#     # params_array: shape (n_samples, 4)
#     stdv, load = predict_properties(params_array, stdv_model, load_model, scaler_X, scaler_y_stdv, scaler_y_load)
    
#     # 例如：我们希望最小化载荷，同时让成形力接近 target_stdv
#     penalty = 0
#     if target_stdv is not None:
#         penalty = np.abs(stdv - target_stdv)
    
#     # 这里定义适应度为“载荷 + 罚分”
#     fitness = load + penalty
#     return fitness



