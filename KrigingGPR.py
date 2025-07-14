import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 1. 数据加载与预处理
df = pd.read_csv('C:\\Users\\16969\\Desktop\\res.txt', sep='\t', header=None)
df.columns = ['Index', 'Workpiece_Temp', 'Die_Temp', 'Forming_Speed', 
              'STDV_grainSize', 'Die_Load']

# 2. 特征与目标变量
X = df[['Workpiece_Temp', 'Die_Temp', 'Forming_Speed']]
y_stdv = df['STDV_grainSize']
y_load = df['Die_Load']

# 3. 数据标准化（包括 y）
scaler_X = StandardScaler()
scaler_y_stdv = StandardScaler()
scaler_y_load = StandardScaler()

X_train, X_test, y_stdv_train, y_stdv_test, y_load_train, y_load_test = train_test_split(
    X, y_stdv, y_load, test_size=0.2, random_state=42
)

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_stdv_train_scaled = scaler_y_stdv.fit_transform(y_stdv_train.values.reshape(-1, 1)).ravel()
y_stdv_test_scaled = scaler_y_stdv.transform(y_stdv_test.values.reshape(-1, 1)).ravel()

y_load_train_scaled = scaler_y_load.fit_transform(y_load_train.values.reshape(-1, 1)).ravel()
y_load_test_scaled = scaler_y_load.transform(y_load_test.values.reshape(-1, 1)).ravel()

# 4. 定义改进后的 Kriging 模型
kernel = C(1.0, (1e-3, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e4))

stdv_model = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=20,
    alpha=0.1,
    random_state=42
)
stdv_model.fit(X_train_scaled, y_stdv_train_scaled)
stdv_pred_scaled, stdv_std = stdv_model.predict(X_test_scaled, return_std=True)
stdv_pred = scaler_y_stdv.inverse_transform(stdv_pred_scaled.reshape(-1, 1)).ravel()
stdv_r2 = r2_score(y_stdv_test, stdv_pred)

# 模具载荷模型
load_model = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=20,
    alpha=0.1,
    random_state=42
)
load_model.fit(X_train_scaled, y_load_train_scaled)
load_pred_scaled, load_std = load_model.predict(X_test_scaled, return_std=True)
load_pred = scaler_y_load.inverse_transform(load_pred_scaled.reshape(-1, 1)).ravel()
load_r2 = r2_score(y_load_test, load_pred)

# 5. 保存模型
os.makedirs("models/Kriging", exist_ok=True)  
joblib.dump(stdv_model, 'models/Kriging/stdv_model.pkl')
joblib.dump(load_model, 'models/Kriging/die_load_model.pkl')

# 6. 输出结果
print("=== 晶粒尺寸标准差===")
print(f"R²分数: {stdv_r2:.4f}")
print("\n实际值 vs. 预测值（前5行）:")
print(pd.DataFrame({'实际值': y_stdv_test.head(5), '预测值': stdv_pred[:5]}))

print("\n=== 模具载荷模型N ===")
print(f"R²分数: {load_r2:.4f}")
print("\n实际值 vs. 预测值（前5行）:")
print(pd.DataFrame({'实际值': y_load_test.head(5), '预测值': load_pred[:5]}))