import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# 1. 数据加载与预处理
df = pd.read_csv('C:\\Users\\16969\\Desktop\\2025-07-03-10-33-32-RES.txt', sep='\t', header=None)
df.columns = ['Index', 'Workpiece_Temp', 'Upper_Die_Temp', 'Lower_Die_Temp', 
              'Forming_Speed', 'Max_Forming_Force', 'Die_Load']

# 2. 特征与目标变量
X = df[['Workpiece_Temp', 'Upper_Die_Temp', 'Lower_Die_Temp', 'Forming_Speed']]
y_force = df['Max_Forming_Force']
y_load = df['Die_Load']

# 3. 数据标准化（SVR对特征尺度敏感）
scaler_X = StandardScaler()
scaler_y_force = StandardScaler()
scaler_y_load = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_force_scaled = scaler_y_force.fit_transform(y_force.values.reshape(-1, 1)).ravel()
y_load_scaled = scaler_y_load.fit_transform(y_load.values.reshape(-1, 1)).ravel()

# 4. 划分数据集
X_train, X_test, y_force_train, y_force_test = train_test_split(
    X_scaled, y_force_scaled, test_size=0.2, random_state=42)
_, _, y_load_train, y_load_test = train_test_split(
    X_scaled, y_load_scaled, test_size=0.2, random_state=42)

# 5. 最大成形力模型（SVR）
force_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
force_model.fit(X_train, y_force_train)
force_pred_scaled = force_model.predict(X_test)
force_pred = scaler_y_force.inverse_transform(force_pred_scaled.reshape(-1, 1)).ravel()  # 反标准化

# 计算R²（需反标准化后计算）
force_r2 = r2_score(
    scaler_y_force.inverse_transform(y_force_test.reshape(-1, 1)), 
    scaler_y_force.inverse_transform(force_pred_scaled.reshape(-1, 1))
)

# 6. 模具载荷模型（SVR）
load_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
load_model.fit(X_train, y_load_train)
load_pred_scaled = load_model.predict(X_test)
load_pred = scaler_y_load.inverse_transform(load_pred_scaled.reshape(-1, 1)).ravel()  # 反标准化

# 计算R²
load_r2 = r2_score(
    scaler_y_load.inverse_transform(y_load_test.reshape(-1, 1)), 
    scaler_y_load.inverse_transform(load_pred_scaled.reshape(-1, 1))
)

# 7. 保存模型和标准化器
os.makedirs("models/SVR", exist_ok=True)
joblib.dump(force_model, 'models/SVR/forming_force_model.pkl')
joblib.dump(load_model, 'models/SVR/die_load_model.pkl')
joblib.dump(scaler_X, 'models/SVR/scaler_X.pkl')
joblib.dump(scaler_y_force, 'models/SVR/scaler_y_force.pkl')
joblib.dump(scaler_y_load, 'models/SVR/scaler_y_load.pkl')

# 8. 输出结果（反标准化后的实际值和预测值）
y_force_test_actual = scaler_y_force.inverse_transform(y_force_test.reshape(-1, 1)).ravel()
y_load_test_actual = scaler_y_load.inverse_transform(y_load_test.reshape(-1, 1)).ravel()

print("=== 最大成形力模型（SVR）===")
print(f"R²分数: {force_r2:.4f}")
print("\n实际值 vs. 预测值（前5行）:")
print(pd.DataFrame({'实际值': y_force_test_actual[:5], '预测值': force_pred[:5]}))

print("\n=== 模具载荷模型（SVR）===")
print(f"R²分数: {load_r2:.4f}")
print("\n实际值 vs. 预测值（前5行）:")
print(pd.DataFrame({'实际值': y_load_test_actual[:5], '预测值': load_pred[:5]}))
