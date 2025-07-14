import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# 1. 数据加载与预处理
df = pd.read_csv('C:\\Users\\16969\\Desktop\\res.txt', sep='\t', header=None)
df.columns = ['Index', 'Workpiece_Temp', 'Die_Temp', 'Forming_Speed', 
              'STDV_grainSize', 'Die_Load']

# 2. 特征与目标变量
X = df[['Workpiece_Temp', 'Die_Temp', 'Forming_Speed']]
y_stdv = df['STDV_grainSize']
y_load = df['Die_Load']

# 3. 划分数据集（确保训练集和测试集的划分一致）
X_train, X_test, y_stdv_train, y_stdv_test, y_load_train, y_load_test = train_test_split(
    X, y_stdv, y_load, test_size=0.2, random_state=42
)

# 4. 数据标准化（仅对训练集拟合，避免数据泄露）
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)  # 注意：测试集只用transform，不用fit_transform

scaler_y_stdv = StandardScaler()
y_stdv_train_scaled = scaler_y_stdv.fit_transform(y_stdv_train.values.reshape(-1, 1)).ravel()
y_stdv_test_scaled = scaler_y_stdv.transform(y_stdv_test.values.reshape(-1, 1)).ravel()

scaler_y_load = StandardScaler()
y_load_train_scaled = scaler_y_load.fit_transform(y_load_train.values.reshape(-1, 1)).ravel()
y_load_test_scaled = scaler_y_load.transform(y_load_test.values.reshape(-1, 1)).ravel()

# 5. 晶粒尺寸标准差模型
stdv_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
stdv_model.fit(X_train_scaled, y_stdv_train_scaled)
stdv_pred_scaled = stdv_model.predict(X_test_scaled)
stdv_pred = scaler_y_stdv.inverse_transform(stdv_pred_scaled.reshape(-1, 1)).ravel()  # 反标准化

# 计算R²（在反标准化后的尺度上计算）
stdv_r2 = r2_score(
    scaler_y_stdv.inverse_transform(y_stdv_test_scaled.reshape(-1, 1)), 
    scaler_y_stdv.inverse_transform(stdv_pred_scaled.reshape(-1, 1))
)

# 6. 模具载荷模型
load_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
load_model.fit(X_train_scaled, y_load_train_scaled)
load_pred_scaled = load_model.predict(X_test_scaled)
load_pred = scaler_y_load.inverse_transform(load_pred_scaled.reshape(-1, 1)).ravel()  # 反标准化

# 计算R²
load_r2 = r2_score(
    scaler_y_load.inverse_transform(y_load_test_scaled.reshape(-1, 1)), 
    scaler_y_load.inverse_transform(load_pred_scaled.reshape(-1, 1))
)

# 7. 保存模型和标准化器
os.makedirs("models/SVR", exist_ok=True)
joblib.dump(stdv_model, 'models/SVR/stdv_model.pkl')
joblib.dump(load_model, 'models/SVR/die_load_model.pkl')
joblib.dump(scaler_X, 'models/SVR/scaler_X.pkl')
joblib.dump(scaler_y_stdv, 'models/SVR/scaler_y_stdv.pkl')
joblib.dump(scaler_y_load, 'models/SVR/scaler_y_load.pkl')

# 8. 输出结果（反标准化后的实际值和预测值）
y_stdv_test_actual = scaler_y_stdv.inverse_transform(y_stdv_test_scaled.reshape(-1, 1)).ravel()
y_load_test_actual = scaler_y_load.inverse_transform(y_load_test_scaled.reshape(-1, 1)).ravel()

print("=== 晶粒尺寸标准差===")
print(f"R²分数: {stdv_r2:.4f}")
print("\n实际值 vs. 预测值（前5行）:")
print(pd.DataFrame({'实际值': y_stdv_test_actual[:5], '预测值': stdv_pred[:5]}))

print("\n=== 模具载荷N ===")
print(f"R²分数: {load_r2:.4f}")
print("\n实际值 vs. 预测值（前5行）:")
print(pd.DataFrame({'实际值': y_load_test_actual[:5], '预测值': load_pred[:5]}))