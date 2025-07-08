import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. 数据加载与预处理
df = pd.read_csv('C:\\Users\\Hmr\\Desktop\\2025-07-03-10-33-32-RES.txt', sep='\t', header=None)
df.columns = ['Index', 'Workpiece_Temp', 'Upper_Die_Temp', 'Lower_Die_Temp', 
              'Forming_Speed', 'Max_Forming_Force', 'Die_Load']

# 2. 随机森林对数据特征尺度不敏感
X = df[['Workpiece_Temp', 'Upper_Die_Temp', 'Lower_Die_Temp', 'Forming_Speed']]
y_force = df['Max_Forming_Force']
y_load = df['Die_Load']

# 3. 划分数据集（确保X_train和X_test完全一致）
X_train, X_test, y_force_train, y_force_test, y_load_train, y_load_test = train_test_split(
    X, y_force, y_load, test_size=0.2, random_state=42
)

# 4. 最大成形力模型（随机森林）
force_model = RandomForestRegressor(n_estimators=300, random_state=42)
force_model.fit(X_train, y_force_train)
force_pred = force_model.predict(X_test)  # 直接预测，无需反标准化

# 计算R²
force_r2 = r2_score(y_force_test, force_pred)

# 5. 模具载荷模型（随机森林）
load_model = RandomForestRegressor(n_estimators=100, random_state=42)
load_model.fit(X_train, y_load_train)
load_pred = load_model.predict(X_test)  # 直接预测，无需反标准化

# 计算R²
load_r2 = r2_score(y_load_test, load_pred)

# 6. 保存模型（不保存标准化器）
os.makedirs("models/RandomForest_NoScale", exist_ok=True)
joblib.dump(force_model, 'models/RandomForest_NoScale/forming_force_model.pkl')
joblib.dump(load_model, 'models/RandomForest_NoScale/die_load_model.pkl')

# 7. 输出结果
print("=== 最大成形力MPa ===")
print(f"R²分数: {force_r2:.4f}")
print("\n实际值 vs. 预测值（前5行）:")
print(pd.DataFrame({'实际值': y_force_test.head(5), '预测值': force_pred[:5]}))

print("\n=== 模具载荷N ===")
print(f"R²分数: {load_r2:.4f}")
print("\n实际值 vs. 预测值（前5行）:")
print(pd.DataFrame({'实际值': y_load_test.head(5), '预测值': load_pred[:5]}))
