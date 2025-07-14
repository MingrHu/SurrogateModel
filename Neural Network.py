import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. 数据加载与预处理
df = pd.read_csv('C:\\Users\\16969\\Desktop\\2025-07-03-10-33-32-RES.txt', sep='\t', header=None)
df.columns = ['Index', 'Workpiece_Temp', 'Upper_Die_Temp', 'Lower_Die_Temp', 
              'Forming_Speed', 'Max_Forming_Force', 'Die_Load']

# 2. 特征与目标变量
X = df[['Workpiece_Temp', 'Upper_Die_Temp', 'Lower_Die_Temp', 'Forming_Speed']]
y_force = df['Max_Forming_Force'].values.reshape(-1, 1)
y_load = df['Die_Load'].values.reshape(-1, 1)

# 3. 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
# 在数据预处理部分添加目标变量标准化
scaler_y_force = StandardScaler()
y_force_scaled = scaler_y_force.fit_transform(y_force)

scaler_y_load = StandardScaler()
y_load_scaled = scaler_y_load.fit_transform(y_load)


# 4. 数据划分
X_train, X_temp, y_train_force, y_temp_force, y_train_load, y_temp_load = train_test_split(
    X_scaled, y_force_scaled, y_load_scaled, test_size=0.4, random_state=42
)
X_val, X_test, y_val_force, y_test_force, y_val_load, y_test_load = train_test_split(
    X_temp, y_temp_force, y_temp_load, test_size=0.5, random_state=42
)

# 5. 定义 DNN 模型构建函数
def build_dnn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),  # 第1层：输入层+隐藏层
        BatchNormalization(),                               # 第2层：批标准化
        Dropout(0.2),                                       # 第3层：Dropout
        Dense(32, activation='relu'),                       # 第4层：隐藏层
        BatchNormalization(),                               # 第5层：批标准化
        Dropout(0.1),                                       # 第6层：Dropout
        Dense(1)                                            # 第7层：输出层（线性激活）
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# 6. 训练最大成形力模型（显式使用验证集）
force_model = build_dnn_model(input_dim=X_train.shape[1])
callbacks = [
    # patience 表示15轮训练损失没有下降就停止
    EarlyStopping(monitor='val_loss', patience = 15, restore_best_weights = True),  # 早停
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)  # 学习率衰减
]
history_force = force_model.fit(
    X_train, y_train_force,
    validation_data=(X_val, y_val_force),  # 显式传入验证集
    epochs = 200,
    batch_size = 32,
    callbacks=callbacks,
    verbose=1
)

# 7. 训练模具载荷模型
load_model = build_dnn_model(input_dim=X_train.shape[1])
history_load = load_model.fit(
    X_train, y_train_load,          # 训练数据和标签
    validation_data=(X_val, y_val_load),  # 验证数据和标签
    epochs = 200,                   # 训练轮数
    batch_size = 32,                # 每个批次的样本数
    callbacks=callbacks,            # 回调函数列表
    verbose=1                       # 日志输出级别
)

# 预测时输出标准化后的值
force_pred_scaled = force_model.predict(X_test).ravel()
load_pred_scaled = load_model.predict(X_test).ravel()

# 反标准化预测值
force_pred = scaler_y_force.inverse_transform(force_pred_scaled.reshape(-1, 1)).ravel()
load_pred = scaler_y_load.inverse_transform(load_pred_scaled.reshape(-1, 1)).ravel()

# 计算R²分数（实际值和预测值均在原始尺度上）
fact_force = scaler_y_force.inverse_transform(y_test_force.reshape(-1, 1)).ravel()
fact_load = scaler_y_load.inverse_transform(y_test_load.reshape(-1, 1)).ravel()
force_r2 = r2_score(fact_force, force_pred)
load_r2 = r2_score(fact_load, load_pred)

# 9. 保存模型和标准化器
os.makedirs("models/DNN_Optimized", exist_ok=True)
force_model.save('models/DNN_Optimized/forming_force_model.keras')
load_model.save('models/DNN_Optimized/die_load_model.keras')
joblib.dump(scaler_X, 'models/DNN_Optimized/scaler_X.pkl')
joblib.dump(scaler_y_force,'models/DNN_Optimized/scaler_y_force.pkl')
joblib.dump(scaler_y_load,'models/DNN_Optimized/scaler_y_load.pkl')

# 10. 输出结果
print("=== 最大成形应力模型N ===")
print(f"R²分数: {force_r2:.4f}")
print("\n实际值 vs. 预测值（前5行）:")
print(pd.DataFrame({'实际值': fact_force[:5], '预测值': force_pred[:5]}))

print("\n=== 模具载荷模型MPa ===")
print(f"R²分数: {load_r2:.4f}")
print("\n实际值 vs. 预测值（前5行）:")
print(pd.DataFrame({'实际值': fact_load[:5], '预测值': load_pred[:5]}))
