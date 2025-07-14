import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 数据加载与预处理
df = pd.read_csv('C:\\Users\\16969\\Desktop\\res.txt', sep='\t', header=None)
df.columns = ['Index', 'Workpiece_Temp', 'Die_Temp', 'Forming_Speed', 
              'STDV_grainSize', 'Die_Load']

# 2. 特征与目标变量
X = df[['Workpiece_Temp', 'Die_Temp', 'Forming_Speed']]
y_stdv = df['STDV_grainSize']
y_load = df['Die_Load']

# 3. 划分数据集（80%训练，20%测试）
X_train, X_test, y_stdv_train, y_stdv_test, y_load_train, y_load_test = train_test_split(
    X, y_stdv, y_load, test_size=0.2, random_state=42
)

# 4. 晶粒尺寸标准差模型
stdv_model = LinearRegression()
stdv_model.fit(X_train, y_stdv_train)
stdv_pred = stdv_model.predict(X_test)
stdv_r2 = r2_score(y_stdv_test, stdv_pred)

# 5. 模具载荷模型
load_model = LinearRegression()
load_model.fit(X_train, y_load_train)
load_pred = load_model.predict(X_test)
load_r2 = r2_score(y_load_test, load_pred)

# 6. 保存模型
os.makedirs("models/LRG", exist_ok=True)  
joblib.dump(stdv_model, 'models/LRG/stdv_model.pkl')
joblib.dump(load_model, 'models/LRG/die_load_model.pkl')

# 7. 输出结果
print("=== 晶粒尺寸标准差===")
print(f"R²分数: {stdv_r2:.4f}")
print("\n实际值 vs. 预测值（前5行）:")
print(pd.DataFrame({'实际值': y_stdv_test.head(5), '预测值': stdv_pred[:5]}))

print("\n=== 模具载荷模型N ===")
print(f"R²分数: {load_r2:.4f}")
print("\n实际值 vs. 预测值（前5行）:")
print(pd.DataFrame({'实际值': y_load_test.head(5), '预测值': load_pred[:5]}))