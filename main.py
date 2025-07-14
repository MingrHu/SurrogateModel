import numpy as np
import pymoo
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX     # 模拟二进制交叉（SBX）
from pymoo.operators.mutation.pm import PM        # 多项式变异（PM）
from pymoo.optimize import minimize
from GAproblem import FormingProcessOptimizationProblem
from pymoo.visualization.scatter import Scatter
from utils import predict_properties,load_models_and_scalers

# 1️ 加载模型和scaler
model_dir = "models/DNN_Optimized"
force_model, load_model, scaler_X, scaler_y_force, scaler_y_load = load_models_and_scalers(model_dir)

# 2️ 定义参数范围
param_bounds = [
    (875, 965),  # 工件温度
    (300, 700),   # 上下模温度
    (10, 50)       # 上模速度
]

# 3️ 实例化优化问题
target_force = 1000  # 希望成形力接近1000
problem = FormingProcessOptimizationProblem(
    force_model, load_model, scaler_X, scaler_y_force, scaler_y_load,
    param_bounds=param_bounds
)

# 4️ 选择算法
algorithm = NSGA2(
    pop_size = 100,
    eliminate_duplicates=True,  # 是否去除重复个体
    n_offsprings=100,           # 每代产生的后代数量（默认=pop_size）
    crossover=SBX(prob=0.9),    # 交叉算子（模拟二进制交叉，概率0.9）
    mutation=PM(prob=0.1),      # 变异算子（多项式变异，概率0.1）
)


# 5️ 开始优化
result = minimize(
    problem,
    algorithm,
    ("n_gen", 100),  # 迭代代数
    seed = 42,      # 随机种子
    verbose = True  # 日志
)

# 6. 分析帕累托前沿
pareto_front = result.F  # 所有非支配解的目标值 [force, load]
design_space = result.X  # 对应的决策变量

print("\n=== 帕累托最优解示例 ===")
for i in range(min(5, len(pareto_front))):  # 打印前5个解
    print(f"解 {i+1}: 参数={design_space[i]}, 目标值=[成形力={pareto_front[i,0]:.2f}, 负载={pareto_front[i,1]:.2f}]")

# 7. 可视化帕累托前沿
plot = Scatter(title="Pareto Front", labels=["Force", "Load"])
plot.add(pareto_front, color="red")
plot.show()

# 单目标优化##############################
# # 6️ 输出结果
# best_params = result.X
# pred_force, pred_load = predict_properties(
#     best_params.reshape(1, -1),
#     force_model, load_model,
#     scaler_X, scaler_y_force, scaler_y_load
# )

# print("\n=== 最优结果 ===")
# print(f"工艺参数: {best_params}")
# print(f"预测最大成形力: {pred_force[0]:.2f}")
# print(f"预测模具载荷: {pred_load[0]:.2f}")
##########################################
# 打印部分帕累托最优解

