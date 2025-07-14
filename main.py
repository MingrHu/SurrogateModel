import numpy as np
import pymoo
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from GAproblem import FormingProcessOptimizationProblem
from pymoo.visualization.scatter import Scatter
from utils import predict_properties, load_models_and_scalers

# 1. 加载模型和标准化器
model_dir = "models/DNN_Optimized"
multi_output_model, scaler_X, scaler_y_stdv, scaler_y_load = load_models_and_scalers(model_dir)

# 2. 定义参数范围
param_bounds = [
    (875, 965),  # 工件温度
    (300, 700),  # 模具温度
    (10, 50)     # 上模速度
]

# 3. 实例化优化问题
problem = FormingProcessOptimizationProblem(
    multi_output_model,  # 传入单个多输出模型
    scaler_X, 
    scaler_y_stdv, 
    scaler_y_load,
    param_bounds=param_bounds
)

# 4. 选择算法
algorithm = NSGA2(
    pop_size=200,       # 种群大小，即每一代保留的个体数量
    eliminate_duplicates=True,  # 是否在种群中移除重复个体
    n_offsprings=200,   # 每代生成的子代数量
    crossover=SBX(prob=0.9, eta=20),    # 模拟二进制交叉
    mutation=PM(prob=0.1, eta=20),      # 变异情况
    constraint_handling="penalty"   
)

# 5. 开始优化
result = minimize(
    problem,
    algorithm,
    ("n_gen", 100),
    seed=42,
    verbose=True
)

# 6. 分析帕累托前沿
pareto_front = result.F  # 所有非支配解的目标值 [stdv, load]
design_space = result.X  # 对应的决策变量

print("\n=== 帕累托最优解示例 ===")
for i in range(min(5, len(pareto_front))):
    print(f"解 {i+1}: 参数={design_space[i]}, 目标值=[晶粒尺寸标准差={pareto_front[i,0]:.2f}, 负载={pareto_front[i,1]:.2f}]")

# 7. 可视化帕累托前沿
plot = Scatter(title="Pareto Front", axis_labels=["晶粒尺寸标准差", "模具载荷"])
plot.add(pareto_front, color="red")
plot.show()