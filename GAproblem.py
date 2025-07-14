from pymoo.core.problem import ElementwiseProblem
from utils import predict_properties
import numpy as np

class FormingProcessOptimizationProblem(ElementwiseProblem):
    def __init__(self, 
                 multi_output_model,  # 改为单个多输出模型
                 scaler_X, 
                 scaler_y_stdv, 
                 scaler_y_load,
                 param_bounds,
                 target_stdv=None):
        
        self.multi_output_model = multi_output_model
        self.scaler_X = scaler_X
        self.scaler_y_stdv = scaler_y_stdv
        self.scaler_y_load = scaler_y_load

        n_var = len(param_bounds)
        xl = np.array([b[0] for b in param_bounds])
        xu = np.array([b[1] for b in param_bounds])
        # 两个目标
        super().__init__(n_var = n_var, n_obj = 2, n_constr=2, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.atleast_2d(x)  # (1, n_var)
        x_full = np.column_stack([
            x[:, 0],  # 工件温度
            x[:, 1],  # 模具温度
            x[:, 2]   # 上模速度
        ])

        stdv, load = predict_properties(
            x_full,
            self.multi_output_model,  # 使用单个多输出模型
            self.scaler_X,
            self.scaler_y_stdv,
            self.scaler_y_load
        )
        
        # 目标：最小化晶粒尺寸标准差(stdv)和最小化模具载荷(load)
        out["F"] = np.column_stack([stdv, load])

        
        # 约束
        max_load = 350000  # 最大允许载荷
        min_stdv = 5       # 最小标准差
        
        out["G"] = np.column_stack([
            load - max_load,  # 约束：load <= max_load
            min_stdv - stdv   # 约束：stdv >= min_stdv
        ])
       
