from pymoo.core.problem import ElementwiseProblem
from utils import predict_properties
import numpy as np
class FormingProcessOptimizationProblem(ElementwiseProblem):
    def __init__(self, 
                 force_model, load_model, scaler_X, scaler_y_force, scaler_y_load,
                 param_bounds,
                 target_force=None):
        
        self.force_model = force_model
        self.load_model = load_model
        self.scaler_X = scaler_X
        self.scaler_y_force = scaler_y_force
        self.scaler_y_load = scaler_y_load
        # self.target_force = target_force

        n_var = len(param_bounds)
        xl = np.array([b[0] for b in param_bounds])
        xu = np.array([b[1] for b in param_bounds])
        # 两个目标
        super().__init__(n_var=n_var, n_obj = 2, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.atleast_2d(x)  # (1, n_var)
        x_full = np.column_stack([
            x[:, 0],  # 工件温度
            x[:, 1],  # 上模温度
            x[:, 1],  # 下模温度（强制等于上模温度）
            x[:, 2]   # 上模速度
        ])

        force, load = predict_properties(
            x_full,
            self.force_model,
            self.load_model,
            self.scaler_X,
            self.scaler_y_force,
            self.scaler_y_load
        )
        # 不带约束
        out["F"] = np.column_stack([force, load])
        
        # 带约束的
        # penalty = 0
        # if self.target_force is not None:
        #     penalty = np.abs(force - self.target_force)
        # fitness = load + penalty
        # out["F"] = fitness
       
