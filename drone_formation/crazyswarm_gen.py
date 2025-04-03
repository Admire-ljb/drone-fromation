import numpy as np
from datasets import Dataset
from typing import Dict, List
import json

def generate_crazyswarm_scenarios(num_samples: int = 1000) -> Dict[str, List]:
    """基于Crazyswarm仿真平台生成无人机编队任务场景"""
    np.random.seed(42)
    
    # 编队类型配置
    formation_types = [
        "grid", "circle", "triangle", 
        "spiral", "arrow", "custom"
    ]
    
    # 环境参数范围
    wind_speeds = np.linspace(0, 15, 10)  # m/s
    obstacle_densities = np.linspace(0, 1, 5)
    
    data = {
        "problem": [],
        "formation_type": [],
        "num_drones": [],
        "start_positions": [],
        "target_positions": [],
        "environment_params": [],
        "control_params": [],
        "test_cases": [],
        "starter_code": [],
        "domain": []
    }
    
    for i in range(num_samples):
        # 生成随机场景参数
        formation = np.random.choice(formation_types)
        n_drones = np.random.randint(3, 10)
        wind = np.random.choice(wind_speeds)
        obstacle = np.random.choice(obstacle_densities)
        
        # 生成起始和目标位置（基于Crazyswarm的典型坐标系）
        start_pos = [[x*0.5, y*0.5, 1.0] for x, y in np.random.rand(n_drones, 2)]
        target_pos = [[x*1.0+2.0, y*1.0+2.0, 1.5] for x, y in np.random.rand(n_drones, 2)]
        
        # 控制参数（PID典型范围）
        control_params = {
            "kp": round(np.random.uniform(0.5, 2.0), 2),
            "ki": round(np.random.uniform(0.0, 0.5), 2),
            "kd": round(np.random.uniform(0.1, 1.0), 2)
        }
        
        # 构建问题描述
        problem_desc = (
            f"在Crazyswarm仿真环境中，使用PID控制实现{n_drones}架无人机的{formation}编队。"
            f"环境参数：风速{wind}m/s，障碍密度{obstacle:.1f}。要求实现从初始位置到目标位置的队形变换，"
            "并满足以下性能指标："
        )
        
        # 测试用例
        test_case = {
            "formation_error": round(np.random.uniform(0.05, 0.2), 2),  # 允许的队形误差(m)
            "convergence_time": np.random.randint(10, 30),         # 收敛时间(s)
            "energy_efficiency": round(np.random.uniform(0.7, 0.95), 2), # 能量效率
            "collision_constraint": "无碰撞"
        }
        
        # 起始代码模板
        starter_code = (
            "from pycrazyswarm import Crazyswarm\n\n"
            "def main():\n"
            "    swarm = Crazyswarm()\n"
            "    timeHelper = swarm.timeHelper\n"
            "    allcfs = swarm.allcfs\n"
            "    \n"
            "    # 在此处实现编队控制算法\n"
            "    # PID参数：{control_params}\n"
            "    # 初始位置：{start_pos}\n"
            "    # 目标位置：{target_pos}\n"
        )
        
        # 填充数据
        data["problem"].append(problem_desc)
        data["formation_type"].append(formation)
        data["num_drones"].append(n_drones)
        data["start_positions"].append(json.dumps(start_pos))
        data["target_positions"].append(json.dumps(target_pos))
        data["environment_params"].append(json.dumps({"wind_speed": wind, "obstacle_density": obstacle}))
        data["control_params"].append(json.dumps(control_params))
        data["test_cases"].append(json.dumps(test_case))
        data["starter_code"].append(starter_code.format(
            control_params=control_params,
            start_pos=start_pos,
            target_pos=target_pos
        ))
        data["domain"].append("drone_formation")
    
    return data

def load_drone_formation_data() -> Dataset:
    """加载基于Crazyswarm生成的1000条编队任务数据"""
    scenarios = generate_crazyswarm_scenarios(1000)
    return Dataset.from_dict(scenarios)
