# 文件：open_thoughts/drone_formation/judge.py（更新）
from datasets import Dataset
import json

def formation_judge(ds: Dataset) -> Dataset:
    """基于Crazyswarm的验证标准"""
    def validate_solution(example):
        try:
            # 检查必要组件
            solution = example["deepseek_solution"]
            required_components = [
                "Crazyswarm()", "allcfs.takeoff()"
            ]
    
            return {
                "correct": all(
                    [comp in solution for comp in required_components] 
                )
            }
        except:
            return {"correct": False}
    
    return ds.map(validate_solution).filter(lambda x: x["correct"])
