# [file name] drone_formation.py
import argparse
import os
import random
import math
from datasets import Dataset

class FormationGenerator:
    def __init__(self):
        self.formation_types = ["圆形", "方形", "三角形", "螺旋形", "V形", "多边形"]  # 新增V形队列
        self.motion_types = ["环绕", "固定"]  
        self.target_motions = ["固定", "线性移动", "正弦曲线","圆形"]  
        self.obstacle_strategies = ["扩大半径", "改变队形"]  # 新增避障策略选项

    def generate_question(self, params: dict) -> str:
        question = [
            f"实现{params['escort_drones']}架护航无人机的动态编队系统，"
            f"围绕1架目标无人机（位于基准位置[{params['base_position'][0]},"
            f"{params['base_position'][1]},{params['base_position'][2]}]) "
            f"形成{params['formation_type']}队形。"
        ]
        
        question.append(
            f"编队整体执行{params['motion_type']}运动，" 
            f"目标无人机采用{params['target_motion']}模式。"
        )
        
        if params["has_obstacle"]:
            obstacle_desc = [
                f"当检测到{params['obstacle_drones']}个位于[{params['obstacle_position']}]位置的障碍物时，"
                "编队应自动执行以下避障策略："
            ]
            
            # 生成避障策略描述
            strategies = []
            if "扩大半径" in params["obstacle_strategies"]:
                strategies.append("扩大队形半径")
            if "改变队形" in params["obstacle_strategies"]:
                strategies.append(f"切换为{params['obstacle_formation']}队形")
            
            obstacle_desc.append("同时".join(strategies) + "并通过危险区域")
            question.append("\n".join(obstacle_desc))

        question.extend([
            "技术要求：",
            "1. 使用pycrazyswarm库实现控制逻辑",
            "2. 目标无人机始终作为编队几何中心",
            "3. 队形变换过程需保持平滑过渡",
            "4. 参考代码存在很多if判断，根据具体编队算法保留有效部分，用不到的队形不要在最后结果中体现",
            "5. 在--sim模式下控制所有飞机，否则仅控制escort_drones",
            f"完成{params['cycle_times']}个运动周期后，所有无人机返回初始位置降落。"
        ])
        
        return "\n".join(question)

    def create_dataset(self, num_samples: int) -> Dataset:
        data = []
        for _ in range(num_samples):
            base_pos = [
                round(random.uniform(-2.0, 2.0), 1),
                round(random.uniform(-2.0, 2.0), 1),
                round(random.uniform(0.7, 1.5), 1)
            ]
            
            params = {
                "escort_drones": random.randint(3, 10),
                "base_position": base_pos,
                "flight_radius": round(random.uniform(1.0, 2.0), 1),
                "cycle_times": random.randint(2, 4),
                "has_obstacle": random.random() > 0.1,
                "formation_type": random.choice(self.formation_types),
                "motion_type": random.choice(self.motion_types),
                "target_motion": random.choice(self.target_motions),
                "obstacle_strategies": random.sample(self.obstacle_strategies, k=random.randint(1,2)),  # 随机选择1-2种策略
                "obstacle_drones": random.randint(1, 3),
                "obstacle_position": []
            }
            
            # 生成障碍物替代队形（需不同于原队形）
            if params["has_obstacle"]:
                available_formations = [f for f in self.formation_types if f != params["formation_type"]]
                params["obstacle_formation"] = random.choice(available_formations) if available_formations else params["formation_type"]
                
                for _ in range(params["obstacle_drones"]):
                    angle = random.uniform(0, 2*math.pi)
                    dist = random.uniform(1.0, 2.0)
                    params["obstacle_position"].append([
                        round(base_pos[0] + dist*math.cos(angle), 1),
                        round(base_pos[1] + dist*math.sin(angle), 1),
                        round(base_pos[2] + 0.3, 1)
                    ])
            else:
                params["obstacle_formation"] = None
                params["obstacle_strategies"] = []
            
            params["question"] = self.generate_question(params)
            params.update({
                "solution": None,
                "messages": [],
                "metadata": {}
            })
            data.append(params)
        
        return Dataset.from_dict({k: [dic[k] for dic in data] for k in data[0]})

def filter_problems(example):
    # ...保持原有过滤条件不变...
    # 新增过滤条件：如果选择改变队形策略，必须存在有效替代队形
    if example['has_obstacle'] and "改变队形" in example['obstacle_strategies']:
        if example['formation_type'] == example['obstacle_formation']:
            return False
    return True


def main():
    seed = random.randint(1,100)
    random.seed(seed)
    parser = argparse.ArgumentParser(description='无人机编队数据集生成器')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()

    generator = FormationGenerator()
    ds = generator.create_dataset(args.num_samples)

    ds = ds.filter(filter_problems)
    ds = ds.rename_columns({
        "formation_type": "formation",
        "motion_type": "motion_pattern"
    })

    ds = ds.add_column("source_subset", ["drone_formation"] * len(ds))
    ds = ds.add_column("domain", ["robotics"] * len(ds))
   

    if args.dry_run:
        ds = ds.select(range(min(1, len(ds))))
        print("===== 数据集样例 =====")
        print(ds[0])

    try:
        from reason import reason
        ds = reason(ds)
        print(ds)
    except ImportError as e:
        print(f"警告：推理模块不可用 - {str(e)}")
        ds = ds.add_column("solution", ["待生成"] * len(ds))

    repo_name = f"{os.environ.get('HF_ORG', 'your_org')}/drone-formations-crazyflies_"+str(seed)
    ds.push_to_hub(
        repo_id=repo_name,
        private=os.environ.get("HF_PRIVATE", False),
        token=os.environ.get("HF_TOKEN")
    )
    print(f"数据集已上传至：{repo_name}")

if __name__ == "__main__":
    main()
