# [file name] reason.py
import os
import re
import yaml
import math
from datasets import Dataset
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from jinja2 import Template
import json
import numpy as np

class FormationReasoner:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        # 控制代码模板
        self.code_template = Template('''\
#!/usr/bin/env python
import math
import yaml
import os
import sys
from pycrazyswarm import *

# 动态参数配置
TARGET_DRONES = 1
OBSTACLE_DRONES = {{ obstacle_drones }}
ESCORT_DRONES = {{ escort_drones }}
BASE_HEIGHT = {{ base_position[2] }}
NORMAL_RADIUS = {{ flight_radius }}
AVOID_RADIUS = {{ flight_radius * 2}}
PATH_RADIUS = {{ path_radius|default(1.0) }}
AVOIDANCE_DIST = 0.7
MOVE_STEP = 0.015
SMOOTH_FACTOR = 0.8
FORMATION_TRANSITION = 0.3

FORMATION_SHAPE = "{{ formation_shape|default('circle') }}"
MOTION_TYPE = "{{ motion_type|default('circular') }}"
FORMATION_MODE = "{{ formation_mode|default('fixed') }}"
AVOIDANCE_STRATEGY = "{{ avoidance_strategy|default('expand') }}"
AVOIDANCE_FORMATION = "{{ avoidance_formation|default('circle')}}"
obstacle_positions = {{ obstacle_positions }}

class DynamicFormation:
    def __init__(self):
        self.swarm = Crazyswarm()
        self.allcfs = self.swarm.allcfs
        self.timeHelper = self.swarm.timeHelper
        self.is_sim = "--sim" in sys.argv

        # 角色分配
        if self.is_sim:
            self.obstacle_drones = self.allcfs.crazyflies[:OBSTACLE_DRONES]
            self.target_drone = self.allcfs.crazyflies[OBSTACLE_DRONES]
            self.escort_drones = self.allcfs.crazyflies[OBSTACLE_DRONES+1 : OBSTACLE_DRONES+1+ESCORT_DRONES]
        else:
            self.escort_drones = self.allcfs.crazyflies[:ESCORT_DRONES]

        # 状态初始化
        self.phase = 0.0
        self.formation_radius = NORMAL_RADIUS
        self.original_radius = NORMAL_RADIUS
        self.target_pos = [0.0, 0.0, BASE_HEIGHT]
        self.path_direction = 1
        self.current_formation = FORMATION_SHAPE

        # 设备初始化
        if self.is_sim:
            for idx, cf in enumerate(self.obstacle_drones):
                cf.cmdPosition(obstacle_positions[idx])
                cf.setLEDColor(1, 0, 0)
            self.target_drone.cmdPosition(self.target_pos)
            self.target_drone.setLEDColor(0, 0, 1)
        for cf in self.escort_drones:
            cf.cmdPosition(cf.initialPosition)
            cf.setLEDColor(0, 1, 0)

    def generate_path(self, angle):
        """多模式路径生成"""
        if MOTION_TYPE == "linear":
            phase = angle % (2 * math.pi)
                
  
            # 计算归一化时间参数 (0.0~1.0)
            t = phase / (2 * math.pi)
            
            # 使用正弦函数实现匀速往复
            # sin(πt) 在[0,2]区间完成一次完整往复
            normalized_pos = math.sin(math.pi * t)  

            # 计算实际位移（-PATH_RADIUS ~ PATH_RADIUS）
            target_x = PATH_RADIUS * normalized_pos
            target_y = 0
        elif MOTION_TYPE == "sinusoidal":
            target_x = PATH_RADIUS * math.cos(angle)
            target_y = PATH_RADIUS * 0.5 * math.sin(2*angle)
        elif MOTION_TYPE == 'fixed':
            return self.target_pos.copy()
        else:  # circular
            target_x = PATH_RADIUS * math.cos(angle)
            target_y = PATH_RADIUS * math.sin(angle)
        
        self.target_pos[0] = self.target_pos[0]*0.9 + target_x*0.1
        self.target_pos[1] = self.target_pos[1]*0.9 + target_y*0.1
        return self.target_pos.copy()

    def avoid_obstacle(self, center):
        """动态避障处理"""
        for obstacle in obstacle_positions:
            dx = center[0] - obstacle[0]
            dy = center[1] - obstacle[1]
            dist = math.hypot(dx, dy)
            if dist < AVOIDANCE_DIST:
                theta = math.atan2(dy, dx)
                offset = (AVOIDANCE_DIST - dist) * 0.6
                return [
                    center[0] + math.cos(theta)*offset,
                    center[1] + math.sin(theta)*offset,
                    BASE_HEIGHT
                ]
        return center

    def update_formation(self):
        """编队自适应逻辑"""
        min_dist = min(
            math.hypot(self.target_pos[0]-o[0], self.target_pos[1]-o[1])
            for o in obstacle_positions
        ) if obstacle_positions else float('inf')
        if min_dist < AVOIDANCE_DIST:
            if AVOIDANCE_STRATEGY == "expand":
                if self.formation_radius < AVOID_RADIUS:
                    self.formation_radius += 0.08 # 平滑改变半径 这个逻辑要保留 可以调整具体参数
            elif AVOIDANCE_STRATEGY == "change":
                self.current_formation = AVOIDANCE_FORMATION
            else:
                if self.formation_radius < AVOID_RADIUS:
                    self.formation_radius += 0.08 # 平滑改变半径
                self.current_formation = AVOIDANCE_FORMATION
        else:
            self.current_formation = FORMATION_SHAPE
        self.formation_radius += (self.original_radius - self.formation_radius) * 0.1


    def get_positions(self):
        """多队形生成器"""
        positions = []
        escort_count = len(self.escort_drones)
        
        if self.current_formation == "triangle":
            angles = [0, 2*math.pi/3, 4*math.pi/3]
            for i in range(escort_count):
                angle = angles[i%3] + (self.phase if FORMATION_MODE == "orbiting" else 0)
                radius = self.formation_radius * (1 + 0.2*(i//3))
                dx = radius * math.cos(angle)
                dy = radius * math.sin(angle)
                positions.append([self.target_pos[0]+dx, self.target_pos[1]+dy, BASE_HEIGHT])
        
        elif self.current_formation == "v-shaped":
            base_angle = math.radians(60)
            for i in range(escort_count):
                tier = i // 2 + 1
                branch_angle = -base_angle if i%2 == 0 else base_angle
                final_angle = branch_angle + (self.phase if FORMATION_MODE == "orbiting" else 0)
                radius = self.formation_radius * tier * 0.8
                dx = radius * math.cos(final_angle)
                dy = radius * math.sin(final_angle)
                positions.append([self.target_pos[0]+dx, self.target_pos[1]+dy, BASE_HEIGHT])

        elif self.current_formation == "spiral":
            for i in range(escort_count):
                base_angle = (2 * math.pi * i) / escort_count
                angle = base_angle + (self.phase if FORMATION_MODE == "orbiting" else 0)
                radius = self.formation_radius * (1 + 0.15 * i)
                dx = radius * math.cos(angle)
                dy = radius * math.sin(angle)
                positions.append([
                    self.target_pos[0] + dx,
                    self.target_pos[1] + dy,
                    BASE_HEIGHT,
                ])
        
        elif self.current_formation == "polygon":
            sides = max(3, min(8, escort_count))
            angle_step = 2 * math.pi / sides
            for i in range(escort_count):
                angle = i * angle_step + (self.phase if FORMATION_MODE == "orbiting" else 0)
                radius = self.formation_radius * (1 + 0.1 * (i // sides))
                dx = radius * math.cos(angle)
                dy = radius * math.sin(angle)
                positions.append([self.target_pos[0]+dx, self.target_pos[1]+dy, BASE_HEIGHT])
        
        elif self.current_formation == "rectangle":
            rows = math.ceil(math.sqrt(escort_count))
            cols = math.ceil(escort_count / rows)
            x_step = self.formation_radius * 2.0 / max(rows, cols)
            y_step = self.formation_radius * 1.5 / max(rows, cols)
            
            cos_phase = math.cos(self.phase) if FORMATION_MODE == "orbiting" else 1.0
            sin_phase = math.sin(self.phase) if FORMATION_MODE == "orbiting" else 0.0
            
            index = 0
            for i in range(rows):
                for j in range(cols):
                    if index >= escort_count:
                        break
                    dx = (j - (cols-1)/2) * x_step
                    dy = (i - (rows-1)/2) * y_step
                    rotated_dx = dx * cos_phase - dy * sin_phase
                    rotated_dy = dx * sin_phase + dy * cos_phase
                    positions.append([
                        self.target_pos[0] + rotated_dx,
                        self.target_pos[1] + rotated_dy,
                        BASE_HEIGHT
                    ])
                    index += 1
        
        else:  # 圆形编队
            for i in range(escort_count):
                angle = (2*math.pi*i)/escort_count + (self.phase if FORMATION_MODE == "orbiting" else 0)
                dx = self.formation_radius * math.cos(angle)
                dy = self.formation_radius * math.sin(angle)
                positions.append([self.target_pos[0]+dx, self.target_pos[1]+dy, BASE_HEIGHT])
        
        return positions


    def run(self):
        """主控制循环"""
        if self.is_sim:    
            for cf in self.obstacle_drones:
                cf.takeoff(targetHeight=BASE_HEIGHT, duration=3.0)                          
            self.timeHelper.sleep(1.5)
            self.target_drone.takeoff(targetHeight=BASE_HEIGHT, duration=3.0)
            self.timeHelper.sleep(1.5)
            for cf in self.escort_drones:
                cf.takeoff(targetHeight=BASE_HEIGHT, duration=2.0)
            self.timeHelper.sleep(3.0)
        else:
            for cf in self.escort_drones:
                cf.takeoff(targetHeight=BASE_HEIGHT, duration=2.0)
            self.timeHelper.sleep(3.0)
            
        angle, cycles = 0.0, 0
        try:
            while cycles < {{ cycle_times }}:
                center = self.generate_path(angle)
                safe_center = self.avoid_obstacle(center)
                self.update_formation()

                if self.is_sim:                      
                    self.target_drone.cmdPosition(safe_center)
                for cf, pos in zip(self.escort_drones, self.get_positions()):
                    current = cf.position()
                    cf.cmdPosition([
                        current[0] + (pos[0]-current[0])*SMOOTH_FACTOR,
                        current[1] + (pos[1]-current[1])*SMOOTH_FACTOR,
                        BASE_HEIGHT
                    ])

                self.phase += 0.05 if FORMATION_MODE == "orbiting" else 0
                angle += MOVE_STEP
                if angle >= 4*math.pi:
                    angle = 0
                    cycles += 1
                self.timeHelper.sleep(0.03)
        finally:
            for cf in self.escort_drones:
                cf.land(targetHeight=0.06, duration=2.0)
            if self.is_sim:
                self.target_drone.land(targetHeight=0.06, duration=2.0)
                for cf in self.obstacle_drones:
                    cf.land(targetHeight=0.06, duration=2.0)

def generate_config():
    """动态生成配置文件"""
    config = {"crazyflies": []}
    is_sim = "--sim" in sys.argv

    if is_sim:
        for i in range(OBSTACLE_DRONES):
            config["crazyflies"].append({
                "id": 10+i,
                "channel": 80,
                "initialPosition": obstacle_positions[i],
                "type": "obstacle"
            })
        config["crazyflies"].append({
            "id": 99,
            "channel": 80,
            "initialPosition": [{{ base_position[0] }}, {{ base_position[1] }}, 0.0],
            "type": "target"
        })

    for i in range(ESCORT_DRONES):
        config["crazyflies"].append({
            "id": 100+i,
            "channel": 80,
            "initialPosition": [
                (i%3)*1.2 - 1.5,
                (i//3)*1.0 - 0.5, 
                0.0
            ],
            "type": "CF21SingleMarker"
        })

    os.makedirs("../launch", exist_ok=True)
    with open("../launch/crazyflies.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

if __name__ == "__main__":
    generate_config()
    controller = DynamicFormation()
    controller.run()
    ''')

    def parse_question(self, question: str) -> dict:
        """使用大模型解析用户问题"""
        prompt = f"""## 指令
请从无人机编队控制的需求描述中提取以下参数，严格按JSON格式返回, 如果不清楚参数可以自行决定，不要返回null：
1. escort_drones (int): 护航无人机数量 
2. target_drones (int): 目标无人机数量
3. obstacle_drones(int): 障碍物无人机数量
4. obstacle_positions: 障碍物坐标列表（三维数组）
5. base_position (list[float]): 三维基准坐标如[0,0,0.7]
6. flight_radius (float): 飞行半径（米）
7. cycle_times (int): 运动周期次数
8. formation_shape (circle/triangle/v-shaped/spiral/polygon/rectangle) : 护航无人机的编队队形
9. motion_type (circular/linear/sinusoidal/fixed)：目标无人机的运动模式
10.formation_mode (fixed/orbiting) ： 护航无人机的编队模式，环绕则绕目标旋转，固定则和目标保持相对静止
11.avoidance_strategy (expand/change/mixed): 护航无人机的编队避障策略
12.avoidance_formation (circle/triangle/v-shaped/spiral/polygon/rectangle) : 护航无人机的避障模式队形,即变换队形为该队形

## 示例
输入：实现5架护航无人机环绕1架目标的动态编队，基准位置(0,0,0.7)，半径0.5m，进行5次周期变换
输出：{{
  "escort_drones": 5,
  "target_drones": 1,
  "obstacle_drones": 0,
  "obstacle_positions": [],
  "base_position": [0.0, 0.0, 0.7],
  "flight_radius": 0.5,
  "cycle_times": 5,
  "formation_shape": "circle",
  "motion_type": "circular",
  "formation_mode": "orbiting",
  "avoidance_strategy": "expand",
  "avoidance_formation": "circle",
}}

## 当前需求
{question}"""

        try:
            print(prompt)
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=512
            )
            print(response.choices[0].message.content)
            # 解析JSON响应
            res = json.loads(response.choices[0].message.content)


            params = {
                "escort_drones": min(10, max(3, int(res.get("escort_drones", 3)))),
                "obstacle_drones": min(3, max(0, int(res.get("obstacle_drones", 0)))),
                "base_position": list(map(float, res.get("base_position", [0,0,0.7])))[:3],
                "flight_radius": float(res.get("flight_radius", 0.5)),
                "cycle_times": max(1, int(res.get("cycle_times", 3))),
                "formation_shape": res.get("formation_shape", "circle"),
                "motion_type": res.get("motion_type", "circular"),
                "formation_mode": res.get("formation_mode", "fixed"),
                "avoidance_strategy": res.get("avoidance_strategy", "expand"),
                "path_radius": float(res.get("path_radius", 1.0)),
            }
            # 自动补全三维坐标
            if len(params["base_position"]) < 3:
                params["base_position"] += [0.7]*(3 - len(params["base_position"]))
            params["obstacle_positions"] = res.get("obstacle_positions", [])
            if params["obstacle_drones"] > 0 and not params["obstacle_positions"]:
                base = params["base_position"]
                params["obstacle_positions"] = [
                                                   [base[0] + 0.5, base[1], base[2]],
                                                   [base[0] - 0.5, base[1], base[2]]
                                               ][:params["obstacle_drones"]]
            return params

        except Exception as e:
            print(f"参数解析失败: {str(e)}")
            # 返回安全默认值
            return {
                "escort_drones": 3,
                "target_drones": 1,
                "base_position": [0.0, 0.0, 0.7],
                "flight_radius": 0.5,
                "cycle_times": 3,
                "color_map": {
                    "target": [1, 0, 0],
                    "escort": [0, 1, 0]
                }
            }

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60),
          stop=stop_after_attempt(3))
    def generate_solution(self, question: str) -> dict:
        """生成完整解决方案"""
        try:
            # 参数提取
            params = self.parse_question(question)
            params["color_map"] = {
                "escort": [0, 1, 0],  # 绿色
                "target": [1, 0, 0]   # 红色
            }
        
            # 渲染控制代码
            example_code = self.code_template.render(**params)
            ques = f"""你是一名资深无人机编队算法工程师，请根据问题描述和参考代码实现控制逻辑：
            # 问题描述:
            {question}

            参考代码:
            {example_code}

            按以下格式响应我的问题：
            
            【代码实现】
            基于参考代码实现完整控制程序

            【代码说明】
            用中文解释算法亮点和实现细节

            """
            # 生成自然语言说明
            print(ques)
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{
                    "role": "user",
                    "content": ques
                }],
                temperature=0.0,
                max_tokens=5120
            )
            res = response.choices[0].message.content
            print(res)
            code = re.search(r"【代码实现】\n(.*?)\n【代码说明】",  res, re.DOTALL).group(1)

            reasoning = re.search(r"【代码说明】\n(.*)", res, re.DOTALL).group(1)
            return {
                "code": code,
                "reasoning": reasoning,
                "params": params
            }
        except Exception as e:
            print(f"Generation Error: {str(e)}")
            return {}

def reason(ds: Dataset) -> Dataset:
    """处理数据集"""
    reasoner = FormationReasoner()

    def _processor(example):
        result = reasoner.generate_solution(example["question"])
        if result:
            return {
                "question": example["question"],
                "solution": f"# 控制代码\n{result['code']}\n\n# 说明\n{result['reasoning']}",
                "messages": [
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": result['reasoning']}
                ],
                "metadata": result["params"]
            }
        return example

    return ds.map(_processor)



if __name__ == "__main__":
    # 测试用例
    test_data = Dataset.from_dict({
        "question": ["实现八架无人机环绕一架目标无人机编队前进的算法，遇到障碍物则变换队形通过，变化队形由环形变为矩形，目标无人机圆形移动，护航无人机环绕，障碍物位置为[0，0，0.9]，向着障碍物前后来回三次, 然后降落。"]
    })

    processed = reason(test_data)
    print(processed[0]["solution"])
   # with open("../launch/crazyflies.yaml") as f:
    #    print("\n生成的配置文件内容：\n" + f.read())

