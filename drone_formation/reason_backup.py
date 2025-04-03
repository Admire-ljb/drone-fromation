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
            api_key="sk-b279dca187df4d0f983c40ae5a218121",
            #api_key=os.getenv("DEEPSEEK_API_KEY"),
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
TARGET_DRONES = {{ target_drones }}      # 编队中心无人机
OBSTACLE_DRONES = 1                    # 障碍物无人机 
ESCORT_DRONES = {{ escort_drones }}      # 护航无人机
BASE_HEIGHT = {{ base_position[2] }}    # 飞行高度
NORMAL_RADIUS = {{ flight_radius }}     # 正常队形半径
AVOID_RADIUS = {{ flight_radius * 1.5 }} # 避障队形半径 
PATH_RADIUS = 1.0                      # 路径半径
AVOIDANCE_DIST = 0.7                   # 避障触发距离
MOVE_STEP = 0.015                      # 路径步长
SMOOTH_FACTOR = 0.22                   # 平滑系数
FORMATION_TRANSITION = 0.3             # 队形变换速度

class DynamicFormation:
    def __init__(self):
        self.swarm = Crazyswarm()
        self.allcfs = self.swarm.allcfs
        self.timeHelper = self.swarm.timeHelper

        self.is_sim = True if "--sim" in sys.argv else False

        # 角色分配和初始化
        if self.is_sim:
            self.obstacle_drones = self.allcfs.crazyflies[0:OBSTACLE_DRONES]
            self.target_drone = self.allcfs.crazyflies[OBSTACLE_DRONES]

        self.escort_drones = self.allcfs.crazyflies[OBSTACLE_DRONES+1::]

        # 状态初始化
        self.phase = 0.0
        self.obstacle_pos = [[0.0, 0.0, 0.9]]  # 固定障碍物 3*OBSTACLE_DRONES
        self.target_pos = [0.0, 0.0, BASE_HEIGHT]
        self.formation_radius = NORMAL_RADIUS

        
        # LED颜色设置
        if self.is_sim:
            for cf in self.obstacle_drones:
                cf.setLEDColor(1, 0, 0)  # 红
            self.target_drone.setLEDColor(0, 0, 1)    # 蓝
            for cf in self.escort_drones:
                cf.setLEDColor(0, 1, 0)              # 绿
        else:
            for cf in self.escort_drones:
                cf.setLEDColor(0, 1, 0)              # 绿
            
        # 初始定位
        if self.is_sim:
            for id, cf in enumerate(self.obstacle_drones):
                cf.cmdPosition(self.obstacle_pos[0])

            self.target_drone.cmdPosition(self.target_pos)
            for cf in self.escort_drones:
                cf.cmdPosition(cf.initialPosition)
        else:
            for cf in self.escort_drones:
                cf.cmdPosition(cf.initialPosition)

    def generate_path(self, angle):
        """生成中心运动轨迹"""
        target_x = PATH_RADIUS * math.cos(angle)
        target_y = PATH_RADIUS * math.sin(angle)
        self.target_pos[0] = self.target_pos[0]*0.9 + target_x*0.1
        self.target_pos[1] = self.target_pos[1]*0.9 + target_y*0.1
        return self.target_pos.copy()

    def avoid_obstacle(self, center):
        """动态避障处理"""
        dx = center[0] - self.obstacle_pos[0]
        dy = center[1] - self.obstacle_pos[1]
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
        """队形自适应调整"""
        dx = self.target_pos[0] - self.obstacle_pos[0]
        dy = self.target_pos[1] - self.obstacle_pos[1]
        current_dist = math.hypot(dx, dy)
        
        target_radius = AVOID_RADIUS if current_dist < AVOIDANCE_DIST else NORMAL_RADIUS
        self.formation_radius += (target_radius - self.formation_radius) * FORMATION_TRANSITION

    def get_positions(self):
        """生成护航编队"""
        positions = []
        for i in range(ESCORT_DRONES):
            angle = (2*math.pi*i)/ESCORT_DRONES + self.phase
            dx = self.formation_radius * math.cos(angle)
            dy = self.formation_radius * math.sin(angle)
            positions.append([
                self.target_pos[0] + dx,
                self.target_pos[1] + dy,
                BASE_HEIGHT
            ])
        return positions

    def run(self):
        """分级飞行控制"""
        # 分阶段起飞
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
            while cycles < {{ cycle_times }}:  # 动态周期控制
                center = self.generate_path(angle)
                safe_center = self.avoid_obstacle(center)
                self.update_formation()
                
                # 移动控制
                if self.is_sim:                      
                    self.target_drone.cmdPosition(safe_center)
                for cf, pos in zip(self.escort_drones, self.get_positions()):
                    current = cf.position()
                    cf.cmdPosition([
                        current[0] + (pos[0]-current[0])*SMOOTH_FACTOR,
                        current[1] + (pos[1]-current[1])*SMOOTH_FACTOR,
                        BASE_HEIGHT
                    ])
                
                # 状态更新
                self.phase += 0.05
                angle += MOVE_STEP
                if angle >= 4*math.pi:  # 完成往返
                    angle = 0
                    cycles += 1
                self.timeHelper.sleep(0.03)
        finally:
            # 安全降落
            for cf in self.escort_drones:
                cf.land(targetHeight=0.06, duration=2.0)
            self.timeHelper.sleep(2.0)
            if self.is_sim:
                self.target_drone.land(targetHeight=0.06, duration=2.0)
                self.obstacle_drone.land(targetHeight=0.06, duration=2.0)

def generate_config():
    # 角色分配和初始化

    config = {
        "crazyflies": [
            *[
                {
                    "id": 1+i,
                    "channel": 80,
                    "initialPosition": [0.0, 0.0, 0.0],  # 障碍物初始位置
                    "type": "obstacle"
                } for i in range(OBSTACLE_DRONES)
            ],
            {
                "id": 999,
                "channel": 80,
                "initialPosition": [{{ base_position[0] }}, {{ base_position[1] }}, 0.0],
                "type": "center"
            },
            *[{
                "id": 3+i,
                "channel": 80,
                "initialPosition": [
                    (i%2)*0.6 - 0.3,
                    (i//2)*0.6 - 0.3,
                    0.0
                ],
                "type": "CF21SingleMarker"
            } for i in range(ESCORT_DRONES)]
        ]
    }

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
4. base_position (list[float]): 三维基准坐标如[0,0,0.7]
5. flight_radius (float): 飞行半径（米）
6. cycle_times (int): 运动周期次数
7. others: 其他

## 示例
输入：实现5架护航无人机环绕1架目标的动态编队，基准位置(0,0,0.7)，半径0.5m，进行5次周期变换
输出：{{"escort_drones":5, "target_drones":1, "obstacle_drones":0, "base_position":[0.0,0.0,0.7], "flight_radius":0.5, "cycle_times":5}}

## 当前需求
{question}"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=512
            )
            print(response.choices[0].message.content) 
            # 解析JSON响应
            res_json = json.loads(response.choices[0].message.content)
            
            # 参数验证
            params = {
                "escort_drones": int(res_json.get("escort_drones" , 3)),
                "target_drones": max(1, int(res_json.get("target_drones", 1))),
                "obstacle_drones": int(res_json.get("escort_drones" , 1)),
                "base_position": list(map(float, res_json.get("base_position", [0,0,0.7])))[:3],
                "flight_radius": float(res_json.get("flight_radius"or 0.5)),
                "cycle_times": max(1, int(res_json.get("cycle_times", 3))),
                "color_map": {
                    "target": [1, 0, 0],
                    "escort": [0, 1, 0]
                }
            }
            
            # 自动补全三维坐标
            if len(params["base_position"]) < 3:
                params["base_position"] += [0.7]*(3 - len(params["base_position"]))
                
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
            
            # 生成自然语言说明
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{
                    "role": "user",
                    "content": f"""你是一名资深无人机编队算法工程师，请根据问题描述和参考代码实现控制逻辑。按以下格式响应：

【问题分析】
用中文解释需求中的关键参数和控制目标

【代码实现】
基于参考代码实现完整控制程序，要求：
1. 使用pycrazyswarm库
2. 保持运动队形不变和LED指示

【参考代码】
{example_code}

【代码说明】
用中文解释算法亮点和实现细节

当前问题：{question}"""
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
        "question": ["实现五架无人机环绕一架目标无人机编队前进的算法，遇到障碍物则变换队形通过，变化队形为矩形，目标无人机线性前进通过，障碍物位置为[0，0，0.9]，向着障碍物前后来回三次, 然后降落。"]
    })
    
    processed = reason(test_data)
    print(processed[0]["solution"])
   # with open("../launch/crazyflies.yaml") as f:
    #    print("\n生成的配置文件内容：\n" + f.read())
