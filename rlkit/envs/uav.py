import numpy as np
from gym import spaces
from gym import Env

from . import register_env

@register_env('uav')
class UAVEnv(Env):
    def __init__(self, randomize_tasks=False, n_tasks=2):
        self.coverage_radius = 0.2
        self.time_slots_covered = [0, 0]  # 为每个POI维护一个覆盖计数
        self.time_slots_total = 0
        self.energy = 0

        if randomize_tasks:
            np.random.seed(1337)
            self.poi = [np.random.uniform(-1., 1., size=(2,)) for _ in range(n_tasks)]  # 生成随机POI
        else:
            # 设置特定的POI坐标
            self.poi = [np.array([0.5, 0.5]), np.array([-0.5, -0.5])]

        self.current_poi_index = 0  # 当前POI索引
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))
        self.reset_task(0)

    def reset_task(self, idx):
        self.current_poi_index = idx  
        self.reset()

    def reset_model(self):
        self._state = np.random.uniform(-1., 1., size=(2,))
        self.time_slots_total = 0
        self.time_slots_covered = [0] * len(self.poi)
        self.energy = 0
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)
    
    def get_all_task_idx(self):
        return range(len(self.poi))

    def step(self, action):
        self.time_slots_total += 1
        pre_state = self._state.copy()
        self._state += action

        for i, poi in enumerate(self.poi):
            if np.linalg.norm(self._state - poi) <= self.coverage_radius:
                self.time_slots_covered[i] += 1

        distance = np.linalg.norm(self._state - pre_state)
        self.energy += distance * 50
        reward = self.get_coverage_score() / (self.energy if self.energy > 0 else 1)  # 总覆盖得分与能量的比

        done = False
        ob = self._get_obs()
        return ob, reward, done, dict()
    
    def get_coverage_score(self):
        if self.time_slots_total > 0:
            return sum(self.time_slots_covered) / self.time_slots_total
        return 0

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)
        print('coverage score: ', self.get_coverage_score())
        print('energy consumption: ', self.energy)


