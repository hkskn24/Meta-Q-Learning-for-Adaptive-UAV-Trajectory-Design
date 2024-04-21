import numpy as np
from rand_param_envs.gym.envs.mujoco import MujocoEnv
from rand_param_envs.gym import utils

class UAVENV:
    RAND_PARAMS = ["poi_loactions"]
    def __init__(self, target_range, poi):
        MujocoEnv.__init__(self, 'uav.xml', 1)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def sample_task(self, n_tasks):
        """
        Generates tasks with different POI numbers and locations for UAV environment
        
        Args:
            n_tasks(int): number of different meta-tasks needed
            
        Returns:
            tasks(list): a list of taskes containing parameters
        """
        max_poi=9
        tasks=[]
        for _ in range(n_tasks):
            target_area_size = 10 # size of the target area

            # randomly generate the number of poi
            n_poi = np.random.randint(1, max_poi + 1)

            # calculate grid size based on n_poi
            grid_size = int(np.sqrt(n_poi))

            grid_step = target_area_size / grid_size

            # generate poi locations
            poi_locations = []
            for i in range(grid_size):
                for j in range(grid_size):
                    x_center = (i + 0.5) * grid_step - target_area_size / 2
                    y_center = (j + 0.5) * grid_step - target_area_size / 2
                    poi_locations.append((x_center, y_center))

            task = {
                "target_area_size": target_area_size,
                "poi_locations":poi_locations
            }
            tasks.append(task)

        return 
    
    def set_task(self, task):
        pass

    def get_task(self):
        pass
    
    def save_parameters(self):
        pass

if __name__ == "__main__":

    env = UAVENV()
    tasks = env.sample_tasks(40)
    while True:
        env.reset()
        env.set_task(np.random.choice(tasks))
        print(env.model.poi_locations)
        for _ in range(100):
            env.render()
            env.step(env.action_space.sample())  # take a random action
culate_energy(action)  # 计算动作对应的能量消耗
            obs, reward, done, info = env.step(action)  # 执行动作并获取观察、奖励等信息

