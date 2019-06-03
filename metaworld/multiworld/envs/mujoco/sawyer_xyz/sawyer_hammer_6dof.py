from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box


from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

from pyquaternion import Quaternion
from multiworld.envs.mujoco.utils.rotation import euler2quat

class SawyerHammer6DOFEnv(SawyerXYZEnv):
    def __init__(
            self,
            hand_low=(-0.5, 0.40, 0.05),
            hand_high=(0.5, 1, 0.5),
            obj_low=(-0.1, 0.5, 0.02),
            obj_high=(0.1, 0.6, 0.02),
            random_init=False,
            # tasks = [{'goal': np.array([0.24, 0.74, 0.11]),  'hammer_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_pos':np.array([0.24, 0.65, 0.11]), 'obj_init_qpos':0.}], 
            tasks = [{'goal': np.array([0.24, 0.85, 0.05]), 'screw_init_pos':np.array([0.24, 0.71, 0.11]), 'hammer_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_pos':np.array([0.24, 0.71, 0.11])}], 
            goal_low=(0., 0.85, 0.05),
            goal_high=(0.3, 0.9, 0.05),
            hand_init_pos = (0, 0.6, 0.2),
            liftThresh = 0.09,
            rotMode='fixed',#'fixed',
            rewMode='orig',
            **kwargs
    ):
        self.quick_init(locals())
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./100,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low

        if goal_low is None:
            goal_low = self.hand_low

        if obj_high is None:
            obj_high = self.hand_high
        
        if goal_high is None:
            goal_high = self.hand_high

        self.random_init = random_init
        self.liftThresh = liftThresh
        self.max_path_length = 200#150
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.rewMode = rewMode
        self.rotMode = rotMode
        self.hand_init_pos = np.array(hand_init_pos)
        if rotMode == 'fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            self.action_rot_scale = 1./50
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        elif rotMode == 'quat':
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 1]),
            )
        else:
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi/2, -np.pi/2, 0, -1]),
                np.array([1, 1, 1, np.pi/2, np.pi/2, np.pi*2, 1]),
            )
        self.obj_and_goal_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self.observation_space = Box(
                np.hstack((self.hand_low, obj_low, obj_low, obj_low, goal_low)),
                np.hstack((self.hand_high, obj_high, obj_high, obj_high, goal_high)),
        )
        self.reset()


    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
    }

    @property
    def model_name(self):     

        return get_asset_full_path('sawyer_xyz/sawyer_hammer.xml')
        #return get_asset_full_path('sawyer_xyz/pickPlace_fox.xml')

    def viewer_setup(self):
        # top view
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 1.0
        # self.viewer.cam.lookat[2] = 0.5
        # self.viewer.cam.distance = 0.6
        # self.viewer.cam.elevation = -45
        # self.viewer.cam.azimuth = 270
        # self.viewer.cam.trackbodyid = -1
        # side view
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.4
        self.viewer.cam.lookat[1] = 0.75
        self.viewer.cam.lookat[2] = 0.4
        self.viewer.cam.distance = 0.4
        self.viewer.cam.elevation = -55
        self.viewer.cam.azimuth = 180
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        self.render()
        # self.set_xyz_action_rot(action[:7])
        if self.rotMode == 'euler':
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == 'fixed':
            self.set_xyz_action(action[:3])
        elif self.rotMode == 'rotz':
            self.set_xyz_action_rotz(action[:4])
        else:
            self.set_xyz_action_rot(action[:7])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward , reachRew, reachDist, pickRew, hammerRew , hammerDist, screwDist = self.compute_reward(action, obs_dict, mode = self.rewMode)
        self.curr_path_length +=1
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return ob, reward, done, { 'reachRew':reachRew, 'reachDist': reachDist, 'pickRew':pickRew,
                                    'hammerRew': hammerRew, 'epRew' : reward, 'hammerDist': hammerDist, 'screwDist': screwDist}
   
    def _get_obs(self):
        hand = self.get_endeff_pos()
        hammerPos = self.get_body_com('hammer').copy()
        hammerHeadPos = self.data.get_geom_xpos('hammerHead').copy()
        objPos =  self.data.site_xpos[self.model.site_name2id('screwHead')]
        flat_obs = np.concatenate((hand, hammerPos, hammerHeadPos, objPos))
        return np.concatenate([
                flat_obs,
                self._state_goal
            ])

    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        hammerPos = self.get_body_com('hammer').copy()
        hammerHeadPos = self.data.get_geom_xpos('hammerHead').copy()
        objPos =  self.data.site_xpos[self.model.site_name2id('screwHead')]
        flat_obs = np.concatenate((hand, hammerPos, hammerHeadPos, objPos))
        return dict(
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=objPos,
        )

    def _get_info(self):
        pass

    def _set_objCOM_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        objPos =  self.data.get_geom_xpos('handle')
        self.data.site_xpos[self.model.site_name2id('objSite')] = (
            objPos
        )
    

    def _set_obj_xyz_quat(self, pos, angle):
        quat = Quaternion(axis = [0,0,1], angle = angle).elements
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qpos[12:16] = quat.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)


    def _set_hammer_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[16] = pos
        qvel[15] = 0
        self.set_state(qpos, qvel)


    def sample_goals(self, batch_size):
        #Required by HER-TD3
        goals = []
        for i in range(batch_size):
            task = self.tasks[np.random.randint(0, self.num_tasks)]
            goals.append(task['goal'])
        return {
            'state_desired_goal': goals,
        }


    def sample_task(self):
        task_idx = np.random.randint(0, self.num_tasks)
        return self.tasks[task_idx]


    def reset_model(self):
        self._reset_hand()
        task = self.sample_task()
        self.sim.model.body_pos[self.model.body_name2id('box')] = np.array(task['goal'])
        self.sim.model.body_pos[self.model.body_name2id('screw')] = np.array(task['screw_init_pos'])
        self._state_goal = self.sim.model.site_pos[self.model.site_name2id('goal')] + self.sim.model.body_pos[self.model.body_name2id('box')]
        self.obj_init_pos = task['obj_init_pos']
        # self.obj_init_qpos = task['obj_init_qpos']
        self.hammer_init_pos = task['hammer_init_pos']
        self.hammerHeight = self.get_body_com('hammer').copy()[2]
        self.heightTarget = self.hammerHeight + self.liftThresh
        if self.random_init:
            goal_pos = np.random.uniform(
                self.obj_and_goal_space.low,
                self.obj_and_goal_space.high,
                size=(self.obj_and_goal_space.low.size),
            )
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = np.random.uniform(
                    self.obj_and_goal_space.low,
                    self.obj_and_goal_space.high,
                    size=(self.hand_and_obj_space.low.size),
                )
            self.hammer_init_pos = np.concatenate((goal_pos[:2], [self.hammer_init_pos[-1]]))
            screw_pos = goal_pos[-3:].copy()
            screw_pos[1] -= 0.14
            screw_pos[2] += 0.06
            # self.obj_init_qpos = goal_pos[-1]
            self.sim.model.body_pos[self.model.body_name2id('box')] = goal_pos[-3:]
            self.sim.model.body_pos[self.model.body_name2id('screw')] = screw_pos
            self._state_goal = self.sim.model.site_pos[self.model.site_name2id('goal')] + self.sim.model.body_pos[self.model.body_name2id('box')]
        self._set_hammer_xyz(self.hammer_init_pos)
        # self._set_obj_xyz(self.obj_init_qpos)
        # self.obj_init_pos = self.data.site_xpos[self.model.site_name2id('screwHead')]
        self.obj_init_pos = self.sim.model.site_pos[self.model.site_name2id('screwHead')] + self.sim.model.body_pos[self.model.body_name2id('screw')]
        #self._set_obj_xyz_quat(self.obj_init_pos, self.obj_init_angle)
        self.maxHammerDist = np.linalg.norm(np.array([self.hammer_init_pos[0], self.hammer_init_pos[1], self.heightTarget]) - np.array(self.obj_init_pos)) + \
                                self.heightTarget + np.abs(self.obj_init_pos[1] - self._state_goal[1])
        self.curr_path_length = 0
        #Can try changing this
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
            #self.do_simulation(None, self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obsBatch):
        #Required by HER-TD3
        assert isinstance(obsBatch, dict) == True
        obsList = obsBatch['state_observation']
        rewards = [self.compute_reward(action, obs)[0] for  action, obs in zip(actions, obsList)]
        return np.array(rewards)

    def compute_reward(self, actions, obs, mode='orig'):
        if isinstance(obs, dict): 
            obs = obs['state_observation']

        hammerPos = obs[3:6]
        hammerHeadPos = obs[6:9]
        objPos = obs[9:12]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        hammerGoal = self._state_goal[1]

        hammerDist = np.linalg.norm(objPos - hammerHeadPos)
        screwDist = np.abs(objPos[1] - self._state_goal[1])
        reachDist = np.linalg.norm(hammerPos - fingerCOM)

        def reachReward():
            reachRew = -reachDist# + min(actions[-1], -1)/50
            reachDistxy = np.linalg.norm(hammerPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
            reachRew = -reachDist
            #incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew , reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            if hammerPos[2] >= (heightTarget- tolerance):
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True


        def objDropped():
            return (hammerPos[2] < (self.hammerHeight + 0.005)) and (hammerDist >0.02) and (reachDist > 0.02) 
            # Object on the ground, far away from the goal, and from the gripper
            #Can tweak the margin limits
       
        def objGrasped(thresh = 0):
            sensorData = self.data.sensordata
            return (sensorData[0]>thresh) and (sensorData[1]> thresh)

        def orig_pickReward():       
            # hScale = 50
            hScale = 100
            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
            # elif (reachDist < 0.1) and (hammerPos[2]> (self.hammerHeight + 0.005)) :
            elif (reachDist < 0.1) and (hammerPos[2]> (self.hammerHeight + 0.005)) :
                return hScale* min(heightTarget, hammerPos[2])
            else:
                return 0

        def general_pickReward():
            hScale = 50
            if self.pickCompleted and objGrasped():
                return hScale*heightTarget
            elif objGrasped() and (hammerPos[2]> (self.hammerHeight + 0.005)):
                return hScale* min(heightTarget, hammerPos[2])
            else:
                return 0

        def hammerReward():
            # c1 = 1000 ; c2 = 0.03 ; c3 = 0.003
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            if mode == 'general':
                cond = self.pickCompleted and objGrasped()
            else:
                cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
            if cond:
                hammerRew = 1000*(self.maxHammerDist - hammerDist - screwDist) + c1*(np.exp(-((hammerDist+screwDist)**2)/c2) + np.exp(-((hammerDist+screwDist)**2)/c3))
                hammerRew = max(hammerRew,0)
                return [hammerRew , hammerDist, screwDist]
            else:
                return [0 , hammerDist, screwDist]

        reachRew, reachDist = reachReward()
        if mode == 'general':
            pickRew = general_pickReward()
        else:
            pickRew = orig_pickReward()
        hammerRew , hammerDist, screwDist = hammerReward()
        assert ((hammerRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + hammerRew
        return [reward, reachRew, reachDist, pickRew, hammerRew, hammerDist, screwDist] 

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass

if __name__ == '__main__':
    import time
    env = SawyerHammer6DOFEnv()
    for _ in range(1000):
        env.reset()
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.05]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.25]))
        #     # env.data.set_mocap_pos('mocap', np.array([0, 0.6, 0.25]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        for _ in range(100):
            env.render()
            # env.step(env.action_space.sample())
            # if _ < 10:
            #     env.step(np.array([0, 0, -1, 0, 0]))
            # elif _ < 50:
            #     env.step(np.array([0, 0, 0, 0, 1]))
            if _ < 10:
                env.step(np.array([0, 0, -1, 0, 0]))
            else:
                env.step(np.array([0, 1, 0, 0, 1]))
                # env.step(np.array([0, 1, 0, 0, 0]))
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
