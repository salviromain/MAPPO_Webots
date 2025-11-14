import numpy as np
import torch
from deepbots.supervisor import CSVSupervisorEnv
from collections.abc import Iterable

class CSVSupervisorSingle(CSVSupervisorEnv):
    def __init__(self):
        super().__init__(emitter_name='emitter', receiver_name='receiver')
        self.robot_node = self.getFromDef('turtlebot_single')
        self.goal_position = np.array([1, -1])  # x, y goal coordinates
        self.position_threshold = 0.1  
        self.max_steps = 10000
        self.timestep = int(self.getBasicTimeStep())
        self.previous_distance = None


    def get_observations(self):
        raw = self.handle_receiver()
        if raw is None:
            return [0.0] * 9

        parts = [x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else str(x) for x in raw]
        
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except Exception:
                nums.append(0.0)

        # Ensure exactly 9 values (id + 8 sensors)
        if len(nums) < 9:
            nums += [0.0] * (9 - len(nums))
        elif len(nums) > 9:
            nums = nums[:9]

        # debug: show received observation (compact)
        try:
            if (self.timestep%10 == 0):
                print(f"Supervisor received (type={type(raw).__name__}): {nums[:9]}")
        except Exception:
            pass

        return nums

    def compute_action(self):
        pos = np.array(self.robot_node.getPosition()[:2])
        vector_to_goal = self.goal_position - pos
        distance = np.linalg.norm(vector_to_goal)
        if distance < self.position_threshold:
            print(f"At goal: distance={distance:.4f}")
            return [0.0, 0.0]

        angle_to_goal = np.arctan2(vector_to_goal[1], vector_to_goal[0])

        # orientation matrix (flattened 3x3)
        rot = self.robot_node.getOrientation()
        yaw = np.arctan2(rot[3], rot[0])  # r10, r00

        angle_diff = np.arctan2(np.sin(angle_to_goal - yaw),
                                np.cos(angle_to_goal - yaw))

        # adjust gains or flip sign if needed
        turn = float(2.0 * angle_diff)
        speed = float(2.0 * max(0.0, 1.0 - abs(angle_diff)))

        print(f"compute_action: pos={pos.tolist()} goal={self.goal_position.tolist()} dist={distance:.3f} "
            f"yaw={yaw:.3f} diff={angle_diff:.3f} turn={turn:.3f}")
        return [turn, speed]


    def step_loop(self):
        obs = self.reset()
        done = False
        steps = 0

        while steps < self.max_steps:
            action = self.compute_action()
            #checks here because of previous errors
            if isinstance(action, np.ndarray):
                action = action.tolist()
            if not isinstance(action, list):
                try:
                    action = list(action)
                except Exception:
                    action = [float(action)]
            action = [float(x) for x in action]

            # show action being sent
            if (self.timestep%10 == 0):
                print(f"Step {steps}: sending action={action} (types: {[type(x).__name__ for x in action]})")

            # call the supervisor step
            result = self.step(action)

            # checks here because of previous errors
            if isinstance(result, tuple) and len(result) >= 3:
                obs, reward, done = result[0], result[1], result[2]
                # stop if done (single-agent: done can be bool)
                if isinstance(done, bool) and done:
                    break
                # multi-agent done might be list/tuple
                if isinstance(done, (list, tuple)) and any(done):
                    break
                # debug: print reward
            if (self.timestep%10 == 0):
                print(f"Step {steps}: reward={reward} done={done}")
            steps += 1




    def reset(self):
        self.simulationResetPhysics()
        super().reset()
        # initialize previous_distance used by get_reward
        pos = np.array(self.robot_node.getPosition()[:2])
        try:
            self.previous_distance = float(np.linalg.norm(self.goal_position - pos))
        except Exception:
            self.previous_distance = None
        return self.get_observations()

    def get_reward(self, action):
        """Return scalar reward based on progress toward the goal.
        Reward = previous_distance - current_distance (positive if we get closer).
        If previous_distance is not set, return 0.0.
        """
        pos = np.array(self.robot_node.getPosition()[:2])
        current_distance = float(np.linalg.norm(self.goal_position - pos))
        if self.previous_distance is None:
            reward = 0.0
        else:
            reward = float(self.previous_distance - current_distance)
        # update for next step
        self.previous_distance = current_distance
        return reward

    def is_done(self):
        pos = np.array(self.robot_node.getPosition()[:2])
        distance = float(np.linalg.norm(self.goal_position - pos))
        return bool(distance < self.position_threshold)

    def get_info(self):
        return None

    def get_default_observation(self):
        return [[0.0]*9]  # match observation size

if __name__=='__main__':
    env = CSVSupervisorSingle()
    env.step_loop()
