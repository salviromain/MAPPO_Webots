# csv_supervisor_multi.py
import json
import math
import re
import numpy as np
from controller import Supervisor as RealSupervisor

from deepbots.supervisor import CSVSupervisorEnv

class CSVSupervisorMulti(CSVSupervisorEnv):
    CONTROLLER_NAME = "single_controller_simple"  # controller name used by robots

    def __init__(self):
        super().__init__(emitter_name='emitter_super', receiver_name='receiver_super')

        # discovery
        self.robots = []
        self.robot_names = []
        self.max_steps = 10000
        self.timestep = int(self.getBasicTimeStep())
        self.position_threshold = 0.1 #how close a robot must be to the goal to end episode
        self.goal_position = np.array([-2.4, 1.5]) #setting the position of the goal
        
        # Arena bounds (CircleArena has radius 4)
        self.arena_radius = 3.8  # slightly less than 4 to keep goal inside
        
        # Find the goal marker visual in the world
        self.goal_marker = None
        root_children = self.getRoot().getField('children')
        count = root_children.getCount()
        for i in range(count):
            node = root_children.getMFNode(i)
            try:
                node_name = node.getDef()
                if node_name == 'goal_marker':
                    self.goal_marker = node
                    print("Goal marker found!")
                    break
            except Exception:
                pass
        
        #get robots
        root_children = self.getRoot().getField('children')
        count = root_children.getCount()
        for i in range(count):
            node = root_children.getMFNode(i)
            try:
                controller_field = node.getField('controller')
                controller_value = controller_field.getSFString()
            except Exception:
                controller_value = ""
            try:
                name_field = node.getField('name')
                name_value = name_field.getSFString()
            except Exception:
                name_value = node.getDef() or f"robot_{i}"
            if controller_value == self.CONTROLLER_NAME or "turtlebot" in name_value.lower():
                self.robots.append(node)
                self.robot_names.append(name_value)

        # order the robots
        if self.robots:
            self.robot_names, self.robots = zip(*sorted(zip(self.robot_names, self.robots)))
            self.robot_names = list(self.robot_names)
            self.robots = list(self.robots)
        else:
            self.robot_names, self.robots = [], []

        # mapping supervisor order -> index
        self.num_robots = len(self.robot_names)
        self.robot_index = {name: idx for idx, name in enumerate(self.robot_names)}

        print(f"Supervisor found {self.num_robots} robot(s): {self.robot_names}")

        self.previous_distances = [None] * self.num_robots

    # consistent name->index mapping used by robots as well
    def name_to_idx(self, name):
        m = re.search(r'(\d+)$', name)
        if m:
            return int(m.group(1))

    # --- goal generation --------------------------------------------------------
    def generate_random_goal(self):
        """Generate a random goal position within the arena."""
        # Random angle and radius within arena
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.5, self.arena_radius)
        
        goal_x = radius * np.cos(angle)
        goal_y = radius * np.sin(angle)
        
        self.goal_position = np.array([goal_x, goal_y])
        
        # Update the visual marker if found
        if self.goal_marker:
            try:
                trans_field = self.goal_marker.getField('translation')
                if trans_field:
                    trans_field.setSFVec3f([float(goal_x), float(goal_y), 0.05])
                    print(f"Goal updated to: ({goal_x:.2f}, {goal_y:.2f})")
            except Exception as e:
                print(f"Could not update goal marker: {e}")
        else:
            print(f"Goal updated to: ({goal_x:.2f}, {goal_y:.2f}) [marker not found]")

    # --- receiver parsing -----------------------------------------------------
    def handle_receiver(self):
        """
        Read all packets and return a dict: {robot_name: [float,...]}
        Accepts JSON messages of form {"name": "...", "lidar": [..]}.
        Also accepts legacy "name:v1,v2,..." strings for backward compatibility.
        Sanitizes non-finite values by replacing them with a safe max (3.5).
        """
        messages = {}
        max_range = 3.5

        while self.receiver.getQueueLength() > 0:
            raw = self.receiver.getString()  # str
            raw_str = str(raw).strip()
            # try JSON first
            parsed_list = None
            try:
                obj = json.loads(raw_str)
                if isinstance(obj, dict) and 'name' in obj:
                    name = obj['name']
                    lidar = obj.get('lidar', [])
                    # check numeric list not containing inf 
                    clean = []
                    for v in lidar:
                        try:
                            fv = float(v)
                            if not math.isfinite(fv):
                                fv = float(max_range)
                        except Exception:
                            fv = float(max_range)
                        clean.append(fv)
                    # ensure length 8
                    if len(clean) < 8:
                        clean += [float(max_range)] * (8 - len(clean))
                    else:
                        clean = clean[:8]
                    parsed_list = (name, clean)
            except Exception:
                parsed_list = None

            if parsed_list is None:
                # malformed: log and skip
                print(f"Ignoring malformed receiver packet: {raw_str}")
                self.receiver.nextPacket()
                continue

            name, clean_list = parsed_list
            messages[name] = clean_list
            self.receiver.nextPacket()

        return messages

    # --- observation collection ------------------------------------------------
    def get_observations(self):
        """
        Build flattened observation list. Each robot contributes 9 numbers:
         [0.0, lidar0, lidar1, ..., lidar7]
        If a robot didn't send anything this timestep, we use a default.
        """
        raw_messages = self.handle_receiver()  # dict name -> list[8 floats]
        per_robot_obs = {name: [0.0] * 9 for name in self.robot_names}

        if not raw_messages:
            # flatten defaults
            return [val for name in self.robot_names for val in per_robot_obs[name]]

        for name, lidar_list in raw_messages.items():
            if name in per_robot_obs:
                per_robot_obs[name] = [0.0] + [float(x) for x in lidar_list]
            else:
                # unknown robot: ignore but print debug
                print(f"Received message from unknown robot '{name}', ignoring.")

        # flatten in deterministic order
        flattened = []
        for name in self.robot_names:
            flattened.extend(per_robot_obs.get(name, [0.0] * 9))
        return flattened

    # --- compute action for all robots ----------------------------------------
    def compute_action(self):
    # not using the observations yet
        actions = []
        for idx, node in enumerate(self.robots):
            pos = np.array(node.getPosition()[:2])
            vector_to_goal = self.goal_position - pos
            distance = np.linalg.norm(vector_to_goal)
            if distance < self.position_threshold:
                actions.extend([0.0, 0.0])
                continue

            angle_to_goal = np.arctan2(vector_to_goal[1], vector_to_goal[0])
            rot = node.getOrientation()

            # yaw extraction
            try:
                if len(rot) == 9:
                    # rotation matrix flattened row-major: r10 = rot[3], r00 = rot[0]
                    yaw = np.arctan2(rot[3], rot[0])
                elif len(rot) == 4:
                    # axis-angle: (ax,ay,az,angle)
                    ax, ay, az, ang = rot
                    n = math.sqrt(ax*ax + ay*ay + az*az)
                    if n < 1e-8:
                        yaw = 0.0
                    else:
                        ax, ay, az = ax/n, ay/n, az/n
                        c = math.cos(ang); s = math.sin(ang); C = 1 - c
                        r00 = ax*ax*C + c
                        r10 = ay*ax*C + az*s
                        yaw = math.atan2(r10, r00)
                else:
                    yaw = 0.0
            except Exception:
                yaw = 0.0

            angle_diff = math.atan2(math.sin(angle_to_goal - yaw), math.cos(angle_to_goal - yaw))
            #weights modifiable depending on how we want the robots to behave
            k_turn = 1.0
            k_speed = 0.6
            max_omega = 1.2   # rad/s
            max_speed = 0.5   # m/s (linear)

            turn = float(np.clip(k_turn * angle_diff, -max_omega, max_omega))
            # reduce forward speed when large angle_diff; never negative to avoid unexpected backward motion
            speed = float(max(0.0, min(max_speed, k_speed * (1.0 - 0.5 * abs(angle_diff)))))

            actions.extend([turn, speed])
        return actions

    # --- step loop -------------------------------------------------------------
    def step_loop(self):
        obs = self.reset()
        done = False
        steps = 0

        while steps < self.max_steps:
            action = self.compute_action()
            action = [float(x) for x in action]

            
            # send actions to robots by name mapping (consistent channel mapping)
            import json

            for idx, name in enumerate(self.robot_names):
                turn = float(action[2 * idx])
                speed = float(action[2 * idx + 1])

                target_channel = 100 + self.name_to_idx(name)
                self.emitter.setChannel(target_channel)

                msg = json.dumps({"name": name, "turn": turn, "speed": speed})
                # send a single JSON object (no manual newline/framing needed)
                self.emitter.send(msg.encode("utf-8"))
                if (steps%50 ==0):
                    # debug print
                    print("SUPERVISOR SENDING:", msg)

            # step simulation forward once (let Webots deliver messages)
            result = self.step(action)

            # Real Webots step: advances the simulation clock
            RealSupervisor.step(self, self.timestep)



            # Check if all robots have reached the goal
            if self.is_done():
                print(f"All robots reached the goal at step {steps}! Resetting...")
                obs = self.reset()
                steps = 0
                continue

            if isinstance(result, tuple) and len(result) >= 3:
                obs, reward, done = result[:3]
                if isinstance(done, bool) and done:
                    break
                if isinstance(done, (list, tuple)) and any(done):
                    break

            if steps % 50 == 0:
                rew = result[1] if isinstance(result, tuple) and len(result) > 1 else "N/A"
                #print(f"Step {steps}: reward={rew}")

            steps += 1

    # --- reset & reward helpers -----------------------------------------------
    def reset(self):
        self.simulationResetPhysics()
        super().reset()
        
        # Generate a new random goal
        self.generate_random_goal()
        
        for idx, node in enumerate(self.robots):
            try:
                pos = np.array(node.getPosition()[:2])
                self.previous_distances[idx] = float(np.linalg.norm(self.goal_position - pos))
            except Exception:
                self.previous_distances[idx] = None
        return self.get_observations()

    def get_reward(self, action):
        total = 0.0
        for idx, node in enumerate(self.robots):
            pos = np.array(node.getPosition()[:2])
            current_distance = float(np.linalg.norm(self.goal_position - pos))
            prev = self.previous_distances[idx]
            if prev is None:
                delta = 0.0
            else:
                delta = prev - current_distance
            self.previous_distances[idx] = current_distance
            total += float(delta)
        return float(total)

    def is_done(self):
        """Check if ALL robots have reached the goal."""
        if not self.robots:
            return False
        for node in self.robots:
            pos = np.array(node.getPosition()[:2])
            if np.linalg.norm(self.goal_position - pos) >= self.position_threshold:
                return False
        return True

    def get_info(self):
        return None

    def get_default_observation(self):
        return [[0.0] * 9] * max(1, self.num_robots)

if __name__ == "__main__":
    supervisor = CSVSupervisorMulti()
    supervisor.step_loop()
