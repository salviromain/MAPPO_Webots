# single_controller_simple.py
from deepbots.robots import CSVRobot
import numpy as np
import re

class SingleTurtleBot(CSVRobot):
    def __init__(self):
        super().__init__(emitter_name='emitter', receiver_name='receiver')

        # Identify robot by name and derive channel index.
        # Try: if name contains a digit -> use that as index; else map common words
        self.robot_name = self.getName()  # e.g., "turtlebot_single" or "turtlebot_double"
        idx = self._name_to_index(self.robot_name)
        # convention: emitter_channel = idx, receiver_channel = 100 + idx
        # idx should start at 1 for first robot
        self.emitter_channel = idx
        self.receiver_channel = 100 + idx

        try:
            if self.emitter:
                self.emitter.setChannel(self.emitter_channel)
            if self.receiver:
                self.receiver.setChannel(self.receiver_channel)
        except Exception as e:
            print("Failed to set channels:", e)

        # motors
        self.left_motor = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # lidar (optional)
        try:
            self.lidar = self.getDevice('LDS-01')
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
            print(f"LIDAR initialized for {self.robot_name}")
        except Exception as e:
            self.lidar = None
            print(f"No LiDAR found for {self.robot_name}:", e)

    def _name_to_index(self, name):
        # try to find trailing digits
        m = re.search(r'(\d+)$', name)
        if m:
            return int(m.group(1))
        # map common names
        if 'single' in name.lower():
            return 1
        if 'double' in name.lower() or 'two' in name.lower() or 'second' in name.lower():
            return 2
        # fallback deterministic but simple: hash the name into 1..10
        h = abs(hash(name)) % 10 + 1
        return h

    def create_message(self):
        """Send LiDAR distances to supervisor, prefixing with robot name."""
        if self.lidar:
            ranges = self.lidar.getRangeImage()
            n = len(ranges)
            sampled = [ranges[i] for i in np.linspace(0, n - 1, 8, dtype=int)]
        else:
            sampled = [0.0] * 8

        # first token is robot name so supervisor can attribute the message
        tokens = [self.robot_name] + [str(float(x)) for x in sampled]
        if self.emitter:
            try:
                self.emitter.send(','.join(tokens).encode('utf-8'))
            except Exception as e:
                print(f"{self.robot_name} failed to send message:", e)
        return tokens

    def use_message_data(self, message):
        """
        message is expected to be like ['turn','speed'] or a list where
        message[0] = turn, message[1] = speed (strings or numbers).
        """
        if not message or len(message) < 2:
            return

        try:
            omega_bz = float(message[0])  # angular velocity
            v_bx = float(message[1])      # forward speed
        except (ValueError, TypeError) as e:
            print(f"{self.robot_name} invalid incoming message {message}: {e}")
            return

        # Robot parameters and kinematics
        w = 0.16  # distance between wheels
        left_speed = v_bx - omega_bz * w / 2
        right_speed = v_bx + omega_bz * w / 2

        max_vel = 6.67
        left_speed = np.clip(left_speed, -max_vel, max_vel)
        right_speed = np.clip(right_speed, -max_vel, max_vel)

        # apply velocities
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

# instantiate and run
robot = SingleTurtleBot()
robot.run()
