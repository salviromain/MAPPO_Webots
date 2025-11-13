# single_controller_simple.py
from deepbots.robots import CSVRobot
import numpy as np
import re
import json
import sys
import math

class SingleTurtleBot(CSVRobot):
    def __init__(self):
        # initialize parent (sets up emitter/receiver attributes)
        super().__init__(emitter_name='emitter', receiver_name='receiver')

        # ensure we have a timestep attribute
        try:
            self.timestep = int(self.getBasicTimeStep())
        except Exception:
            self.timestep = 64

        # robot identity
        self.robot_name = self.getName()
        idx = self._name_to_index(self.robot_name)

        # channels:
        # - robots send to supervisor on channel 100 (emitter_channel)
        # - robots listen for supervisor commands on channel 100 + idx (receiver_channel)
        self.emitter_channel = 100
        self.receiver_channel = 100 + idx
        try:
            # emitter sends to supervisor's receiver channel (100)
            self.emitter.setChannel(self.emitter_channel)
            # each robot's receiver listens to its dedicated channel
            self.receiver.setChannel(self.receiver_channel)
            print(f"{self.robot_name}: emitter→{self.emitter_channel}, receiver→{self.receiver_channel}", file=sys.stderr)
        except Exception as e:
            print(f"{self.robot_name} channel setup failed: {e}", file=sys.stderr)

        # motors
        self.left_motor = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # lidar (optional) - store max_range for sanitizing
        try:
            self.lidar = self.getDevice('LDS-01')
            self.lidar.enable(self.timestep)
            # some Lidars provide point cloud enable; ignore if not available
            try:
                self.lidar.enablePointCloud()
            except Exception:
                pass
            # try to get a declared maxRange, fallback to 3.5 as in the world file
            try:
                self.lidar_max_range = float(self.lidar.getMaxRange())
            except Exception:
                self.lidar_max_range = 3.5
            print(f"LIDAR initialized for {self.robot_name} (max_range={self.lidar_max_range})", file=sys.stderr)
        except Exception as e:
            self.lidar = None
            self.lidar_max_range = 3.5
            print(f"No LiDAR found for {self.robot_name}: {e}", file=sys.stderr)

        # initialize command storage
        self.command_turn = 0.0
        self.command_speed = 0.0

    def _name_to_index(self, name):
        m = re.search(r'(\d+)$', name)
        if m:
            return int(m.group(1))
        if 'single' in name.lower():
            return 1
        if 'double' in name.lower() or 'two' in name.lower() or 'second' in name.lower():
            return 2
        return abs(hash(name)) % 10 + 1

    def create_message(self):
        """Send a JSON message with sanitized lidar distances."""
        if self.lidar:
            ranges = self.lidar.getRangeImage()
            n = len(ranges)
            # sample 8 evenly spaced points
            sampled = [float(ranges[i]) for i in np.linspace(0, n - 1, 8, dtype=int)]
        else:
            sampled = [0.0] * 8

        # sanitize: replace non-finite values with lidar_max_range
        clean = []
        for v in sampled:
            try:
                fv = float(v)
                if not math.isfinite(fv):
                    fv = float(self.lidar_max_range)
            except Exception:
                fv = float(self.lidar_max_range)
            clean.append(fv)

        payload = {
            "name": self.robot_name,
            "lidar": clean
        }

        if self.emitter:
            try:
                message = json.dumps(payload)
                # send single JSON object (no newline). Supervisor reads whole packet.
                self.emitter.send(message.encode("utf-8"))
                # debug to stderr only
                print(f"{self.robot_name} -> sent JSON payload (lidar sample): {clean}", file=sys.stderr)
            except Exception as e:
                print(f"{self.robot_name} failed to send message: {e}", file=sys.stderr)

        # return the list form for compatibility with parent class expectations (if used)
        return clean

    def use_message_data(self, message):
        """
        Called by CSVRobot base when a packet arrives (message likely bytes).
        Decode one packet, parse JSON, and if it's intended for this robot, apply it.
        """
        if not message:
            return

        # 1. Decode bytes → str
        if isinstance(message, (bytes, bytearray)):
            try:
                message = message.decode("utf-8").strip()
            except Exception:
                print(f"{self.robot_name} could not decode message", file=sys.stderr)
                return

        # 2. Parse JSON (expecting a single JSON object per packet)
        try:
            data = json.loads(message)
        except Exception:
            print(f"{self.robot_name} bad JSON from supervisor: {message}", file=sys.stderr)
            return

        # 3. Ignore messages intended for other robots
        if data.get("name") != self.robot_name:
            return

        # 4. Extract turn + speed safely
        try:
            omega_bz = float(data["turn"])
            v_bx = float(data["speed"])
        except Exception:
            print(f"{self.robot_name} missing turn/speed fields: {data}", file=sys.stderr)
            return

        # 5. Success → store them (used each step)
        self.command_turn = omega_bz
        self.command_speed = v_bx

        # Immediately apply motors (or will be applied at next step() call)
        self._apply_wheel_speeds(self.command_turn, self.command_speed)

    def _apply_wheel_speeds(self, omega_bz, v_bx):
        # === 1. Clamp body velocities to physical limits ===
        MAX_V = 0.22       # m/s (TurtleBot3 Burger real max)
        MAX_OMEGA = 2.75   # rad/s

        v_bx = np.clip(v_bx, -MAX_V, MAX_V)
        omega_bz = np.clip(omega_bz, -MAX_OMEGA, MAX_OMEGA)

        # === 2. Differential-drive kinematics ===
        wheel_radius = 0.033    # m
        wheel_base = 0.160      # m
        max_wheel_rad_s = 6.67  # rad/s

        # convert body linear/ang to wheel linear speeds (m/s)
        left_lin = v_bx - omega_bz * wheel_base / 2.0
        right_lin = v_bx + omega_bz * wheel_base / 2.0

        # convert to wheel angular velocity (rad/s)
        left_speed = left_lin / wheel_radius
        right_speed = right_lin / wheel_radius

        # === 3. Clamp wheel speeds ===
        left_speed = float(np.clip(left_speed, -max_wheel_rad_s, max_wheel_rad_s))
        right_speed = float(np.clip(right_speed, -max_wheel_rad_s, max_wheel_rad_s))

        # === 4. Apply ===
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
    def run(self):
        """
        Main loop overriding CSVRobot.run to ensure we send lidar and poll for packets each timestep.
        CSVRobot may already have a run() but overriding here gives explicit behaviour:
         - send lidar every timestep
         - process all incoming packets via receiver queue and call use_message_data
         - step the simulation
        """
        # If CSVRobot already provides run() with expected behaviour, this will replace it.
        # Use caution if the deepbots API expects a different contract, but this pattern is common.
        while self.step() != -1:
            # send lidar (non-blocking)
            self.create_message()

            # process all incoming packets
            while self.receiver.getQueueLength() > 0:
                try:
                    raw = self.receiver.getString()
                    # pass raw to use_message_data which handles decoding & JSON parse
                    self.use_message_data(raw)
                except Exception as e:
                    print(f"{self.robot_name} error reading packet: {e}", file=sys.stderr)
                finally:
                    # inform Webots we consumed the packet
                    try:
                        self.receiver.nextPacket()
                    except Exception:
                        pass

            # wait one timestep (CSVRobot.step() already advances the controller)
            # loop continues until Webots stops controller

# instantiate and run
if __name__ == "__main__":
    robot = SingleTurtleBot()
    robot.run()
