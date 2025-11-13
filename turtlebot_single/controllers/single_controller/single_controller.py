from deepbots.robots import CSVRobot
import numpy as np

class SingleTurtleBot(CSVRobot):
    def __init__(self):
        super().__init__(emitter_name='emitter', receiver_name='receiver')

        # Get motors
        self.left_motor = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        try:
            self.lidar = self.getDevice('LDS-01')
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
            print("LIDAR initialized")
        except Exception as e:
            self.lidar = None
            print("No LiDAR found:", e)

        

    def create_message(self):
        """Send LiDAR distances to supervisor"""
        if self.lidar:
            ranges = self.lidar.getRangeImage()
            # Pick 8 samples evenly spaced across the scan
            n = len(ranges)
            sampled = [ranges[i] for i in np.linspace(0, n - 1, 8, dtype=int)]
        else:
            sampled = [0.0] * 8

        msg = [0.0] + sampled  #modify this line!!!!
        if self.emitter:
            self.emitter.send(','.join(map(str, msg)).encode('utf-8'))
        return msg

    def use_message_data(self, message):
        if not message or len(message) < 2:
            return

        try:
            v_bx = float(message[1])  # forward speed
            omega_bz = float(message[0])  # body angular velocity
        except (ValueError, TypeError) as e:
            print("Invalid message:", message, "error:", e)
            return

        if (self.timestep%10 == 0):
            print("Message received:", message)

        # Robot parameters
        w = 0.16  #distance between wheels

        # Inverse kinematics for differential drive
        left_speed = v_bx - omega_bz * w / 2
        right_speed = v_bx + omega_bz * w / 2

        # Clamp velocities to motor limits
        max_vel = 6.67
        left_speed = np.clip(left_speed, -max_vel, max_vel)
        right_speed = np.clip(right_speed, -max_vel, max_vel)

        if (self.timestep%10 == 0):
            print(f"Setting velocities -> left: {left_speed}, right: {right_speed}")

        # Apply to motors
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        


robot = SingleTurtleBot()
robot.run()
