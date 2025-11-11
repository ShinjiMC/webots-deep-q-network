# environment.py
import numpy as np

class EPuckEnv:
    def __init__(self, robot, time_step):
        self.robot = robot
        self.TIME_STEP = time_step
        self.MAX_SPEED = 6.28
        self.END_MIN = 200
        self.END_MAX = 300
        self.left_motor = robot.getMotor('left wheel motor')
        self.right_motor = robot.getMotor('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.ps = []
        ps_names = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
        for name in ps_names:
            sensor = robot.getDistanceSensor(name)
            sensor.enable(self.TIME_STEP)
            self.ps.append(sensor)
        self.gs = []
        gs_names = ['gs0', 'gs1', 'gs2']
        for name in gs_names:
            sensor = robot.getDistanceSensor(name)
            sensor.enable(self.TIME_STEP)
            self.gs.append(sensor)
        print("Clase EPuckEnv inicializada con motores y sensores.")

    def get_state(self):
        ps_values = [s.getValue() for s in self.ps]
        state = np.array(ps_values) / 1000.0 
        return state
    
    def apply_action(self, action):
        turn_speed = self.MAX_SPEED * 0.8
        if action == 0: # Avanzar
            left_speed = self.MAX_SPEED
            right_speed = self.MAX_SPEED
        elif action == 1: # Girar Izquierda
            left_speed = -turn_speed
            right_speed = turn_speed
        else: # Girar Derecha (acción 2)
            left_speed = turn_speed
            right_speed = -turn_speed
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def get_reward_and_done(self, action, state_values):
        reward = 0.0
        done = False
        floor_values = [s.getValue() for s in self.gs]
        if (self.END_MIN <= floor_values[0] <= self.END_MAX and
            self.END_MIN <= floor_values[1] <= self.END_MAX and
            self.END_MIN <= floor_values[2] <= self.END_MAX):
            reward = 1.0
            done = True
        else:
            front_obstacle_threshold = 0.1 
            front_obstacle = (state_values[0] > front_obstacle_threshold or 
                              state_values[7] > front_obstacle_threshold)
            if front_obstacle:
                reward = -1.0
            elif action == 0:
                reward = 0.1
            else:
                reward = -0.1
        return reward, done