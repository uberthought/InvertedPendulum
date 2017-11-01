from math import sin, cos, pi, sqrt, copysign, pow
from numpy import matrix, array
from control.matlab import *

import random


# original code is here https://github.com/toddsifleet/inverted_pendulum

M = .6  # mass of cart+pendulum
m = .3  # mass of pendulum
Km = 2  # motor torque constant
Kg = .01  # gear ratio
R = 6  # armiture resistance
r = .01  # drive radiu3
K1 = Km*Kg/(R*r)
K2 = Km**2*Kg**2/(R*r**2)
l = .3  # length of pendulum to CG
I = 0.006  # inertia of the pendulum
L = (I + m*l**2)/(m*l)
g = 9.81  # gravity
Vsat = 1000.  # saturation voltage

A11 = -1 * Km**2*Kg**2 / ((M - m*l/L)*R*r**2)
A12 = -1*g*m*l / (L*(M - m*l/L))
A31 = Km**2*Kg**2 / (M*(L - m*l/M)*R*r**2)
A32 = g/(L-m*l/M)
A = matrix([
    [0, 1, 0, 0],
    [0, A11, A12, 0],
    [0, 0, 0, 1],
    [0, A31, A32, 0]
])

B1 = Km*Kg/((M - m*l/L)*R*r)
B2 = -1*Km*Kg/(M*(L-m*l/M)*R*r)

B = matrix([
    [0],
    [B1],
    [0],
    [B2]
])
Q = matrix([
    [10000, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 10000, 0],
    [0, 0, 0, 1]
])

(K, X, E) = lqr(A, B, Q, R);

def cmp(x, y):
    return (x > y) - (x < y)

def constrain(theta):
    theta = theta % (2*pi)
    if theta > pi:
        theta = -2*pi+theta
    return theta

def sat(Vsat, V):
    if abs(V) > Vsat:
        return Vsat * cmp(V, 0)
    return V
 
def average(x):
    x_i, k1, k2, k3, k4 = x
    return x_i + (k1 + 2.0*(k3 + k4) +  k2) / 6.0

theta = []
class Pendulum(object):
    state_size = 6
    action_size = 3
    range = 0.01

    def random_theta():
        p = random.random() * Pendulum.range
        if random.choice([True, False]):
            return p * pi
        else:
            return (2 - p) * pi
        
    def __init__(self, initial_theta):
        # deta t
        self.dt = 0.01
        self.t = 0.0
        self.initial_theta = initial_theta

        # x, delta x, theta, delta theta
        self.x = [0, 0., self.initial_theta, 0.]

        self.state_list = [initial_theta] * Pendulum.state_size

        # max time
        self.end = 4

        # theta acceleration
        self.a = 0

        # max x position
        self.max_x = 100

    def derivative(self, u, a):
        V = sat(Vsat, a)
        #x1 = x, x2 = x_dt, x3 = theta, x4 = theta_dt
        x1, x2, x3, x4 = u
        x1_dt, x3_dt =  x2, x4
        x2_dt = (K1*V - K2*x2 - m*l*g*cos(x3)*sin(x3)/L + m*l*sin(x3)*x4**2) / (M - m*l*cos(x3)**2/L)
        x4_dt = (g*sin(x3) - m*l*x4**2*cos(x3)*sin(x3)/L - cos(x3)*(K1*V + K2*x2)/M) / (L - m*l*cos(x3)**2/M)
        x = [x1_dt, x2_dt, x3_dt, x4_dt]
        return x

    def rk4_step(self, dt, action):
        self.a = Pendulum.action_to_acceleration(action)
        dx = self.derivative(self.x, self.a)
        k2 = [ dx_i*dt for dx_i in dx ]

        xv = [x_i + delx0_i/2.0 for x_i, delx0_i in zip(self.x, k2)]
        k3 = [ dx_i*dt for dx_i in self.derivative(xv, self.a)]

        xv = [x_i + delx1_i/2.0 for x_i,delx1_i in zip(self.x, k3)]
        k4 = [ dx_i*dt for dx_i in self.derivative(xv, self.a) ]

        xv = [x_i + delx1_2 for x_i,delx1_2 in zip(self.x, k4)]
        k1 = [self.dt*i for i in self.derivative(xv, self.a)]

        self.t += dt
        self.x = list(map(average, zip(self.x, k1, k2, k3, k4)))
        self.x[2] = self.x[2] % (2 * pi)
        theta.append(constrain(self.x[2]))

        del self.state_list[0]
        self.state_list.append(self.x[2])

    def state(self):
        return np.copy(self.state_list)

    def terminal(self):
        p = 1 - abs(self.x[2] / pi - 1)
        return self.t >= self.end or abs(self.x[0]) > self.max_x or p > (Pendulum.range + 0.15)
        # return self.t >= self.end or abs(self.x[0]) > self.max_x

    def score(self):
        if abs(self.x[0]) < self.max_x:
            return pow(abs(pi - self.x[2]) / pi, 27)
        else:
            return 0

    def action_to_acceleration(action):
        if action == 0:
            return 0.0

        elif action == 1:
            return -10.0
        elif action == 2:
            return 10.0
        elif action == 3:
            return -5.0
        elif action == 4:
            return 5.0
        elif action == 5:
            return -2.0
        elif action == 6:
            return 2.0
        elif action == 7:
            return -1.0
        elif action == 8:
            return 1.0
        elif action == 9:
            return -0.5
        elif action == 10:
            return 0.5

        # elif action == 1:
        #     return -100.0
        # elif action == 2:
        #     return 100.0
        # elif action == 3:
        #     return -10.0
        # elif action == 4:
        #     return 10.0
        # elif action == 5:
        #     return -1.0
        # elif action == 6:
        #     return 1.0
        # elif action == 7:
        #     return -0.1
        # elif action == 8:
        #     return 0.1
        # elif action == 9:
        #     return -0.01
        # elif action == 10:
        #     return 0.01
