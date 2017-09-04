from math import sin, cos, pi
from numpy import matrix, array
from control.matlab import *

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
Vsat = 20.  # saturation voltage

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
    state_size = 4
    action_size = 7

    def __init__(self):
        # deta t
        self.dt = 0.01
        self.t = 0.0

        # x, delta x, theta, delta theta
        # self.x = [0, 0., pi, 0.]

        # start with pendulum at highest point
        self.x = [0, 0., 0, 0.]

        # start with pendulum at lowest point
        # self.x = [0, 0., pi, 0.]
        self.end = 10

        # theta acceleration
        self.a = 0

        # max x position
        self.max_x = 10

    def derivative(self, u, a):
        V = sat(Vsat, a)
        #x1 = x, x2 = x_dt, x3 = theta, x4 = theta_dt
        x1, x2, x3, x4 = u
        x1_dt, x3_dt =  x2, x4
        x2_dt = (K1*V - K2*x2 - m*l*g*cos(x3)*sin(x3)/L + m*l*sin(x3)*x4**2) / (M - m*l*cos(x3)**2/L)
        x4_dt = (g*sin(x3) - m*l*x4**2*cos(x3)*sin(x3)/L - cos(x3)*(K1*V + K2*x2)/M) / (L - m*l*cos(x3)**2/M)
        x = [x1_dt, x2_dt, x3_dt, x4_dt]
        return x

    def rk4_step(self, dt, a):
        dx = self.derivative(self.x, a)
        k2 = [ dx_i*dt for dx_i in dx ]

        xv = [x_i + delx0_i/2.0 for x_i, delx0_i in zip(self.x, k2)]
        k3 = [ dx_i*dt for dx_i in self.derivative(xv, a)]

        xv = [x_i + delx1_i/2.0 for x_i,delx1_i in zip(self.x, k3)]
        k4 = [ dx_i*dt for dx_i in self.derivative(xv, a) ]

        xv = [x_i + delx1_2 for x_i,delx1_2 in zip(self.x, k4)]
        k1 = [self.dt*i for i in self.derivative(xv, a)]

        self.t += dt
        self.x = list(map(average, zip(self.x, k1, k2, k3, k4)))
        self.x[2] = self.x[2] % (2 * pi)
        theta.append(constrain(self.x[2]))

    def state(self):
        return self.x

    def terminal(self):
        return self.t >= self.end or abs(self.x[0]) > self.max_x

    def score(self):
        if abs(self.x[0]) <= self.max_x:
            return cos(self.x[2])
        else:
            return -1.1
