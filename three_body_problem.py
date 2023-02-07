import numpy as np
import math 

class Vector:
    '''
    Class that stores values representing a vector in a numpy array with methods 
    that are more intuitive to read than vector[0] or vector[1]
    '''
    __slots__ = ("_data")

    def __init__(self, x: float, y: float):
        self._data = np.zeros(2)
        self._data[0] = x
        self._data[1] = y
    
    def __str__(self) -> str:
        '''
        Returns a string representation of the Vector object
        Parameters:
            None
        Returns:
            a str representation of the object 
        '''
        return f"|{self._data[0]}, {self._data[1]}|"

    def __add__(self, other: "Vector") -> None:
        return Vector(self._data[0] + other._data[0], self._data[1] + other._data[1])

    def __sub__(self, other: "Vector") -> None:
        return Vector(self._data[0] - other._data[0], self._data[1] - other._data[1])
    
    def __mul__(self, other: "Vector | float") -> "Vector":
        if isinstance(other, Vector) == False:
            new_vector = self._data * other
            return Vector(new_vector[0], new_vector[1])
    
    def __truediv__(self, other: "Vector | float") -> "Vector":
        if isinstance(other, Vector) == False:
            new_vector = self._data / other
            return Vector(new_vector[0], new_vector[1])
    
    def getX(self) -> float:
        '''
        Returns the x component of the vector
        Parameters:
            None
        Returns:
            a float
        '''
        return self._data[0]

    def getY(self) -> float:
        '''
        Returns the y component of the vector
        Parameters:
            None
        Returns:
            a float
        '''
        return self._data[1]
    
    def getAll(self) -> "numpy array":
        '''
        Returns the numpy array of the vector
        Parameters:
            None
        Returns:
            a numpy array
        '''
        return self._data
    
    def updateAll(self, new_vector: "numpy array") -> None:
        '''
        Updates the array of the vector
        Parameters:
            new_vector: the new numpy array to be placed inside of the vector
        Returns:
            None
        '''
        self._data = new_vector

    def negative(self) -> "numpy array":
        '''
        Returns a numpy array with the signs of each component flipped
        Parameters:
            None
        Returns:
            a numpy array
        '''
        return np.negative(self._data)

    def getMax(self) -> float:
        '''  
        Returns the largest number in the object
        Parameters:
            None
        Returns:
            float
        '''
        if self._data[0] > self._data[1]:
            return self._data[0]
        else:
            return self._data[1]

    def getMin(self) -> float:
        '''  
        Returns the smallest number in the object
        Parameters:
            None
        Returns:
            float
        '''
        if self._data[0] < self._data[1]:
            return self._data[0]
        else:
            return self._data[1]

class Particle():
    '''
    Class that stores the position, velocity, and mass of a body in Vector forms for convenience
    '''
    __slots__ = ("_position", "_velocity", "_mass")

    def __init__(self, x_position: float, y_position: float, x_velocity: float, y_velocity: float, mass: float) -> None:
        self._position = Vector(x_position, y_position)
        self._velocity = Vector(x_velocity, y_velocity)
        self._mass     = mass

    def __str__(self) -> str:
        '''
        Returns a string representation of the Particle object
        Parameters:
            None
        Returns:
            a str
        '''
        new_str  = f"r: {self._position}, m: {self._mass}, v: {self._velocity}"
        return new_str

    def getPosition(self) -> Vector:
        '''
        Returns the position Vector of the Particle
        Parameters:
            None
        Returns:
            Vector object
        '''
        return self._position

    def getXPosition(self) -> float:
        '''
        Returns the x component of the position Vector
        Parameters:
            None
        Returns:
            a float
        '''
        return self._position._data[0]

    def getYPosition(self) -> float:
        '''
        Returns the y component of the position Vector
        Parameters:
            None
        Returns:
            a float
        '''
        return self._position._data[1]

    def updatePosition(self, new_position: Vector) -> None:
        '''
        Updates the position Vector of the Particle
        Parameters:
            new_position: a Vector object that will be the new position
        Returns:
            None
        '''
        self._position = new_position
    
    def getVelocity(self) -> Vector:
        '''
        Returns the velocity Vector of the Particle
        Parameters:
            None
        Returns:
            Vector object
        '''
        return self._velocity
    
    def updateVelocity(self, new_velocity: Vector) -> None:
        '''
        Updates the velocity Vector of the Particle
        Parameters:
            new_velocity: a Vector object that will be the new velocity
        Returns:
            None
        '''
        self._velocity = Vector(new_velocity._data[0], new_velocity._data[1])

    def getMass(self) -> float:
        '''
        Returns the mass float of the Particle
        Parameters:
            None
        Returns:
            a float
        '''
        return self._mass
    
    def getPositionMin(self) -> float:
        '''
        Returns the lowest value in the position Vector of the Particle
        Parameters:
            None
        Returns:
            float
        '''
        if self._position._data[0] < self._position._data[1]:
            return self._position._data[0]
        else:
            return self._position._data[1] 
    
    def getPositionMax(self) -> float:
        '''
        Returns the largest value in the position Vector of the Particle
        Parameters:
            None
        Returns:
            float
        '''
        if self._position._data[0] > self._position._data[1]:
            return self._position._data[0]
        else:
            return self._position._data[1] 


def getForce(p1: Particle, p2: Particle, rk4_factor: Vector) -> Vector:
    '''
    Computes the gravitational force between two Particle objects and returns a new 
    Vector object of the force on p1
    Parameters:
        p1: the first Particle 
        p2: the second Particle 
    Returns:
        a Vector object
    '''
    G = 6.67430e-11

    pos_1 = p1.getPosition() + rk4_factor
    pos_2 = p2.getPosition()

    radius = np.sqrt(((pos_1.getX() - pos_2.getX())**2) + ((pos_1.getY() - pos_2.getY())**2))

    magnitude = (G * p1.getMass() * p2.getMass()) / (radius**2)
    theta = math.atan2(pos_2.getY() - pos_1.getY(), pos_2.getX() - pos_1.getX()) #angle of the vector from p1 -> p2 

    force_vector = Vector(magnitude * np.cos(theta), magnitude * np.sin(theta))

    return force_vector

def getAcceleration(p1: Particle, p2: Particle, p3: Particle, rk4_factor: Vector = Vector(0, 0)) -> Vector:
    '''
    Calculates the force on p1 by p2 and p3 respectivly. Then finds the net force and then uses
    the mass of p1 to find the net acceleration vector for p1
    Parameters:
        p1: the Particle object that is being acted on
        p2: Particle object
        p3: Particle object
        rk4_factor: a Vector object used by rk4 to find the minor changes of the different calculations
    Returns:
        a new Vector object for the acceleration on p1
    '''
    acceleration  = getForce(p1, p2, rk4_factor) + getForce(p1, p3, rk4_factor)
    acceleration /= p1.getMass()
    return acceleration

def rk4(p1: Particle, p2: Particle, p3: Particle, t_step: float) -> "tuple[Vector]":
    '''
    Fourth order Runge-Kutta solver that finds the new position and velocity vectors for p1
    Parameters:
        p1: the Particle object that the solver is finding the change for
        p2: the first Particle acting on p1
        p3: the second Particle acting on p1
        t_step: the time difference that the method is using
    '''
    p1_position = p1.getPosition()
    p1_velocity = p1.getVelocity()

    kv_0 = p1_velocity                 #initial velocity 
    ka_0 = getAcceleration(p1, p2, p3) #initial acceleration

    kv_1 = p1_velocity + (ka_0 * 0.5 * t_step)                #first velocity step using the inital acceleration
    ka_1 = getAcceleration(p1, p2, p3, (kv_0 * 0.5 * t_step)) #first acceleration step using the inital velocity to change position

    kv_2 = p1_velocity + (ka_1 * 0.5 * t_step)
    ka_2 = getAcceleration(p1, p2, p3, (kv_1 * 0.5 * t_step))

    kv_3 = p1_velocity + (ka_2 * 0.5 * t_step)
    ka_3 = getAcceleration(p1, p2, p3, (kv_2 * 0.5 * t_step))

    kv_4 = p1_velocity + (ka_3 * 0.5 * t_step)
    ka_4 = getAcceleration(p1, p2, p3, (kv_3 * 0.5 * t_step))

    p1_position += (kv_1 + (kv_2 * 2.0) + (kv_3 * 2.0) + kv_4) * t_step / 6.0
    p1_velocity += (ka_1 + (ka_2 * 2.0) + (ka_3 * 2.0) + ka_4) * t_step / 6.0

    return p1_position, p1_velocity
 
def runSimulation(p1: Particle, p2: Particle, p3: Particle, t_total: "int | float") -> "tuple(array[array])":
    '''
    Runs the three body simulation on the three particles for the given amount of time
    Parameters:
        p1: first Particle object
        p2: second Particle object
        p3: third Particle object
        t_total: the total duration of the simulation in seconds
    Returns:
        a nested numpy array that has the x and y positions of each particle over the entire simulation
    '''

    t_step = 0.05

    total_steps = int((t_total / t_step)) + 1   #total num of new steps + the inital positions

    p1_positions = np.array([np.zeros(total_steps), np.zeros(total_steps)])   #saves the position vectors from each particle over the whole simulation
    p2_positions = np.array([np.zeros(total_steps), np.zeros(total_steps)])
    p3_positions = np.array([np.zeros(total_steps), np.zeros(total_steps)])

    p1_positions[0][0] = p1.getXPosition()
    p1_positions[1][0] = p1.getYPosition()

    p2_positions[0][0] = p2.getXPosition()
    p2_positions[1][0] = p2.getYPosition()  #puts the inital positions in their respective arrays

    p3_positions[0][0] = p3.getXPosition()
    p3_positions[1][0] = p3.getYPosition()

    for i in range(1, total_steps):
        new_p1 = rk4(p1, p2, p3, t_step)
        new_p2 = rk4(p2, p1, p3, t_step)   #uses rk4 on each particle
        new_p3 = rk4(p3, p1, p2, t_step)

        p1.updatePosition(new_p1[0])
        p1.updateVelocity(new_p1[1]) 

        p2.updatePosition(new_p2[0])
        p2.updateVelocity(new_p2[1])   #updates the position and velocity arrays inside the particle object

        p3.updatePosition(new_p3[0])
        p3.updateVelocity(new_p3[1])

        p1_positions[0][i] = new_p1[0].getX()
        p1_positions[1][i] = new_p1[0].getY()

        p2_positions[0][i] = new_p2[0].getX()
        p2_positions[1][i] = new_p2[0].getY()  #puts the new positions in their respective arrays

        p3_positions[0][i] = new_p3[0].getX()
        p3_positions[1][i] = new_p3[0].getY()
    
    
    print("Simulation is over")
    return p1_positions, p2_positions, p3_positions

import random

def randomParticles(p_param: dict) -> "list[float]":
    '''
    Generates a list of random velocities between the given min and max values
    The velocities are given to a tenth.
    '''
    v_list = []
    v_min = p_param["v_min"] * 10
    v_max = p_param["v_max"] * 10
    for i in range(6):
        new_v = random.randint(v_min, v_max)
        v_list.append(new_v / 10)

    x_list = []
    y_list = []
    x_min = p_param["x_min"] * 10; x_max = p_param["x_max"] * 10
    y_min = p_param["y_min"] * 10; y_max = p_param["y_max"] * 10

    for i in range(3):
        new_x = random.randint(x_min, x_max)
        new_y = random.randint(y_min, y_max)
        x_list.append(new_x / 10)
        y_list.append(new_x / 10)
    
    m_list = []
    m_min = p_param["m_min"]; m_max = p_param["m_max"]
    for i in range(3):
        new_m = random.randint(m_min, m_max)
        if p_param["same_mass"] and i == 0:
            m_list.append(new_m)
        elif p_param["same_mass"] and i != 0:
            m_list.append(m_list[0])
        else:
            m_list.append(new_m)

    p1 = Particle(x_list[0], y_list[0], v_list[0], v_list[1], m_list[0])
    p2 = Particle(x_list[1], y_list[1], v_list[2], v_list[3], m_list[1])
    p3 = Particle(x_list[2], y_list[2], v_list[4], v_list[5], m_list[2])

    return p1, p2, p3

'''v is in m/s, x and y are in m, mass is in kg'''

random_parameters = {
    "v_min": 0,
    "v_max": 0,
    "x_min": -50,
    "x_max": 50,
    "y_min": -50,
    "y_max": 50,
    "m_min": 5e10,
    "m_max": 5e12,
    "same_mass": True
}
    
    

random_p = randomParticles(random_parameters)
b1 = random_p[0]
b2 = random_p[1]
b3 = random_p[2]

print(b1)
print(b2)
print(b3)

#You can set the parameters manually here by uncommenting commenting the random section
b1 = Particle(10, 10, -1.7, 0.5, 5e12)
b2 = Particle(10, -10, 3, -0.2, 5e12)
b3 = Particle(-10, 0, -2.1, 1.7, 5e12)

results = runSimulation(b1, b2, b3, 25)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

p1_x = results[0][0]
p1_y = results[0][1]

p2_x = results[1][0]
p2_y = results[1][1]

p3_x = results[2][0]
p3_y = results[2][1]

metadata = dict(title="Movie", artist="Phasma")
writer = animation.PillowWriter(fps=15, metadata=metadata)

fig = plt.figure()
fig.set_size_inches(5, 5)

with writer.saving(fig, "three_body_simulation.gif", 100): #title of the gif can be changed here
    for i in range(len(p1_x)):
        
        plt.xlabel("meters")
        plt.ylabel("meters")
        plt.plot(p1_x[0:i+1], p1_y[0:i+1], marker="None", color="red", label="p1") 
        plt.plot(p2_x[0:i+1], p2_y[0:i+1], marker="None", color="green", label="p2")
        plt.plot(p3_x[0:i+1], p3_y[0:i+1], marker="None", color="blue", label="p3")


        plt.plot(p1_x[i], p1_y[i], marker=".", color="red") 
        plt.plot(p2_x[i], p2_y[i], marker=".", color="green")
        plt.plot(p3_x[i], p3_y[i], marker=".", color="blue")
        writer.grab_frame()
        plt.clf()
