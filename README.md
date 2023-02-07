# Three-Body-Problem-Python
### Introduction:

The three body problem is the issue of showing the movement of three gravitational bodies interacting with one another at the same time. Each body in the system has two forces of different magnitudes and directions acting on it by the others. The three body problem is well known because it has no general closed-form solution, i.e can be solved with a specific number of steps each time it is run, and because the systems that result tend to be chaotic for most initial values.

There are non-chaotic systems found within three body systems, but all of them are special case solutions found by either restricting the problem to simply it, or found through numerical methods. Euler's three body problem solutions model the motion of an outer particle while the two central ones are held stationary. The three families found by Euler, and a fourth family found by Lagrange, are the only solutions to the problem that have an exact solution.

Because of the lack of a general solution for most cases and sensitivity to initial conditions, the problem is very interesting to model as it requires a differential equation solver and has pleasing visual results.


### Methods

The acceleration on each object is given by the equations

$a_1 = -\frac{Gm_2}{(r_1-r_2)^2}-\frac{Gm_3}{(r_1-r_3)^2}$

$a_2 = -\frac{Gm_1}{(r_2-r_1)^2}-\frac{Gm_3}{(r_2-r_3)^2}$

$a_3 = -\frac{Gm_1}{(r_3-r_1)^2}-\frac{Gm_2}{(r_3-r_2)^2}$

Where $a_x$​ is the vector acceleration on body $x$, and $r_x$ is the vector position of body $x$

Solving these equations requires solving for the distance between their positions, the acceleration, and the change to their velocities in each time step. Using a fourth order Runge-Kutta algorithm, the change to their position and velocities can be found with an error of O(h5)O(h5).

To do this, a Vector and Object class were created in order to store the vectors for the position, and velocity of each body, along with the mass, in their respective objects. This allows for simpler tracking of these variables throughout the program. It also allows for custom methods to be made that simplify the code and make it more readable. Then the vector of the acceleration between each body is found with Newton's law of gravitation, and then summed together for the equations above. With the Runge-Kutta method this is evaluated four times with minor changes to the position and velocities to take a weighted average of the differences.

Inside of the Runge-Kutta solver, there are separate k variables for the velocities and acceleration at each trial position. In effect, it is the same as having the function gg being a derivative of ff since the acceleration is the derivative of velocity. Afterwards, the weighted average of the velocities and acceleration are used with the time step in order to find the new position and velocities of the particle.
