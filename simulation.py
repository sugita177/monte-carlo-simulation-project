import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random
import time


def fene_potential(bond_length: float,
                   max_length: float,
                   natural_bond_length: float,
                   spring_constant: float) -> float:
    return -spring_constant/2 * (max_length - bond_length)**2\
           * math.log(1 - ((bond_length - natural_bond_length)/(max_length +10**(-3) - natural_bond_length))**2)

def morse_potential():
    return 0.0

def bond_angle_potential(bond_angle: float, bond_angle_potential_const: float) -> float:
    return bond_angle_potential_const * (1.0 - math.cos(bond_angle))



def get_initial_state(particles_number: int, max_length: float, bond_length: float):
    points_list = [[0.0, 0.0]]
    for _ in range(particles_number-1):
        distance = random.uniform(-max_length + 2 * bond_length, max_length)
        angle = random.uniform(0, math.pi * 2)
        points_list.append([points_list[-1][0] + distance * math.cos(angle),
                            points_list[-1][1] + distance * math.sin(angle)])
    return np.asarray(points_list)


def proceed_mc_step(points, particles_number: int, inverse_temperature: float, max_length: float, natural_bond_length: float, spring_constant: float, bond_angle_potential_const: float):
    diff_potential = 0.0
    i = random.randint(0, particles_number-1)

    if i == 0:
        bond_length = random.uniform(-max_length + 2 * natural_bond_length, max_length)
        angle = random.uniform(0, math.pi * 2)
        old_point = np.array(points[i])
        new_point = np.array([points[i][0] + bond_length * math.cos(angle) * 10**(-1), points[i][1] + bond_length * math.sin(angle) * 10**(-1)])
        old_bond_vector_2 = np.array([points[i+1][0] - old_point[0], points[i+1][1] - old_point[1]])
        new_bond_vector_2 = np.array([points[i+1][0] - new_point[0], points[i+1][1] - new_point[1]])

        old_bond_length_2 = np.linalg.norm(old_bond_vector_2)
        new_bond_length_2 = np.linalg.norm(new_bond_vector_2)

        if not (-max_length + 2*natural_bond_length < new_bond_length_2 < max_length):
            return points

        diff_potential = + fene_potential(new_bond_length_2, max_length, natural_bond_length, spring_constant)\
                         - fene_potential(old_bond_length_2, max_length, natural_bond_length, spring_constant)


        
    elif i == particles_number-1:
        bond_length = random.uniform(-max_length + 2 * natural_bond_length, max_length)
        angle = random.uniform(0, math.pi * 2)
        old_point = np.array(points[i])
        new_point = np.array([points[i][0] + bond_length * math.cos(angle) * 10**(-1), points[i][1] + bond_length * math.sin(angle) * 10**(-1)])
        old_bond_vector_1 = np.array([old_point[0] - points[i-1][0], old_point[1] - points[i-1][1]])
        new_bond_vector_1 = np.array([new_point[0] - points[i-1][0], new_point[1] - points[i-1][1]])

        old_bond_length_1 = np.linalg.norm(old_bond_vector_1)
        new_bond_length_1 = np.linalg.norm(new_bond_vector_1)

        if not (-max_length + 2*natural_bond_length < new_bond_length_1 < max_length):
            return points
        
        diff_potential = + fene_potential(new_bond_length_1, max_length, natural_bond_length, spring_constant)\
                         - fene_potential(old_bond_length_1, max_length, natural_bond_length, spring_constant)
                         

    else:
        bond_length = random.uniform(-max_length + 2 * natural_bond_length, max_length)
        angle = random.uniform(0, math.pi * 2)
        old_point = np.array(points[i])
        new_point = np.array([points[i][0] + bond_length * math.cos(angle) * 10**(-1), points[i][1] + bond_length * math.sin(angle) * 10**(-1)])
        old_bond_vector_1 = np.array([old_point[0] - points[i-1][0], old_point[1] - points[i-1][1]])
        new_bond_vector_1 = np.array([new_point[0] - points[i-1][0], new_point[1] - points[i-1][1]])
        old_bond_vector_2 = np.array([points[i+1][0] - old_point[0], points[i+1][1] - old_point[1]])
        new_bond_vector_2 = np.array([points[i+1][0] - new_point[0], points[i+1][1] - new_point[1]])

        old_bond_length_1 = np.linalg.norm(old_bond_vector_1)
        new_bond_length_1 = np.linalg.norm(new_bond_vector_1)
        old_bond_length_2 = np.linalg.norm(old_bond_vector_2)
        new_bond_length_2 = np.linalg.norm(new_bond_vector_2)

        old_bond_angle = np.dot(old_bond_vector_1, old_bond_vector_2) / (old_bond_length_1 * old_bond_length_2)
        new_bond_angle = np.dot(new_bond_vector_1, old_bond_vector_2) / (new_bond_length_1 * old_bond_length_2)

        if not (-max_length + 2*natural_bond_length < new_bond_length_1 < max_length) or not (-max_length + 2*natural_bond_length < new_bond_length_2 < max_length):
            return points
        diff_potential = + fene_potential(new_bond_length_1, max_length, natural_bond_length, spring_constant)\
                         - fene_potential(old_bond_length_1, max_length, natural_bond_length, spring_constant)\
                         + fene_potential(new_bond_length_2, max_length, natural_bond_length, spring_constant)\
                         - fene_potential(old_bond_length_2, max_length, natural_bond_length, spring_constant)\
                         + bond_angle_potential(new_bond_angle, bond_angle_potential_const)\
                         - bond_angle_potential(old_bond_angle, bond_angle_potential_const)
    
    if diff_potential <= 0:
        points[i] = new_point.tolist()
    else:
        random_number = random.uniform(0, 1)
        if math.exp(-inverse_temperature * diff_potential) > random_number:
            points[i] = new_point.tolist()

    return points


def draw_fig(points):
    plt.cla()
    x_points = np.array(points[:, 0])
    y_points = np.array(points[:, 1])
    plt.plot(x_points, y_points, marker="o")
    plt.pause(0.01)


def main():
    TEMPERATURE = 300.0
    BOLTZMAN_CONSTANT = 1.38 * 10**(-23)
    INVRESE_TEMPERATURE = 1.0 / (BOLTZMAN_CONSTANT * TEMPERATURE)
    STEP_NUMBER = 10**3
    PARTICLES_NUMBER = 50
    BOND_LENGTH = 1.0
    MAX_LENGTH = 1.2 * BOND_LENGTH
    SPRING_CONSTANT = 50.0 / (INVRESE_TEMPERATURE * BOND_LENGTH**2)
    BOND_ANGLE_POTENTIAL_CONST = SPRING_CONSTANT * 1.2
    points = get_initial_state(PARTICLES_NUMBER, MAX_LENGTH, BOND_LENGTH)

    for _ in range(STEP_NUMBER):
        for _ in range(PARTICLES_NUMBER):
            points = proceed_mc_step(points, PARTICLES_NUMBER, INVRESE_TEMPERATURE,
                                 MAX_LENGTH, BOND_LENGTH, SPRING_CONSTANT, BOND_ANGLE_POTENTIAL_CONST)
        draw_fig(points)


main()
