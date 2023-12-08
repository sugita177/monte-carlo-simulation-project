import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random
import time


def calc_potential(distance: float,
                   max_length: float,
                   bond_length: float,
                   spring_constant: float) -> float:
    return -spring_constant/2 * (max_length - bond_length)**2\
           * math.log(1 - ((distance - bond_length)/(max_length +10**(-3) - bond_length))**2)


def get_initial_state(particles_number: int, max_length: float, bond_length: float):
    points_list = [[0.0, 0.0]]
    for _ in range(particles_number):
        distance = random.uniform(-max_length + 2 * bond_length, max_length)
        angle = random.uniform(0, math.pi * 2)
        points_list.append([points_list[-1][0] + distance * math.cos(angle),
                            points_list[-1][1] + distance * math.sin(angle)])
    return np.asarray(points_list)


def proceed_mc_step(points, particles_number: int, inverse_temperature: float, max_length: float, bond_length: float, spring_constant: float):
    i = random.randint(1, particles_number-1)
    distance = random.uniform(-max_length + 2 * bond_length, max_length)
    angle = random.uniform(0, math.pi * 2)
    new_point = [points[i][0] + distance * math.cos(angle) * 10**(-1),
                 points[i][1] + distance * math.sin(angle) * 10**(-1)]
    old_distance_1 = math.sqrt((points[i][0] - points[i-1][0])**2 + (points[i][1] - points[i-1][1])**2)
    new_distance_1 = math.sqrt((new_point[0] - points[i-1][0])**2 + (new_point[1] - points[i-1][1])**2)
    old_distance_2 = math.sqrt((points[i][0] - points[i+1][0])**2 + (points[i][1] - points[i+1][1])**2)
    new_distance_2 = math.sqrt((new_point[0] - points[i+1][0])**2 + (new_point[1] - points[i+1][1])**2)
    if not (-max_length + 2*bond_length < new_distance_1 < max_length) or not (-max_length + 2*bond_length < new_distance_2 < max_length):
        return points
    diff_potential = + calc_potential(new_distance_1, max_length, bond_length, spring_constant)\
                     - calc_potential(old_distance_1, max_length, bond_length, spring_constant)\
                     + calc_potential(new_distance_2, max_length, bond_length, spring_constant)\
                     - calc_potential(old_distance_2, max_length, bond_length, spring_constant)
    if diff_potential <= 0:
        points[i] = new_point
    else:
        random_number = random.uniform(0, 1)
        if math.exp(-inverse_temperature * diff_potential) > random_number:
            points[i] = new_point

    return points


def draw_fig(points):
    plt.cla()
    x_points = np.array(points[:, 0])
    y_points = np.array(points[:, 1])
    plt.plot(x_points, y_points, marker="o")
    plt.pause(0.01)


def main():
    TEMPERATURE = 300
    BOLTZMAN_CONSTANT = 1.38 * 10**(-23)
    INVRESE_TEMPERATURE = 1 / (BOLTZMAN_CONSTANT * TEMPERATURE)
    STEP_NUMBER = 10**4
    PARTICLES_NUMBER = 50
    BOND_LENGTH = 1.0
    MAX_LENGTH = 1.2 * BOND_LENGTH
    SPRING_CONSTANT = 50 / (INVRESE_TEMPERATURE * BOND_LENGTH**2)
    points = get_initial_state(PARTICLES_NUMBER, MAX_LENGTH, BOND_LENGTH)

    for _ in range(STEP_NUMBER):
        points = proceed_mc_step(points, PARTICLES_NUMBER, INVRESE_TEMPERATURE,
                                 MAX_LENGTH, BOND_LENGTH, SPRING_CONSTANT)
        draw_fig(points)


main()
