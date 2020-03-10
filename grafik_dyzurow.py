import numpy as np
import os
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)

#   Reading data
data = np.genfromtxt('input/test.csv', usecols=(1, 2, 3, 4, 5, 6), skip_header=1, delimiter=',', dtype=str)

############################################################
#   Parameters and initializations
############################################################

#   algorithm parameters
gamma = 0.005
prob_of_changing_row = 0.1
prob_of_changing_row_best = 0.5
NI = 1000
iterration = 0
curr_best_penalty = 10000

breaks_lengths = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
ND = 5              # number of days
NB = 10             # number of breaks
NT = data.shape[0]  # number of teachers
NP = 5            # number of places
NF = 15             # number of fireflies

solutions = np.zeros((ND * NB, NP, NF), dtype=int)      # initializing solutions array
best_solution = np.zeros((ND * NB, NP), dtype=int)      # initializing solutions array
preferences = np.zeros((NT, NB * ND), dtype=bool)       # initializing teacher preferences array
teachers_time_limits = data[:, 5].copy().astype(int)    # creating teachers time limits vector

#   Creating vector with breaks duration
breaks_durations = np.ones(NB, dtype=int) * breaks_lengths
breaks_durations = np.tile(breaks_durations, ND)
breaks_durations = breaks_durations[:, np.newaxis]

############################################################
#   Creating preferences array
############################################################

for teacher in np.arange(NT):
    teacher_pref = []
    for day in np.arange(ND):
        day_pref_chars = data[teacher, day].split('-')
        for idx, el in enumerate(day_pref_chars):
            if el == '':
                day_pref_chars[idx] = '-1'
        day_pref_ints = [int(el) for el in day_pref_chars]
        teacher_pref.append(day_pref_ints)

    for idx, el in enumerate(teacher_pref):
        offset = idx * NB
        if len(el) < 2:
            continue
        for i in range(preferences.shape[1]):
            if el[0] <= i <= el[1]:
                preferences[teacher, i + offset] = 1

############################################################
#   Creating initial solutions
############################################################

for i in np.arange(NF):
    for j in np.arange(solutions.shape[0]):
        row = np.random.choice(NT, size=NP, replace=False)
        solutions[j, :, i] = row

############################################################
#   Checking preferences constraints and time limits constraints
############################################################


def checkConstraints(n, best=False):
    if not best:
        solution = solutions[:, :, n]
    else:
        solution = best_solution
    time_limits = teachers_time_limits.copy()
    penalty = 0
    for timeslot in np.arange(solution.shape[0]):
        for teacher in solution[timeslot]:
            if not preferences[teacher, timeslot]:
                penalty += 1
            time_limits[teacher] -= breaks_durations[timeslot % NB]
    time_reached = np.where(time_limits < 0, 1, 0)
    penalty += np.count_nonzero(time_reached)
    return penalty

############################################################
#   Calculating distance
############################################################


def calculateDistanceOld(s1, s2):
    s1_copy = s1.copy()
    s1_copy -= s2
    distance = np.count_nonzero(s1_copy)
    return distance, np.argwhere(s1_copy != 0)


def calculateDistance(s1, s2):
    repeated_array = []
    distance = 0
    for row in np.arange(s1.shape[0]):
        repeated_teachers = np.isin(s1[row, :], s2[row, :])
        unique_idx = np.argwhere(repeated_teachers == 0)
        repeated_array.append(unique_idx.flatten().tolist())
        distance += np.count_nonzero(repeated_teachers == 0)
    return distance, repeated_array


############################################################
#   Calculating Beta (atractiveness modification)
############################################################


def calcBeta(x):
    beta = np.exp(-1*gamma*x)
    return beta


############################################################
#   Modify solution
############################################################


def modifySolution(m, n):
    for row in np.arange(solutions.shape[0]):
        for idx in np.arange(len(darker_args[row])):
            if len(darker_args[row]) <= idx or len(brighter_args[row]) <= idx:
                break
            rng = np.random.rand()
            if rng < beta:
                k = darker_args[row][idx]
                l = brighter_args[row][idx]
                if solutions[row, l, n] not in solutions[row, :, m]:
                    solutions[row, k, m] = solutions[row, l, n].copy()
# def modifySolution(m, n):
#     for row in np.arange(solutions.shape[0]):
#             rng = np.random.rand()
#             if rng < beta:
#                 solutions[row, :, m] = solutions[row, :, n].copy()


############################################################
#   Generate random vector
############################################################


def addRandomVector(s, best=False):
    if best:
        prob = prob_of_changing_row_best
    else:
        prob = prob_of_changing_row
    rand_vec = np.random.randint(0, NT, solutions.shape[0])
    for row in np.arange(solutions.shape[0]):
        rng = np.random.rand()
        if rng > prob:
            continue
        while rand_vec[row] in solutions[row, :, s]:
            rand_vec[row] = np.random.randint(NT)
        rand_idx = np.random.randint(NP)
        solutions[row, rand_idx, s] = rand_vec[row]


def findBestSolution():
    penalty_vec = np.ones(solutions.shape[0], dtype=int) * 10000
    for firefly in np.arange(NF):
        si_penalty = checkConstraints(firefly)
        penalty_vec[firefly] = si_penalty
    best_idx = np.argmin(penalty_vec)
    return best_idx


def generatePenaltyVect():
    penalty_vec = np.ones(NF, dtype=int) * 10000
    for firefly in np.arange(NF):
        si_penalty = checkConstraints(firefly)
        penalty_vec[firefly] = si_penalty
    best_idx = np.argmin(penalty_vec)
    worst_idx = np.argmax(penalty_vec)
    return best_idx, worst_idx, penalty_vec


def saveResults():
    teacher = ['' for i in np.arange(ND)]
    results = [teacher.copy() for i in np.arange(NT)]
    cum_time = np.zeros(NT, dtype=int)
    sum_breaks = np.zeros(NT, dtype=int)
    os.remove('output/test_out.csv')

    results_map = np.zeros((NT, NB * ND), dtype=bool)

    for i in np.arange(ND*NB):
        day = i // 10
        break_id = i % 10
        for j in np.arange(NP):
            teacher_id = best_solution[i, j]
            results[teacher_id][day] += str(break_id)
            sum_breaks[teacher_id] += 1
            cum_time[teacher_id] += breaks_lengths[break_id]

            results_map[teacher_id, i] = True



    with open('output/test_out.csv', 'a') as file:
        for idx, line in enumerate(results):
            file.write('%d,p%s,w%s,s%s,c%s,p%s,%d,%d\n' % (idx, line[0], line[1], line[2], line[3], line[4], sum_breaks[idx], cum_time[idx]))


best_penalties = []
temp_penalties = []

print(solutions.shape)
while iterration < NI:
    # solutions = solutions.copy()
    best_firefly, worst_firefly, penalty_vec = generatePenaltyVect()
    temp_best_penalty = penalty_vec[best_firefly]
    if temp_best_penalty < curr_best_penalty:
        best_solution = solutions[:, :, best_firefly].copy()
        curr_best_penalty = temp_best_penalty
    else:
        solutions[:, :, best_firefly] = best_solution
    for i in np.arange(NF):
        for j in np.arange(NF):
            if penalty_vec[j] < penalty_vec[i]:
                dist, darker_args = calculateDistance(solutions[:, :, i], solutions[:, :, j])
                dist, brighter_args = calculateDistance(solutions[:, :, j], solutions[:, :, i])
                beta = calcBeta(dist)
                modifySolution(i, j)
                new_penalty = checkConstraints(i)
                penalty_vec[i] = new_penalty
                if new_penalty < curr_best_penalty:
                    best_firefly = i
                    best_solution = solutions[:, :, best_firefly].copy()
                    curr_best_penalty = new_penalty
                addRandomVector(i)
    addRandomVector(best_firefly, best=True)
    iterration += 1
    print('i = %d,  lowest_penalty = %d, temp_best_penalty = %d, best_firefly = %d' % (iterration, curr_best_penalty, temp_best_penalty, best_firefly))
    best_penalties.append(curr_best_penalty)
    temp_penalties.append(temp_best_penalty)


plt.subplot(1, 1, 1)
t = np.arange(NI)
plt.plot(t, best_penalties)
plt.xlabel('iteracja')
plt.ylabel('kara najlepszego rozwiązania')
print(best_solution)
# plt.subplot(2, 1, 2)
# plt.plot(t, temp_penalties)
# plt.xlabel('iteracja')
# plt.ylabel('kara iteracji')

plt.show()

saveResults()
