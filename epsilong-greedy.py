import random
import matplotlib.pyplot as plt
import math

means = [0.1, 0.5, 0.8, 0.7, 0.65]
n_arms = len(means)

def Bernouli(mean):
    reward = []
    for i in range(len(mean)):
        if random.random() > mean[i]:
            reward.append(0.0)
        else:
            reward.append(1.0)
    return reward

def epsilon_algorithm(means, num_sims, horizon, n_arms, flag):
    if flag == 0:
        epsilon = 0.1
    cumulative_rewards = [0.0 for i in range(num_sims * horizon)]
    for sim in range(num_sims):
        counts = [0 for col in range(n_arms)]
        values = [0.0 for col in range(n_arms)]
        sim = sim + 1
        for t in range(horizon):
            t = t + 1
            if flag == 1:
                a = float(t)
                epsilon = 1.0/a
            index = (sim -1) * horizon + t - 1 
            if random.random() > epsilon:
                chosen_arm = values.index(max(values))
            else:
                chosen_arm =  random.randrange(len(values))  
            result = Bernouli(means)
            reward = result[chosen_arm]
            if t == 1:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward
            counts[chosen_arm] = counts[chosen_arm] + 1
            n = counts[chosen_arm]
            value = values[chosen_arm]
            new_value = ((n-1) / float(n)) * value + (1 / float(n)) * reward 
            values[chosen_arm] = new_value

    return cumulative_rewards

def UCB_algorithm(means, num_sims, horizon, n_arms):
    cumulative_rewards = [0.0 for i in range(num_sims * horizon)]
    a = 2
    for sim in range(num_sims):
        counts = [0 for col in range(n_arms)]
        u = [0 for col in range(n_arms)]
        values = [0.0 for col in range(n_arms)]
        arg = [0.0 for col in range(n_arms)]
        sim = sim + 1
        for t in range(horizon):
            t = t + 1
            index = (sim -1) * horizon + t - 1
            if t<6:
                chosen_arm = t-1
                counts[chosen_arm] = counts[chosen_arm] + 1
                result = Bernouli(means)
                reward = result[chosen_arm]
                values[chosen_arm] += reward
                u[chosen_arm] = values[chosen_arm] / counts[chosen_arm]
                arg[chosen_arm] = u[chosen_arm] + math.sqrt(a*math.log(t)/(2*counts[chosen_arm]))
            else:
                chosen_arm = arg.index(max(arg))
                counts[chosen_arm] = counts[chosen_arm] + 1
                result = Bernouli(means)
                reward = result[chosen_arm]
                values[chosen_arm] += reward
                u[chosen_arm] = values[chosen_arm] / counts[chosen_arm]
                arg[chosen_arm] = u[chosen_arm] + math.sqrt(a*math.log(t)/(2*counts[chosen_arm]))
            if t == 1:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward            
    return cumulative_rewards
results= epsilon_algorithm(means, 100, 1000, n_arms, 0)
results_1 = epsilon_algorithm(means, 100, 1000, n_arms, 1)
results_2 = UCB_algorithm(means, 100, 1000, n_arms)
result = [0.0 for i in range(1000)]
result_1 = [0.0 for i in range(1000)]
result_2 = [0.0 for i in range(1000)]
for i in range(100):
    a = 1000*i
    for j in range(1000):
        b = a + j
        result[j] = result[j] + (results[b])
        result_1[j] = result_1[j] + results_1[b]
        result_2[j] = result_2[j] + results_2[b]
for i in range(1000):
    result[i] = result[i]/((i+1)*100)
    result_1[i] = result_1[i]/((i+1)*100)
    result_2[i] = result_2[i]/((i+1)*100)

f, (ax1) = plt.subplots(1, 1)
plt.plot(result, label="Fixed Epsilon Greedy")
plt.plot(result_1, label="Flexible Epsilon Greedy")
plt.plot(result_2, label="UCB")
plt.xlabel('Horizon T')
plt.ylabel('Reward proportion')
plt.legend()
plt.show()

