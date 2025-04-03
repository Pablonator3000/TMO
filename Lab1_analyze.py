import simpy
import numpy as np
import matplotlib.pyplot as plt
from Lab1_main import MM1Queue


def run_experiment(lambda_, mu, simulation_time):
    env = simpy.Environment()
    queue = MM1Queue(env, lambda_, mu)
    env.process(queue.arrival_process())
    env.run(until=simulation_time)

    p_rejection = queue.lost / queue.arrived if queue.arrived > 0 else 0
    rho = queue.served / queue.arrived if queue.arrived > 0 else 0

    return queue.arrived, queue.served, queue.lost, p_rejection, rho


# Параметры экспериментов
mu = 6  # Фиксированная интенсивность обслуживания
lambda_values = np.linspace(1, 10, 10)  # Разные значения lambda
simulation_time = 1000

results = []
for lambda_ in lambda_values:
    results.append(run_experiment(lambda_, mu, simulation_time))

# Разбираем результаты
arrived_list, served_list, lost_list, p_rejection_list, rho_list = zip(*results)

# График вероятности отказа
plt.figure(figsize=(8, 5))
plt.plot(lambda_values, p_rejection_list, marker='o', linestyle='-', label='Имитация')
plt.plot(lambda_values, lambda_values / (lambda_values + mu), linestyle='--', label='Теория (Эрланг)')
plt.xlabel('Интенсивность поступления заявок (λ)')
plt.ylabel('Вероятность отказа')
plt.legend()
plt.title('Зависимость вероятности отказа от λ при u = 6')
plt.grid()
plt.show()

