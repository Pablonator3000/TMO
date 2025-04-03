import numpy as np
import matplotlib.pyplot as plt
import simpy
import math
import pandas as pd


def erlang_b(lambda_, mu, n):
    """Вычисление вероятности простоя P0 (формула Эрланга B)"""
    alpha = (lambda_ / mu)
    sum_ = sum(alpha ** k / math.factorial(k) for k in range(n + 1)) + ((alpha ** n) / (math.factorial(n)*(1-alpha/n)))
    return 1 / sum_


def mmn_characteristics(lambda_, mu, n):
    """Расчет характеристик системы M/M/n/∞"""
    P0 = erlang_b(lambda_, mu, n)  # Вероятность простоя
    alpha = lambda_ / mu
    rho = lambda_ / (n * mu)  # Коэффициент загрузки
    Pw = ((alpha ** (n + 1)) / (math.factorial(n) * (n - alpha))) * P0  # Вероятность ожидания
    Lq = (lambda_ / mu) ** n * P0 / (math.factorial(n) * (1 - rho) ** 2)  # Среднее число заявок в очереди
    Wq = Lq / lambda_  # Среднее время ожидания заявки в очереди
    W = Wq + 1 / mu  # Среднее время пребывания заявки в системе

    return P0, Pw, Lq, Wq, W, rho


def simulate_queue(env, lambda_, mu, n, results):
    """Имитационная модель системы M/M/n/∞"""
    server = simpy.Resource(env, capacity=n)
    queue_lengths = []
    waiting_times = []

    def customer(env, name):
        arrival_time = env.now
        with server.request() as req:
            yield req
            waiting_times.append(env.now - arrival_time)
            yield env.timeout(np.random.exponential(1 / mu))
        queue_lengths.append(len(server.queue))

    def arrival_process(env):
        while True:
            yield env.timeout(np.random.exponential(1 / lambda_))
            env.process(customer(env, f'Customer-{env.now}'))

    env.process(arrival_process(env))
    env.run(until=1000)

    results['avg_queue_length'] = np.mean(queue_lengths)
    results['avg_waiting_time'] = np.mean(waiting_times)



lambda_ = 10  # Интенсивность потока заявок (заявок в час)
mu = 3  # Интенсивность обслуживания (заявок в час на 1 канал)
n_values = range(4, 10)  # Различные количества каналов


results = [mmn_characteristics(lambda_, mu, n) for n in n_values]
P0_values, Pw_values, Lq_values, Wq_values, W_values, rho_values = zip(*results)


sim_results = {}
for n in n_values:
    env = simpy.Environment()
    sim_results[n] = {}
    simulate_queue(env, lambda_, mu, n, sim_results[n])


data = []
for i, n in enumerate(n_values):
    data.append([
        n, P0_values[i], Pw_values[i], Lq_values[i], Wq_values[i], W_values[i], rho_values[i],
        sim_results[n]['avg_queue_length'], sim_results[n]['avg_waiting_time']
    ])

df = pd.DataFrame(data, columns=[
    'n', 'Вероятность простоя', 'Вероятность ожидания', 'Ср.длина очереди', 'Ср.время ожидания', 'Ср. время пребывания', 'Кф. загрузки',
    'Ср.длина(моделирование)', 'Ср.время(моделирование)'
])
pd.set_option('display.max_columns', None)  # Показывать все столбцы
pd.set_option('display.width', 1000)        # Увеличить ширину вывода
print(df)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(n_values, Wq_values, marker='o', linestyle='-', label='Теоретическое время')
plt.plot(n_values, [sim_results[n]['avg_waiting_time'] for n in n_values], marker='x', linestyle='--',
         label='Имитационное время')
plt.xlabel('Количество каналов (n)')
plt.ylabel('Среднее время ожидания (часы)')
plt.title('Зависимость ср.ожидания от n')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_values, Lq_values, marker='s', linestyle='-', label='Теоретическая длина')
plt.plot(n_values, [sim_results[n]['avg_queue_length'] for n in n_values], marker='d', linestyle='--',
         label='Имитационная длина')
plt.xlabel('Количество каналов (n)')
plt.ylabel('Средняя длина очереди')
plt.title('Зависимость длины очереди от n')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
