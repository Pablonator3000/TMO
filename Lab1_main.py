import simpy
import numpy as np
import matplotlib.pyplot as plt


class MM1Queue:
    def __init__(self, env, lambda_, mu):
        self.env = env
        self.server = simpy.Resource(env, capacity=1)  # Одноканальная система
        self.lambda_ = lambda_
        self.mu = mu
        self.arrived = 0
        self.served = 0
        self.lost = 0

    def arrival_process(self):
        while True:
            self.arrived += 1  # Новая заявка
            if self.server.count == 0:  # Если сервер свободен
                self.env.process(self.service_process())
            else:
                self.lost += 1  # Заявка теряется
            yield self.env.timeout(np.random.exponential(1 / self.lambda_))

    def service_process(self):
        with self.server.request():
            self.served += 1
            yield self.env.timeout(np.random.exponential(1 / self.mu))


# Параметры системы
lambda_ = 5
mu = 6
simulation_time = 1000

# Запуск моделирования
env = simpy.Environment()
queue = MM1Queue(env, lambda_, mu)
env.process(queue.arrival_process())
env.run(until=simulation_time)

# Результаты моделирования
p_rejection = queue.lost / queue.arrived  # Вероятность отказа
rho = queue.served / queue.arrived  # Коэффициент загрузки

print(f"Поступило заявок: {queue.arrived}")
print(f"Обслужено заявок: {queue.served}")
print(f"Потеряно заявок: {queue.lost}")
print(f"Вероятность отказа: {p_rejection:.4f}")
print(f"Коэффициент загрузки: {rho:.4f}")
