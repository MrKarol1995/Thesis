import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.integrate import solve_ivp
import numpy as np

# Parametry procesu
theta = 0.4 # Szybkość powrotu
mu = 0.0     # Średnia stacjonarna
sigma = 0.3  # Intensywność fluktuacji
x0 = 1.0     # Wartość początkowa
T = 10.0     # Całkowity czas symulacji
dt = 0.01   # Długość kroku czasowego
n_paths = 10  # Liczba trajektorii

# Parametry ARMA
phi = 0.8    # Parametr AR
theta_ma = -0.5  # Parametr MA
arma_points = int(T / dt)  # Liczba punktów ARMA

def simulate_ou_process(theta, mu, sigma, x0, T, dt, n_paths):
    """
    Symuluje trajektorie procesu Ornsteina-Uhlenbecka.
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0

    for i in range(n_steps):
        dw = np.sqrt(dt) * np.random.randn(n_paths)
        paths[:, i + 1] = paths[:, i] + theta * (mu - paths[:, i]) * dt + sigma * dw

    return t, paths

def simulate_arma_process(phi, theta, sigma, n_points):
    """
    Symuluje trajektorie modelu ARMA(1,1).
    """
    ar = np.r_[1, -phi]  # Współczynniki AR (autoregresja)
    ma = np.r_[1, theta]  # Współczynniki MA (średnia ruchoma)
    arma_process = ArmaProcess(ar, ma)
    return arma_process.generate_sample(nsample=n_points, scale=sigma)



# Symulacja ARMA
arma_path = simulate_arma_process(phi, theta_ma, sigma, arma_points)

# Symulacja procesu OU
t_ou, paths_ou = simulate_ou_process(theta, mu, sigma, x0, T, dt, n_paths)



# Wykres 2: Porównanie OU z modelem ARMA
plt.figure(figsize=(12, 6))
plt.plot(t_ou, paths_ou[0, :], label='OU (jedna trajektoria)', alpha=0.7)
plt.plot(np.linspace(0, T, arma_points), arma_path, label='ARMA(1,1)', alpha=0.7)
plt.axhline(mu, color='red', linestyle='--', label='mean')
plt.title("Porównanie procesu Ornsteina-Uhlenbecka i ARMA(1,1)")
plt.xlabel("Czas")
plt.ylabel("Wartość")
plt.legend()
plt.grid(True)
plt.show()


def simulate_arma_like_ou(theta, sigma, dt, mu, x0, n_points):
    """
    Symuluje proces AR(1) naśladujący proces Ornsteina-Uhlenbecka.
    """
    phi = 1 - theta * dt  # Parametr AR na podstawie theta i dt
    ar = np.r_[1, -phi]  # AR(1) współczynniki
    ma = np.r_[1]  # MA pominięty (tylko szum biały)
    arma_process = ArmaProcess(ar, ma)

    # Generowanie próbek i dodanie dryfu do średniej stacjonarnej mu
    samples = arma_process.generate_sample(nsample=n_points, scale=sigma * np.sqrt(dt))
    return mu + samples


# Parametry modelu
n_points = int(T / dt)
arma_like_ou = simulate_arma_like_ou(theta, sigma, dt, mu, x0, n_points)

# Dostosowanie t_ou do długości arma_like_ou
t_ou = t_ou[:n_points]

# Wizualizacja porównania
plt.figure(figsize=(12, 6))
plt.plot(t_ou, paths_ou[0, :n_points], label='OU (jedna trajektoria)', alpha=0.7)
plt.plot(t_ou, arma_like_ou, label='AR(1) zbliżony do OU', alpha=0.7)
plt.axhline(mu, color='red', linestyle='--', label='Średnia stacjonarna')
plt.title("Porównanie procesu Ornsteina-Uhlenbecka i AR(1) zbliżonego do OU")
plt.xlabel("Czas")
plt.ylabel("Wartość")
plt.legend()
plt.grid(True)
plt.show()

