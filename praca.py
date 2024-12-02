import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.integrate import solve_ivp

# Parametry procesu
theta = 0.2  # Szybkość powrotu
mu = 0.0     # Średnia stacjonarna
sigma = 0.3  # Intensywność fluktuacji
x0 = 1.0     # Wartość początkowa
T = 100.0     # Całkowity czas symulacji
dt = 0.0001   # Długość kroku czasowego
n_paths = 1  # Liczba trajektorii

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



# Parametry procesu
theta = 0.2  # Szybkość powrotu
mu = 0.0     # Średnia stacjonarna
sigma = 0.3  # Intensywność fluktuacji
x0 = 1.0     # Wartość początkowa
T = 100.0     # Całkowity czas symulacji
dt = 0.0001   # Długość kroku czasowego
n_paths = 1  # Liczba trajektorii

# Parametry ARMA
phi = 0.8    # Parametr AR
theta_ma = -0.5  # Parametr MA
arma_points = int(T / dt)  # Liczba punktów ARMA

# Symulacja procesu OU
t_ou, paths_ou = simulate_ou_process(theta, mu, sigma, x0, T, dt, n_paths)

# Symulacja ARMA
arma_path = simulate_arma_process(phi, theta_ma, sigma, arma_points)


# Wykres 1: Trajektorie procesu OU
plt.figure(figsize=(12, 6))
for i in range(n_paths):
    plt.plot(t_ou, paths_ou[i, :], label=f'Trajektoria {i+1}')
plt.axhline(mu, color='red', linestyle='--', label='Średnia stacjonarna')
plt.title("Trajektorie procesu Ornsteina-Uhlenbecka")
plt.xlabel("Czas")
plt.ylabel("X(t)")
plt.legend()
plt.grid(True)
plt.show()

# Wykres 2: Porównanie OU z modelem ARMA
plt.figure(figsize=(12, 6))
plt.plot(t_ou, paths_ou[0, :], label='OU (jedna trajektoria)', alpha=0.7)
plt.plot(np.linspace(0, T, arma_points), arma_path, label='ARMA(1,1)', alpha=0.7)
plt.axhline(mu, color='red', linestyle='--', label='Średnia stacjonarna')
plt.title("Porównanie procesu Ornsteina-Uhlenbecka i ARMA(1,1)")
plt.xlabel("Czas")
plt.ylabel("Wartość")
plt.legend()
plt.grid(True)
plt.show()


# czesc 2

def solve_stochastic_ou(theta, mu, sigma, x0, T, dt):
    """
    Rozwiązuje stochastyczne równanie różniczkowe dla procesu Ornsteina-Uhlenbecka metodą Euler-Maruyama.

    Parameters:
        theta (float): Szybkość powrotu do średniej.
        mu (float): Średnia stacjonarna.
        sigma (float): Intensywność fluktuacji.
        x0 (float): Wartość początkowa.
        T (float): Czas symulacji.
        dt (float): Długość kroku czasowego.

    Returns:
        t (numpy array): Tablica czasów.
        x (numpy array): Trajektoria rozwiązania.
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)
    x = np.zeros(n_steps + 1)
    x[0] = x0

    for i in range(n_steps):
        dw = np.sqrt(dt) * np.random.randn()  # Przyrost procesu Wienera
        x[i + 1] = x[i] + theta * (mu - x[i]) * dt + sigma * dw

    return t, x


# Parametry procesu
theta = 0.7  # Szybkość powrotu
mu = 0.0  # Średnia stacjonarna
sigma = 0.3  # Intensywność fluktuacji
x0 = 1.0  # Wartość początkowa
T = 10.0  # Całkowity czas symulacji
dt = 0.01  # Długość kroku czasowego

# Rozwiązanie stochastyczne SDE
t_sde, x_sde = solve_stochastic_ou(theta, mu, sigma, x0, T, dt)


# Rozwiązanie deterministyczne (do porównania)
def deterministic_ou(t, theta, mu, x0):
    return mu + (x0 - mu) * np.exp(-theta * t)


t_det = np.linspace(0, T, int(T / dt) + 1)
x_det = deterministic_ou(t_det, theta, mu, x0)

# Wizualizacja
plt.figure(figsize=(12, 6))
plt.plot(t_sde, x_sde, label="Rozwiązanie stochastyczne (SDE)", color="blue")
plt.plot(t_det, x_det, label="Rozwiązanie deterministyczne", color="red", linestyle="--")
plt.axhline(mu, color="green", linestyle=":", label="Średnia stacjonarna")
plt.title("Rozwiązanie stochastycznego równania różniczkowego")
plt.xlabel("Czas")
plt.ylabel("X(t)")
plt.legend()
plt.grid(True)
plt.show()
