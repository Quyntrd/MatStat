# Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.stats as st

# Параметры
a = 0  # математическое ожидание
sigma2 = 2  # дисперсия
sigma = np.sqrt(sigma2)  # стандартное отклонение
gamma = 0.91  # доверительная вероятность
n = 20  # объем выборки
M = 1750  # количество выборок
k = 140  # количество выборок для случайной величины W

# 1.1 Интервальная оценка для математического ожидания при известной дисперсии
X = np.random.normal(a, sigma, size=n)
t_gamma = st.norm.ppf(1/2 + gamma/2)
a_left_known = X.mean() - sigma * t_gamma / np.sqrt(n)
a_right_known = X.mean() + sigma * t_gamma / np.sqrt(n)
print("1.1. Доверительный интервал (известная дисперсия):", (a_left_known, a_right_known))

# 1.2 Интервальная оценка для математического ожидания при неизвестной дисперсии
sigma_estimate = np.std(X, ddof=1)
t_gamma_unknown = st.t.ppf(1/2 + gamma/2, df=n-1)
a_left_unknown = X.mean() - sigma_estimate * t_gamma_unknown / np.sqrt(n)
a_right_unknown = X.mean() + sigma_estimate * t_gamma_unknown / np.sqrt(n)
print("1.2. Доверительный интервал (неизвестная дисперсия):", (a_left_unknown, a_right_unknown))

# 1.3 Интервальная оценка для дисперсии
delta_0 = st.chi2.ppf(1/2 + gamma/2, df=n-1)
delta_1 = st.chi2.ppf(1/2 - gamma/2, df=n-1)
s2 = np.var(X, ddof=1)
sigma2_left = (n - 1) * s2 / delta_0
sigma2_right = (n - 1) * s2 / delta_1
print("1.3. Доверительный интервал для дисперсии:", (sigma2_left, sigma2_right))

# 2. Зависимость длины доверительного интервала от надежности
V = 500
gamma_values = np.linspace(0.7, 0.999, V)
interval_lengths_mx = []
interval_lengths_var = []

for g in gamma_values:
    t_gamma = st.norm.ppf(1/2 + g/2)
    interval_mx = 2 * sigma * t_gamma / np.sqrt(n)
    delta_0 = st.chi2.ppf(1/2 + g/2, df=n-1)
    delta_1 = st.chi2.ppf(1/2 - g/2, df=n-1)
    interval_var = (n - 1) * s2 * (1/delta_1 - 1/delta_0)
    interval_lengths_mx.append(interval_mx)
    interval_lengths_var.append(interval_var)

plt.figure(figsize=(10, 5))
plt.plot(gamma_values, interval_lengths_mx, label="Математическое ожидание")
plt.plot(gamma_values, interval_lengths_var, label="Дисперсия")
plt.xlabel("Надежность γ")
plt.ylabel("Длина интервала")
plt.legend()
plt.title("Длина доверительного интервала от надежности")
plt.show()

# 3. Зависимость длины интервала от объема выборки
n_values = range(5, 101)
lengths_mx = []
lengths_var = []

for n in n_values:
    t_gamma = st.norm.ppf(1/2 + gamma/2)
    interval_mx = 2 * sigma * t_gamma / np.sqrt(n)
    delta_0 = st.chi2.ppf(1/2 + gamma/2, df=n-1)
    delta_1 = st.chi2.ppf(1/2 - gamma/2, df=n-1)
    interval_var = (n - 1) * s2 * (1/delta_1 - 1/delta_0)
    lengths_mx.append(interval_mx)
    lengths_var.append(interval_var)

plt.figure(figsize=(10, 5))
plt.plot(n_values, lengths_mx, label="Математическое ожидание")
plt.plot(n_values, lengths_var, label="Дисперсия")
plt.xlabel("Объем выборки n")
plt.ylabel("Длина интервала")
plt.legend()
plt.title("Длина доверительного интервала от объема выборки")
plt.show()

# 4. Оценка γ* при неизвестной дисперсии
count_gamma = 0

for _ in range(M):
    X = np.random.normal(a, sigma, size=n)
    s = np.std(X, ddof=1)
    a_left = X.mean() - s * t_gamma_unknown / np.sqrt(n)
    a_right = X.mean() + s * t_gamma_unknown / np.sqrt(n)
    if a_left < a < a_right:
        count_gamma += 1

gamma_star = count_gamma / M
print("4. Фактическая γ*:", gamma_star)

# 5. Анализ случайной величины Z
Z_values = []
for _ in range(M):
    X = np.random.normal(a, sigma, size=n)
    s = np.std(X, ddof=1)
    Z = (X.mean() - a) / (s / np.sqrt(n))
    Z_values.append(Z)

# 5.1 Вычисление характеристик
Z_values = np.array(Z_values)
mean_Z = np.mean(Z_values)
var_Z = np.var(Z_values)
skew_Z = st.skew(Z_values)
kurt_Z = st.kurtosis(Z_values)
print(f"5.1. Характеристики Z: Среднее={mean_Z}, Дисперсия={var_Z}, Асимметрия={skew_Z}, Эксцесс={kurt_Z}")

# 5.2 Построение графиков
plt.hist(Z_values, bins='sturges', density=True, alpha=0.6, color='g', label="Гистограмма Z")
x = np.linspace(min(Z_values), max(Z_values), 1000)
plt.plot(x, st.t.pdf(x, df=n-1), 'r-', label="Теоретическая кривая t-распределения")
plt.legend()
plt.title("Гистограмма и теоретическая плотность Z")
plt.show()

sb.boxplot(Z_values)
plt.title("Ящичковая диаграмма для Z")
plt.show()
