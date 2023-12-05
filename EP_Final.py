# Exercicio Programa Calculo Numerico
# Felipe Tilkian de Carvalho 
# Numero USP : 9318696

# Importação de bibliotecas necessárias

import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

# Definição de funções 

# Função de RF45
def RF45(func, t_span, y0, h, epsilon):
    t0, tf = t_span
    y0 = np.atleast_1d(y0)
    y = y0.copy()
    t = t0
    t_values = [t0]
    y_values = [y0]

    while t < tf:
        if t + h > tf:
            h = tf - t

        k1 = func(t, y)
        k2 = func(t + 1/4 * h, y + 1/4 * k1 * h)
        k3 = func(t + 3/8 * h, y + (3/32 * k1 + 9/32 * k2) * h)
        k4 = func(t + 12/13 * h, y + (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3) * h)
        k5 = func(t + h, y + (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4) * h)
        k6 = func(t + 1/2 * h, y - (8/27 * k1 - 2 * k2 + 3544/2565 * k3 - 1859/4104 * k4 + 11/40 * k5) * h)

        y4 = y + (25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4 - 1/5 * k5) * h
        y5 = y + (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6) * h

        erro = np.linalg.norm(y5 - y4, ord=np.inf)

        if erro < epsilon:
            t += h
            y = y5
            t_values.append(t)
            y_values.append(y)
            h *= min(4, 0.84 * (epsilon / erro)**0.25)
        else:
            h *= 0.84 * (epsilon / erro)**0.25

    return np.array(t_values), np.array(y_values).T


def energia_cinetica(omega1, theta1, omega2, theta2):
    "Calcula a energia cinética do pêndulo duplo."
    T = omega1**2 + np.cos(theta1 - theta2) * omega1 * omega2 + (omega2**2)/2
    return T

def energia_potencial(theta1, theta2):
    "Calcula a energia potencial do pêndulo duplo."
    V = 2 * (1 - np.cos(theta1)) + (1 - np.cos(theta2))
    return V

def total_energy(y):
    "Calcula a energia total (cinética + potencial) do sistema para um dado momento."
    theta1, omega1, theta2, omega2 = y
    T = energia_cinetica(omega1, theta1, omega2, theta2)
    V = energia_potencial(theta1, theta2)
    return T + V

# Função do pendulo duplo

def pendulo_duplo(t, Y):
    theta1, omega1, theta2, omega2 = Y
    delta = theta1 - theta2
    
    f1 = (np.cos(delta) * np.sin(theta2) - np.sin(delta) * (np.cos(delta) * omega1**2 + omega2**2)) / (1 + np.sin(delta)**2)
    f2 = (2 * np.cos(delta) * np.sin(theta1) - np.sin(theta2) + np.sin(delta) * (2 * omega1**2 + np.cos(delta) * omega2**2)) / (1 + np.sin(delta)**2)
    
    dot_theta1 = omega1
    dot_omega1 = f1
    dot_theta2 = omega2
    dot_omega2 = f2
    
    return np.array([dot_theta1, dot_omega1, dot_theta2, dot_omega2])

# Converte graus para radianos

def graus_para_rad(degrees):
    return degrees * np.pi / 180


# Teste 1

def test1_func(t, x):
    return 1 + (x - t)**2

# Condições iniciais
x0_test1 = -18.95
t_span_test1 = (1.05, 3)
h_test1 = 0.01
epsilon_test1 = 1e-5

# Exemplo de uso para o Teste 1
t_values_test1, x_values_test1 = RF45(test1_func, t_span_test1, x0_test1, h_test1, epsilon_test1)

# Solução exata para o Teste 1
def solucao_teste1(t):
    return t + 1 / (1 - t)

exact_values_test1 = solucao_teste1(t_values_test1)

# Erro em relação à solução exata
error_test1 = np.abs(x_values_test1[0] - exact_values_test1)


# Plotagem dos resultados do Teste 1
plt.figure(figsize=(12, 5))

plt.plot(t_values_test1, x_values_test1[0], label='Aproximação RKF45')
plt.plot(t_values_test1, exact_values_test1, label='Solução Exata', linestyle='dashed')
plt.title('Teste 1 - Aproximação RKF45 vs Solução Exata')
plt.xlabel('Tempo (t)')
plt.ylabel('Solução x(t)')
plt.legend()
plt.show()

# Imprimir valores de h, solução e erro para cada passo
for i in range(len(t_values_test1)):
    current_solution = x_values_test1[0, i]  # Ajuste para indexação bidimensional
    current_error = error_test1[i]
    print(f"Passo {i + 1} - t: {t_values_test1[i]}, h: {h_test1}, x(t): {current_solution}, Erro: {current_error}")





# Teste 2
def test2_func(t, X):
    A = np.array([[-2, -1, -1, -2],
                  [1, -2, 2, -1],
                  [-1, -2, -2, -1],
                  [2, -1, 1, -2]])

    return A @ X

# Condições iniciais
X0_test2 = np.array([1, 1, 1, -1])
t_span_test2 = (0, 2)
h_test2 = 0.1
epsilon_test2 = 1e-5

# Exemplo de uso para o Teste 2
t_values_test2, X_values_test2 = RF45(test2_func, t_span_test2, X0_test2, h_test2, epsilon_test2)


# Solução exata para o Teste 2
def solucao_teste2(t):
    e_t = np.exp(-t)
    return np.array([e_t,
                     -t * e_t,
                     np.sin(t) + t * np.cos(3 * t),
                     np.cos(t) - 3 * t * np.sin(3 * t)])



# Solução exata para o Teste 2
exact_values_test2 = np.array([solucao_teste2(t) for t in t_values_test2]).T  # Transposta para alinhar com X_values_test2


# Erro em relação à solução exata
error_test2 = np.max(np.abs(X_values_test2 - exact_values_test2), axis=0)  # axis=0 para calcular o máximo ao longo das colunas

# Plotagem dos resultados do Teste 2
plt.figure(figsize=(12, 5))

labels_test2 = [r'$X_1$', r'$X_2$', r'$X_3$', r'$X_4$']
for i in range(X_values_test2.shape[0]):
    plt.plot(t_values_test2, X_values_test2[i], label=labels_test2[i])
    plt.title('Teste 2 - Aproximação RKF45 vs Solução Exata')
    plt.xlabel('Tempo (t)')
    plt.ylabel('Solução X(t)')

plt.legend()
plt.show()

# Imprimir valores de h, solução e erro para cada passo
for i in range(len(t_values_test2)):
    print(f"Passo {i + 1} - t: {t_values_test2[i]}, h: {h_test2}, X(t): {X_values_test2[:, i]}, Erro: {error_test2[i]}")

# Teste 3
def test3_func(t, X):
    m = len(X)
    A = np.zeros((m, m))

    for i in range(m):
        A[i, i] = -2
        if i > 0:
            A[i, i - 1] = 1
            A[i - 1, i] = 1

    return A @ X

# Condições iniciais
m_test3 = 7
y_values_test3 = np.linspace(1, m_test3, m_test3) / (m_test3 + 1)
X0_test3 = np.sin(np.pi * y_values_test3) + np.sin(m_test3 * np.pi * y_values_test3)
t_span_test3 = (0, 2)
h_test3 = 0.1
epsilon_test3 = 1e-5

# Exemplo de uso para o Teste 3
t_values_test3, X_values_test3 = RF45(test3_func, t_span_test3, X0_test3, h_test3, epsilon_test3)


# Solução exata para o Teste 3
def solucao_teste3(t):
    lambda_1 = 2 * (1 - np.cos(np.pi / (m_test3 + 1)))
    lambda_2 = 2 * (1 - np.cos(m_test3 * np.pi / (m_test3 + 1)))
    e_lambda1t = np.exp(-lambda_1 * t)
    e_lambda2t = np.exp(-lambda_2 * t)
    return np.array([e_lambda1t * np.sin(np.pi * y) + e_lambda2t * np.sin(m_test3 * np.pi * y) for y in y_values_test3])

# Solução exata para o Teste 3
exact_values_test3 = np.array([solucao_teste3(t) for t in t_values_test3]).T  # Transposta para alinhar com X_values_test3

# Erro em relação à solução exata
error_test3 = np.max(np.abs(X_values_test3 - exact_values_test3), axis=0)  # axis=0 para calcular o máximo ao longo das colunas

# Plotagem dos resultados do Teste 3
plt.figure(figsize=(12, 5))

for i in range(X_values_test3.shape[0]):
    plt.plot(t_values_test3, X_values_test3[i], label=f'$X_{i + 1}$')
    plt.title('Teste 3 - Aproximação RKF45 vs Solução Exata')
    plt.xlabel('Tempo (t)')
    plt.ylabel('Solução X(t)')

plt.legend()
plt.show()

# Imprimir valores de h, solução e erro para cada passo
for i in range(len(t_values_test3)):
    print(f"Passo {i + 1} - t: {t_values_test3[i]}, h: {h_test3}, X(t): {X_values_test3[:, i]}, Erro: {error_test3[i]}")
    
    
# Agora que os testes foram realizados podemos ir para o pendulo duplo
    
# Valores iniciais para o caso a)
theta1_0_a = graus_para_rad(20)
theta2_0_a = graus_para_rad(0)
omega1_0 = omega2_0 = 0

# Condições iniciais
y0_a = [theta1_0_a, omega1_0, theta2_0_a, omega2_0]

# Parâmetros da simulação
t_span = (0, 120)
h = 0.1  # Passo inicial - pode ser ajustado pelo método
epsilon = 1e-8  # Tolerância

# Executa a simulação para o caso a)
t_values_a, y_values_a = RF45(pendulo_duplo, t_span, y0_a, h, epsilon)

# Calcula a energia para cada passo de tempo
energies_a = np.array([total_energy(state) for state in y_values_a.T])

# Plotagem da energia total e dos ângulos para o caso a
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(t_values_a, energies_a, label='Energia Total')
plt.xlabel('Tempo (t)')
plt.ylabel('Energia (E)')
plt.title('Energia Total do Pêndulo Duplo ao Longo do Tempo - Caso a')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_values_a, y_values_a[0], label='Theta 1')
plt.plot(t_values_a, y_values_a[2], label='Theta 2')
plt.xlabel('Tempo (t)')
plt.ylabel('Ângulo (radianos)')
plt.title('Ângulos do Pêndulo Duplo ao Longo do Tempo - Caso a')
plt.legend()

plt.tight_layout()
plt.show()


# Condições iniciais para o caso b
theta1_0_b = graus_para_rad(20)
theta2_0_b = graus_para_rad(20)
y0_b = [theta1_0_b, omega1_0, theta2_0_b, omega2_0]

# Condições iniciais para o caso c
theta1_0_c = graus_para_rad(130)
theta2_0_c = graus_para_rad(130)
y0_c = [theta1_0_c, omega1_0, theta2_0_c, omega2_0]

# Simulação para o caso b
t_values_b, y_values_b = RF45(pendulo_duplo, t_span, y0_b, h, epsilon)

# Simulação para o caso c
t_values_c, y_values_c = RF45(pendulo_duplo, t_span, y0_c, h, epsilon)

# Energia para o caso b
energies_b = np.array([total_energy(state) for state in y_values_b.T])

# Energia para o caso c
energies_c = np.array([total_energy(state) for state in y_values_c.T])

# Plotagem da energia total e dos ângulos para o caso b
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(t_values_b, energies_b, label='Energia Total')
plt.xlabel('Tempo (t)')
plt.ylabel('Energia (E)')
plt.title('Energia Total do Pêndulo Duplo ao Longo do Tempo - Caso b')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_values_b, y_values_b[0], label='Theta 1')
plt.plot(t_values_b, y_values_b[2], label='Theta 2')
plt.xlabel('Tempo (t)')
plt.ylabel('Ângulo (radianos)')
plt.title('Ângulos do Pêndulo Duplo ao Longo do Tempo - Caso b')
plt.legend()

plt.tight_layout()
plt.show()

# Plotagem da energia total e dos ângulos para o caso c
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(t_values_c, energies_c, label='Energia Total')
plt.xlabel('Tempo (t)')
plt.ylabel('Energia (E)')
plt.title('Energia Total do Pêndulo Duplo ao Longo do Tempo - Caso c')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_values_c, y_values_c[0], label='Theta 1')
plt.plot(t_values_c, y_values_c[2], label='Theta 2')
plt.xlabel('Tempo (t)')
plt.ylabel('Ângulo (radianos)')
plt.title('Ângulos do Pêndulo Duplo ao Longo do Tempo - Caso c')
plt.legend()

plt.tight_layout()
plt.show()
