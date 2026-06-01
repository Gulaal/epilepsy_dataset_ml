import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

x = np.linspace(-4, 4, 100)
y_relu = relu(x)
y_gelu = gelu(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y_relu, label='ReLU', linewidth=2)
plt.plot(x, y_gelu, label='GELU (approximated)', linewidth=2)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12)
plt.title('Функции активации ReLU и GELU', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.xlim(-4, 4)
plt.ylim(-1, 4.2)
plt.savefig('relu_gelu.pdf', bbox_inches='tight')
plt.savefig('relu_gelu.png', dpi=300)
plt.show()