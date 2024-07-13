from optimizador_joshua_8 import (
    objetive_function, restriction_functions, univariablefunction, 
    cauchy, fletcher_reeves, hooke_jeeves, neldermead, newton, random_walk,
    biseccion, fibonacci, goldensearch, interval, newton_raphson, secante
)

# Prueba de uso de las funciones y clases importadas
import numpy as np
from optimizador_joshua_8.metodos_multivariables.cauchy import cauchy
from optimizador_joshua_8.metodos_multivariables.newtonmethod import newton
from optimizador_joshua_8.metodos_multivariables.fletcherreeves import fletcher_reeves
from optimizador_joshua_8.metodos_multivariables.hookejeeves import hooke_jeeves
from optimizador_joshua_8.funciones.objetivo import objetive_function
from optimizador_joshua_8.metodos_multivariables.neldermead import neldermead

# Definir el nombre de la función objetivo y el espacio de búsqueda
func_name = "himmelblau"
espaciobusqueda = np.array([[0, 0], [1, 1]])

# Crear una instancia de la clase objetive_function
obj_func = objetive_function(func_name, espaciobusqueda)
print("Límite inferior:", obj_func.get_limiteinf())

# Obtener la función objetivo y probarla
f = obj_func.get_function()
print("Valor de la función en [3, 2]:", f([3, 2]))

# Definir parámetros comunes para los optimizadores
epsilon = 0.01
x_inicial = 1
x_limite = 10
iteraciones = 100
inicial = [1, 1]
inicial_n = [2, 3]

# Crear y probar instancias de los optimizadores
c = cauchy(inicial, obj_func, epsilon=epsilon, epsilon2=epsilon)
print("Resultado del método Cauchy:", c.optimize())

n = newton(inicial_n, obj_func, epsilon=epsilon, epsilon2=epsilon)
print("Resultado del método Newton:", n.optimize())

g = fletcher_reeves(inicial, obj_func, epsilon=epsilon, epsilon2=epsilon, epsilon3=epsilon)
print("Resultado del método Fletcher-Reeves:", g.optimize())

#Hookejeeves 
func_name2 = "booth"
espaciobusqueda = np.array([[-5, -2.5], [5, 2.5]])
obj_func2 = objetive_function(func_name2, espaciobusqueda)

# Definir parámetros para Hooke-Jeeves
x_inicial = [-5, -2.5]
delta = [0.5, 0.25]
alpha = 2
e = 0.01

# Crear y probar la instancia de Hooke-Jeeves
hj = hooke_jeeves(x_inicial, delta,obj_func2,e)
print(hj.funcion)
print("Resultado del método Hooke-Jeeves:", hj.optimize())
#nelder mead 
func_name3 = "rosenbrock"
espaciobusqueda = np.array([[-5, -2.5], [5, 2.5]])  # Puedes ajustar el rango según sea necesario
obj_func3 = objetive_function(func_name3)

# Definir parámetros para Nelder-Mead
initialpoint = [2, 1.5, 3, -1.5, -2]
gamma = 1.1
b = 0.1
e = 0.5

# Crear y probar la instancia de Nelder-Mead
nm = neldermead(initialpoint, gamma=gamma, beta=b, f=obj_func3,epsilon=e)
best = nm.optimize()
print("Mejor punto encontrado:", best)
