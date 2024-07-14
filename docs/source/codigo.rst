.. _codigo:

Clase funciones
======================================
Esta es la clase principal de las fuciones de prueba para este paquete

Clase funcion
----------------------------------
.. code-block:: python

    class funcion:
    """
    Clase base para definir una función para ser utilizada eb los métodos de optimización. Es la clase padre de las funciones

    Attributes:
        name (str): Nombre de la función.
        limiteinf (np.array): Límite inferior del espacio de búsqueda.
        limitesup (np.array): Límite superior del espacio de búsqueda.
        espacio (np.array): Espacio de búsqueda definido por los límites.

    Methods:
        validate_search_space(): Valida que el espacio de búsqueda esté dentro de los límites predefinidos para la función.
        get_function(): Método que debe ser implementado por las subclases para devolver la función de optimización.
        get_limitesup(): Método que debe ser implementado por las subclases para devolver el límite superior.
        get_limiteinf(): Método que debe ser implementado por las subclases para devolver el límite inferior.
    """

    def __init__(self, name, espaciobussqueda: np.array):
        """
        Inicializa la clase con el nombre de la función y el espacio de búsqueda.

        Args:
            name (str): Nombre de la función.
            espaciobussqueda (np.array): Espacio de búsqueda definido por los límites inferior y superior.

        Raises:
            ValueError: Si el espacio de búsqueda no es un array de dos elementos.
        """
        self.name = name
        self.limiteinf = espaciobussqueda[0]
        self.limitesup = espaciobussqueda[1]
        self.espacio = espaciobussqueda
        self.validate_search_space()

    def validate_search_space(self):
        """
        Valida que el espacio de búsqueda esté dentro de los límites predefinidos para la función.

        Raises:
            ValueError: Si el espacio de búsqueda no es un array de dos elementos.

        Warnings:
            warnings.warn: Si el espacio de búsqueda está fuera del rango predefinido para la función.
        """
        if len(self.espacio) != 2:
            raise ValueError("Search space must be an array of two elements")
        
        func_name = self.name.lower()
        search_spaces = {
            'himmelblau': [-5, 5, -5, 5],
            'rastrigin': [-5.12, 5.12, -5.12, 5.12],
            'beale': [-4.5, 4.5, -4.5, 4.5],
            'goldstein': [-2, 2, -2, 2],
            'boothfunction': [-10, 10, -10, 10],
            'bukin_n6': [-15, -5, -3, 3],
            'schaffer_n2': [-100, 100, -100, 100],
            'schaffer_n4': [-100, 100, -100, 100],
            'styblinski_tang': [-5, 5, -5, 5],
            'rosenbrock_constrained_cubic_line': [-1.5, 1.5, -0.5, 2.5],
            'rosenbrock_constrained_disk': [-1.5, 1.5, -1.5, 1.5],
            'mishras_bird_constrained': [-10, 0, -6.5, 0],
            'townsend_modified': [-2.25, 2.25, -2.25, 1.75],
            'gomez_levy_modified': [-1, 0.75, -1, 1],
            'simionescu_function': [-1.25, 1.25, -1.25, 1.25]
        }

        if func_name in search_spaces:
            x_min, x_max, y_min, y_max = search_spaces[func_name]
            if not (x_min <= self.limiteinf[0] <= x_max and y_min <= self.limitesup[0] <= y_max):
                warning_msg = f"Warning: Search space is outside the predefined range for function {func_name}."
                warnings.warn(warning_msg)

    def get_function(self):
        """
        Método que debe ser implementado por las subclases para devolver la función de optimización.

        Raises:
            NotImplementedError: Si la subclase no implementa este método.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_limitesup(self):
        """
        Método que debe ser implementado por las subclases para devolver el límite superior.

        Raises:
            NotImplementedError: Si la subclase no implementa este método.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_limiteinf(self):
        """
        Método que debe ser implementado por las subclases para devolver el límite inferior.

        Raises:
            NotImplementedError: Si la subclase no implementa este método.
        """
        raise NotImplementedError("Subclasses should implement this method.")



Clase objetive_funcion 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Esta es la clase heredada que se enfoca en funciones objetivo de prueba para el paquete
.. code-block:: python 
    import numpy as np
    class objetive_function(funcion):
    """
    Clase para definir varias funciones objetivo de optimización.

    Methods:
        himmelblau: Función de Himmelblau. Este es nombre que se debe de poner para acceder a la funcion
        sphere: Función de Sphere.Este es nombre que se debe de poner para acceder a la funcion
        rastrigin: Función de Rastrigin.Este es nombre que se debe de poner para acceder a la funcion
        rosenbrock: Función de Rosenbrock.Este es nombre que se debe de poner para acceder a la funcion
        beale: Función de Beale.Este es nombre que se debe de poner para acceder a la funcion
        goldstein: Función de Goldstein-Price.Este es nombre que se debe de poner para acceder a la funcion
        booth: Función de Booth.Este es nombre que se debe de poner para acceder a la funcion
        bunkinn6: Función de Bukin N.6.Este es nombre que se debe de poner para acceder a la funcion
        matyas: Función de Matyas.Este es nombre que se debe de poner para acceder a la funcion
        levi: Función de Levi.Este es nombre que se debe de poner para acceder a la funcion
        threehumpcamel: Función de Three-Hump Camel.Este es nombre que se debe de poner para acceder a la funcion
        easom: Función de Easom.Este es nombre que se debe de poner para acceder a la funcion
        crossintray: Función de Cross-in-Tray.Este es nombre que se debe de poner para acceder a la funcion
        eggholder: Función de Eggholder.Este es nombre que se debe de poner para acceder a la funcion
        holdertable: Función de Holder Table.Este es nombre que se debe de poner para acceder a la funcion
        mccormick: Función de McCormick.Este es nombre que se debe de poner para acceder a la funcion
        schaffern2: Función de Schaffer N.2.Este es nombre que se debe de poner para acceder a la funcion
        schaffern4: Función de Schaffer N.4.Este es nombre que se debe de poner para acceder a la funcion
        styblinskitan: Función de Styblinski-Tang.Este es nombre que se debe de poner para acceder a la funcion
        shekel: Función de Shekel.Este es nombre que se debe de poner para acceder a la funcion
    """

    def __init__(self, name, espaciobussqueda: np.array=[[0,0],[1,1]]):
        """
        Inicializa la clase con el nombre de la función y el espacio de búsqueda.

        Args:
            name (str): Nombre de la función.
            espaciobussqueda (np.array): Espacio de búsqueda definido por los límites inferior y superior.
        """
        super().__init__(name, espaciobussqueda)

    def himmelblau(self, p):
        """Función de Himmelblau."""
        return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2
    
    def sphere(self, x):
        """Función de Sphere."""
        return np.sum(np.square(x))

    def rastrigin(self, x, A=10):
        """Función de Rastrigin."""
        self.limite = float(5.12)
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    def rosenbrock(self, x):
        """Función de Rosenbrock."""
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def beale(self, x):
        """Función de Beale."""
        self.limite = 4.5
        return ((1.5 - x[0] + x[0] * x[1])**2 +
                (2.25 - x[0] + x[0] * x[1]**2)**2 +
                (2.625 - x[0] + x[0] * x[1]**3)**2)
    
    def goldstein(self, x):
        """Función de Goldstein-Price."""
        self.limite = 2
        part1 = (1 + (x[0] + x[1] + 1)**2 * 
                 (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2))
        part2 = (30 + (2 * x[0] - 3 * x[1])**2 * 
                 (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2))
        return part1 * part2

    def booth(self, x):
        """Función de Booth."""
        self.limite = 10
        return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

    def bunkinn6(self, x):
        """Función de Bukin N.6."""
        return 100 * np.sqrt(np.abs(x[1] - 0.001 * x[0]**2)) + 0.01 * np.abs(x[0] + 10)

    def matyas(self, x):
        """Función de Matyas."""
        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

    def levi(self, x):
        """Función de Levi."""
        part1 = np.sin(3 * np.pi * x[0])**2
        part2 = (x[0] - 1)**2 * (1 + np.sin(3 * np.pi * x[1])**2)
        part3 = (x[1] - 1)**2 * (1 + np.sin(2 * np.pi * x[1])**2)
        return part1 + part2 + part3
    
    def threehumpcamel(self, x):
        """Función de Three-Hump Camel."""
        return 2 * x[0]**2 - 1.05 * x[0]**4 + (x[0]**6) / 6 + x[0] * x[1] + x[1]**2

    def easom(self, x):
        """Función de Easom."""
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)

    def crossintray(self, x):
        """Función de Cross-in-Tray."""
        op = np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))
        return -0.0001 * (op + 1)**0.1

    def eggholder(self, x):
        """Función de Eggholder."""
        op1 = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0] / 2 + (x[1] + 47))))
        op2 = -x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))
        return op1 + op2

    def holdertable(self, x):
        """Función de Holder Table."""
        op = np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))
        return -op

    def mccormick(self, x):
        """Función de McCormick."""
        return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1

    def schaffern2(self, x):
        """Función de Schaffer N.2."""
        self.limite = 100
        numerator = np.sin(x[0]**2 - x[1]**2)**2 - 0.5
        denominator = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
        return 0.5 + numerator / denominator

    def schaffern4(self, x):
        """Función de Schaffer N.4."""
        self.limite = 100
        num = np.cos(np.sin(np.abs(x[0]**2 - x[1]**2))) - 0.5
        den = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
        return 0.5 + num / den

    def styblinskitang(self, x):
        """Función de Styblinski-Tang."""
        self.limite = 5
        return np.sum(x**4 - 16 * x**2 + 5 * x) / 2
    
    def shekel(self, x, a=None, c=None):
        """Función de Shekel."""
        if a is None:
            a = np.array([
                [4.0, 4.0, 4.0, 4.0],
                [1.0, 1.0, 1.0, 1.0],
                [8.0, 8.0, 8.0, 8.0],
                [6.0, 6.0, 6.0, 6.0],
                [3.0, 7.0, 3.0, 7.0],
                [2.0, 9.0, 2.0, 9.0],
                [5.0, 5.0, 3.0, 3.0],
                [8.0, 1.0, 8.0, 1.0],
                [6.0, 2.0, 6.0, 2.0],
                [7.0, 3.6, 7.0, 3.6]
            ])
        if c is None:
            c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        
        m = len(c)
        s = 0
        for i in range(m):
            s -= 1 / (np.dot(x - a[i, :2], x - a[i, :2]) + c[i])
        return s

    def get_function(self):
        """
        Devuelve la función objetivo según el nombre especificado.

        Returns:
            function: Función objetivo correspondiente al nombre.
        
        Raises:
            ValueError: Si la función especificada no está definida en la clase.
        """
        func = getattr(self, self.name.lower(), None)
        if func is None:
            raise ValueError(f"Function '{self.name}' is not defined in the class.")
        return func
    
    def get_limitesup(self):
        """
        Devuelve el límite superior del espacio de búsqueda.

        Returns:
            float: Límite superior del espacio de búsqueda.
        """
        return self.limitesup[1]
    
    def get_limiteinf(self):
        """
        Devuelve el límite inferior del espacio de búsqueda.

        Returns:
            float: Límite inferior del espacio de búsqueda.
        """
        return self.limiteinf[0]



.. code-block:: python
    import numpy as np
    from optimizador_joshua_8.funciones.objetivo import objetive_function

    # Definir el nombre de la función objetivo y el espacio de búsqueda
    func_name = "himmelblau"
    espaciobusqueda = np.array([[0, 0], [1, 1]])

    # Crear una instancia de la clase objetive_function
    obj_func = objetive_function(func_name, espaciobusqueda)
    print("Límite inferior del espacio de búsqueda:", obj_func.get_limiteinf())
    print("Límite superior del espacio de búsqueda:", obj_func.get_limitesup())

    # Obtener la función objetivo y calcular su valor en un punto específico
    f = obj_func.get_function()
    resultado = f([3, 2])
    print(f"Valor de la función {func_name} en [3, 2]:", resultado)
    
Clase restriction_functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    class restriction_functions(funcion):
    def __init__(self, name, espaciobusqueda: np.array):
        super().__init__(name, espaciobusqueda)
        """
        Estos son los nombres de las funciones que se debe poner en name para llamarlas
        rosenbrock_constrained_cubic_line
        rosenbrock_constrained_cubic_line_restriction
        rosenbrock_constrained_disk
        rosenbrock_constrained_disk_restriction
        mishras_bird_constrained
        mishras_bird_constrained_restriction
        townsend_function_modified
        townsend_function_modified_restriction
        gomez_levy_function_modified
        gomez_levy_function_modified_restriction
        simionescu_function
        simionescu_function_restriction
        """
    def rosenbrock_constrained_cubic_line(self, x):
        """
        Calcula la función Rosenbrock con restricción de línea cúbica.

        Parámetros:
        - x (np.array): Vector de entrada de variables.

        Retorna:
        - np.array: Valor de la función evaluada en x.
        """
        return np.array([(1 - x[0])**2 + 100 * (x[1] - (x[0]**2))**2])

    def rosenbrock_constrained_cubic_line_restriction(self, x):
        """
        Verifica la restricción para la función Rosenbrock con restricción de línea cúbica.

        Parámetros:
        - x (np.array): Vector de entrada de variables.

        Retorna:
        - bool: True si la restricción se cumple, False en caso contrario.
        """
        return (((x[0] - 1)**3 - x[1] + 1)) >= 0 and (x[0] + x[1] - 2) <= 0
        
    def rosenbrock_constrained_disk(self, x):
        """
        Calcula la función Rosenbrock con restricción de disco.

        Parámetros:
        - x (np.array): Vector de entrada de variables.

        Retorna:
        - np.array: Valor de la función evaluada en x.
        """
        return np.array([(1 - x[0])**2 + 100 * (x[1] - (x[0]**2))**2])

    def rosenbrock_constrained_disk_restriction(self, x):
        """
        Verifica la restricción para la función Rosenbrock con restricción de disco.

        Parámetros:
        - x (np.array): Vector de entrada de variables.

        Retorna:
        - bool: True si la restricción se cumple, False en caso contrario.
        """
        return (x[0]**2 + x[1]**2)

    def mishras_bird_constrained(self, x):
        """
        Calcula la función de Mishra's Bird con restricción.

        Parámetros:
        - x (np.array): Vector de entrada de variables.

        Retorna:
        - float: Valor de la función evaluada en x.
        """
        return np.sin(x[1]) * np.exp((1 - np.cos(x[0]))**2) + np.cos(x[0]) * np.exp((1 - np.sin(x[1]))**2) + (x[0] - x[1])**2

    def mishras_bird_constrained_restriction(self, x):
        """
        Verifica la restricción para la función de Mishra's Bird con restricción.

        Parámetros:
        - x (np.array): Vector de entrada de variables.

        Retorna:
        - bool: True si la restricción se cumple, False en caso contrario.
        """
        return (x[0] + 5)**2 + (x[1] + 5)**2 < 25

    def townsend_function_modified(self, x):
        """
        Calcula la función Townsend modificada.

        Parámetros:
        - x (np.array): Vector de entrada de variables.

        Retorna:
        - float: Valor de la función evaluada en x.
        """
        return -(np.cos((x[0] - 0.1) * x[1]))**2 - x[0] * np.sin(3 * x[0] + x[1])

    def townsend_function_modified_restriction(self, x):
        """
        Verifica la restricción para la función Townsend modificada.

        Parámetros:
        - x (np.array): Vector de entrada de variables.

        Retorna:
        - bool: True si la restricción se cumple, False en caso contrario.
        """
        t = np.arctan2(x[1], x[0])
        op1 = x[0]**2 + x[1]**2
        op2 = (2 * np.cos(t) - 0.5 * np.cos(2 * t) - 0.25 * np.cos(3 * t) - 0.125 * np.cos(4 * t))**2 + (2 * np.sin(t))**2
        return op1 < op2

    def gomez_levy_function_modified(self, x):
        """
        Calcula la función Gomez-Levy modificada.

        Parámetros:
        - x (np.array): Vector de entrada de variables.

        Retorna:
        - float: Valor de la función evaluada en x.
        """
        return 4 * x[0]**2 - 2.1 * x[0]**4 + (1 / 3) * x[0]**6 + x[0] * x[1] - 4 * x[1]**2 + 4 * x[1]**4

    def gomez_levy_function_modified_restriction(self, x):
        """
        Verifica la restricción para la función Gomez-Levy modificada.

        Parámetros:
        - x (np.array): Vector de entrada de variables.

        Retorna:
        - bool: True si la restricción se cumple, False en caso contrario.
        """
        return -np.sin(4 * np.pi * x[0]) + 2 * np.sin(2 * np.pi * x[1])**2 <= 1.5

    def simionescu_function(self, x):
        """
        Calcula la función Simionescu.

        Parámetros:
        - x (np.array): Vector de entrada de variables.

        Retorna:
        - float: Valor de la función evaluada en x.
        """
        return 0.1 * (x[0] * x[1])

    def simionescu_function_restriction(self, x):
        """
        Verifica la restricción para la función Simionescu.

        Parámetros:
        - x (np.array): Vector de entrada de variables.

        Retorna:
        - bool: True si la restricción se cumple, False en caso contrario.
        """
        r_T = 1
        r_S = 0.2
        n = 8
        angulo = np.arctan2(x[1], x[0]) 
        cosine_term = np.cos(n * angulo)
        op = (r_T + r_S * cosine_term) ** 2
        return x[0]**2 + x[1]**2 - op

    def get_function(self):
        """
        Obtiene la función especificada por 'name'.

        Retorna:
        - function: El objeto de la función.
        
        Lanza:
        - ValueError: Si la función 'name' no está definida en la clase.
        """
        func = getattr(self, self.name.lower(), None)
        if func is None:
            raise ValueError(f"La función '{self.name}' no está definida en la clase.")
        return func

    def get_limitesup(self):
        """
        Obtiene el límite superior del espacio de búsqueda.

        Retorna:
        - float: Límite superior del espacio de búsqueda.
        """
        return self.limiteinf[0]
    
    def get_limiteinf(self):
        """
        Obtiene el límite inferior del espacio de búsqueda.

        Retorna:
        - float: Límite inferior del espacio de búsqueda.
        """
        return self.limitesup[1]
O
.. code-block:: python
    import numpy as np
    from .restriction_functions import restriction_functions

    # Crear una instancia de la clase restriction_functions para acceder a las funciones restringidas
    restriccion = restriction_functions("rosenbrock_constrained_cubic_line", np.array([-5, 5]))

    # Punto de prueba
    x = np.array([0.5, 0.5])

    # Calcular el valor de la función
    valor_funcion = restriccion.rosenbrock_constrained_cubic_line(x)
    cumple_restriccion = restriccion.rosenbrock_constrained_cubic_line_restriction(x)

    print(f"Punto de prueba: {x}")
    print(f"Valor de la función: {valor_funcion}")
    print(f"Cumple con la restricción: {cumple_restriccion}")

Clase univariablefunction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    class univariablefunction(funcion):
    def __init__(self, name, espaciobussqueda: np.array=[0,0]):
        super().__init__(name, espaciobussqueda)
    
    def funcion1(self, x):
        """
        Calcula la función 1.

        Parámetros:
        - x (float): Valor de entrada para la función.

        Retorna:
        - float: Valor de la función evaluada en x.
        """
        self._validate_input(x)
        return (x**2) + (54/x)

    def funcion2(self, x):
        """
        Calcula la función 2.

        Parámetros:
        - x (float): Valor de entrada para la función.

        Retorna:
        - float: Valor de la función evaluada en x.
        """
        self._validate_input(x)
        return (x**3) + (2*x) - 3

    def funcion3(self, x):
        """
        Calcula la función 3.

        Parámetros:
        - x (float): Valor de entrada para la función.

        Retorna:
        - float: Valor de la función evaluada en x.
        """
        self._validate_input(x)
        return (x**4) + (x**2) - 33

    def funcion4(self, x):
        """
        Calcula la función 4.

        Parámetros:
        - x (float): Valor de entrada para la función.

        Retorna:
        - float: Valor de la función evaluada en x.
        """
        self._validate_input(x)
        return (3 * (x**4)) - (8 * (x**3)) - (6 * (x**2)) + 12 * x

    def _validate_input(self, x):
        """
        Valida que x sea un número escalar.

        Lanza:
        - ValueError: Si x es una lista, tupla, conjunto o arreglo.
        """
        if isinstance(x, (list, tuple, set, np.ndarray)):
            raise ValueError("x no debe ser una lista, tupla o arreglo. Debe ser un número escalar.")

    def get_function(self):
        """
        Obtiene la función especificada por 'name'.

        Retorna:
        - function: El objeto de la función.
        
        Lanza:
        - ValueError: Si la función 'name' no está definida en la clase.
        """
        func = getattr(self, self.name.lower(), None)
        if func is None:
            raise ValueError(f"La función '{self.name}' no está definida en la clase.")
        return func


q

.. code-block::python
    import numpy as np
    from .univariablefunction import univariablefunction
    funciones = univariablefunction("funcion1")
    x1 = 2.0
    x2 = -1.5
    x3 = 3.0
    resultado1 = funciones.funcion1(x1)
    print(f"Resultado de la función 1 en x={x1}: {resultado1}")

Métodos de optimizacon
======================================
Clase optimizador
----------------------------------

.. code-block:: python
    class optimizador:
    """
    Esta clase es la clase padre de los metodos 
    """
    def __init__(self, f, epsilon, iter=100):
        """
        Inicializa un optimizador con la función objetivo, la tolerancia y el número máximo de iteraciones.

        Parámetros:
        - f (function or funcion): Función a optimizar. Puede ser una función ordinaria o un objeto de la clase 'funcion'.
        - epsilon (float): Tolerancia para la convergencia del método.
        - iter (int, opcional): Número máximo de iteraciones (predeterminado: 100).
        """
        self.funcion = self.validate_function(f)
        self.epsilon = epsilon
        self.iteraciones = iter

    def validate_function(self, f):
        """
        Valida y obtiene la función a optimizar.

        Parámetros:
        - f (function or funcion): Función a validar.

        Retorna:
        - function: si la funcion es de la clase funcion pertenenciente a este paquete, manda a llamar el metodo get funcion para obternela. 
        Sino la funcion del optimizador sera la indicada.Seguidamente en se esta trabajando para que los métodos de gradiente puenda recibir cualquier funcion por lo que
        no se recomienda utlizar una funcion que no sean las mostradas en el paquete para los métodos de cauchy, newton y fletcher reeves.
        """
        if isinstance(f, funcion):
            return f.get_function()
        else:
            return f


    class optimizador_univariable(optimizador):
        def __init__(self, x_inicial, xlimite, f, epsilon, iter=100):
            """
            Inicializa un optimizador univariable con el punto inicial, el límite de búsqueda, la función objetivo, la tolerancia y el número máximo de iteraciones.

            Parámetros:
            - x_inicial (float): Punto inicial para la búsqueda.
            - xlimite (float): Límite superior del intervalo de búsqueda.
            - f (function or funcion): Función a optimizar. Puede ser una función ordinaria o un objeto de la clase 'funcion'.
            - epsilon (float): Tolerancia para la convergencia del método.
            - iter (int, opcional): Número máximo de iteraciones (predeterminado: 100).
            """
            super().__init__(f, epsilon, iter)
            self.valor_inicial = x_inicial
            self.limite = xlimite
        
        def optimize(self):
            """
            Método abstracto para optimizar. Debe ser implementado en las subclases.

            Lanza:
            - NotImplementedError: Si no se implementa en la subclase.
            """
            raise NotImplementedError("Subclasses should implement this method.")


    class optimizador_multivariable(optimizador):
        def __init__(self, variables, f, epsilon, iter=100):
            """
            Inicializa un optimizador multivariable con las variables de entrada, la función objetivo, la tolerancia y el número máximo de iteraciones.

            Parámetros:
            - variables (list or np.array): Variables de entrada para la optimización.
            - f (function or funcion): Función a optimizar. Puede ser una función ordinaria o un objeto de la clase 'funcion'.
            - epsilon (float): Tolerancia para la convergencia del método.
            - iter (int, opcional): Número máximo de iteraciones (predeterminado: 100).
            """
            super().__init__(f, epsilon, iter)
            self.variables = variables


    class by_regions_elimination(optimizador_univariable):
        def __init__(self, x_inicial, xlimite, f, epsilon, iter=100):
            """
            Inicializa el método de eliminación por regiones para optimización univariable con el punto inicial, el límite de búsqueda, la función objetivo, la tolerancia y el número máximo de iteraciones.

            Parámetros:
            - x_inicial (float): Punto inicial para la búsqueda.
            - xlimite (float): Límite superior del intervalo de búsqueda.
            - f (function or funcion): Función a optimizar. Puede ser una función ordinaria o un objeto de la clase 'funcion'.
            - epsilon (float): Tolerancia para la convergencia del método.
            - iter (int, opcional): Número máximo de iteraciones (predeterminado: 100).
            """
            super().__init__(x_inicial, xlimite, f, epsilon, iter)


    class derivative_methods(optimizador_univariable):
        def __init__(self, x_inicial, xlimite, f, epsilon, iter=100):
            """
            Inicializa el método de métodos derivativos para optimización univariable con el punto inicial, el límite de búsqueda, la función objetivo, la tolerancia y el número máximo de iteraciones.

            Parámetros:
            - x_inicial (float): Punto inicial para la búsqueda.
            - xlimite (float): Límite superior del intervalo de búsqueda.
            - f (function or funcion): Función a optimizar. Puede ser una función ordinaria o un objeto de la clase 'funcion'.
            - epsilon (float): Tolerancia para la convergencia del método.
            - iter (int, opcional): Número máximo de iteraciones (predeterminado: 100).
            """
            super().__init__(x_inicial, xlimite, f, epsilon, iter)


    class direct_methods(optimizador_multivariable):
        def __init__(self, variables, f, epsilon, iter=100):
            """
            Inicializa el método de métodos directos para optimización multivariable con las variables de entrada, la función objetivo, la tolerancia y el número máximo de iteraciones.

            Parámetros:
            - variables (list or np.array): Variables de entrada para la optimización.
            - f (function or funcion): Función a optimizar. Puede ser una función ordinaria o un objeto de la clase 'funcion'.
            - epsilon (float): Tolerancia para la convergencia del método.
            - iter (int, opcional): Número máximo de iteraciones (predeterminado: 100).
            """
            super().__init__(variables, f, epsilon, iter)


    class gradient_methods(optimizador_multivariable):
        def __init__(self, variables, f, epsilon, iter=100):
            """
            Inicializa el método de métodos de gradiente para optimización multivariable con las variables de entrada, la función objetivo, la tolerancia y el número máximo de iteraciones.

            Parámetros:
            - variables (list or np.array): Variables de entrada para la optimización.
            - f (function or funcion): Función a optimizar. Puede ser una función ordinaria o un objeto de la clase 'funcion'.
            - epsilon (float): Tolerancia para la convergencia del método.
            - iter (int, opcional): Número máximo de iteraciones (predeterminado: 100).
            """
            super().__init__(variables, f, epsilon, iter)

    
Métodos univariables
----------------------------------
Método de biseccion 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    class biseccion(derivative_methods):
    def __init__(self, x_inicial, xlimite, f, epsilon, iter=100):
        """
        Inicializa el método de bisección para encontrar raíces de una función.

        Parámetros:
        - x_inicial (float): Punto inicial para la búsqueda de la raíz.
        - xlimite (float): Límite superior del intervalo de búsqueda.
        - f (function): Función objetivo cuya raíz se desea encontrar.
        - epsilon (float): Tolerancia para la convergencia del método.
        - iter (int, opcional): Número máximo de iteraciones (predeterminado: 100).
        """
        super().__init__(x_inicial, xlimite, f, epsilon, iter)

    def primeraderivadanumerica(self, x_actual):
        """
        Calcula la primera derivada numérica de la función en un punto dado.

        Parámetros:
        - x_actual (float): Punto en el cual se evalúa la derivada.

        Retorna:
        - float: Valor numérico de la primera derivada en x_actual.
        """
        delta = 0.0001
        numerador = self.funcion(x_actual + delta) - self.funcion(x_actual - delta)
        return numerador / (2 * delta)

    def segundaderivadanumerica(self, x_actual):
        """
        Calcula la segunda derivada numérica de la función en un punto dado.

        Parámetros:
        - x_actual (float): Punto en el cual se evalúa la derivada.

        Retorna:
        - float: Valor numérico de la segunda derivada en x_actual.
        """
        delta = 0.0001
        numerador = self.funcion(x_actual + delta) - 2 * self.funcion(x_actual) + self.funcion(x_actual - delta)
        return numerador / (delta**2)

    def optimize(self):
        """
        Método principal de la funcion. La velocidad de este método depende la obtencion de a y b.

        Retorna:
        - float: Valor aproximado de la raíz encontrada.
        """
        a = np.random.uniform(self.valor_inicial, self.limite)
        b = np.random.uniform(self.valor_inicial, self.limite)

        while self.primeraderivadanumerica(a) > 0:
            a = np.random.uniform(self.valor_inicial, self.limite)

        while self.primeraderivadanumerica(b) < 0:
            b = np.random.uniform(self.valor_inicial, self.limite)

        x1 = a
        x2 = b
        z = (x2 + x1) / 2

        while self.primeraderivadanumerica(z) > self.epsilon:
            if self.primeraderivadanumerica(z) < 0:
                x1 = z
                z = (x2 + x1) / 2
            elif self.primeraderivadanumerica(z) > 0:
                x2 = z
                z = (x2 + x1) / 2

        return (x1 + x2) / 2


**Ejemplo**


.. code-block:: python
    from ..metodos_univariables import biseccion
    import numpy as np

    def funcion_ejemplo(x):
        return x**2 - 4*x + 4

    optimizador = biseccion(x_inicial=0, xlimite=5, f=funcion_ejemplo, epsilon=0.0001, iter=100)
    resultado = optimizador.optimize()

    print(f"El mínimo aproximado de la función es: {resultado}")
    Método de Fibonacci
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    class fibonacci(by_regions_elimination):
    def __init__(self, x_inicial, x_limite, f, iter=100):
        """
        Inicializa el método de optimización Fibonacci con el punto inicial, el límite de búsqueda, la función objetivo y el número máximo de iteraciones.

        Parámetros:
        - x_inicial (float): Punto inicial para la búsqueda.
        - x_limite (float): Límite superior del intervalo de búsqueda.
        - f (function or funcion): Función a optimizar. Puede ser una función ordinaria o un objeto de la clase 'funcion'.
        - iter (int, opcional): Número máximo de iteraciones (predeterminado: 100).
        """
        super().__init__(x_inicial, x_limite, f, iter)

    def findregions(self, rangomin, rangomax, x1, x2):
        """
        Encuentra las regiones mínima y máxima basadas en la evaluación de dos puntos.

        Parámetros:
        - rangomin (float): Límite inferior de la región.
        - rangomax (float): Límite superior de la región.
        - x1 (float): Primer punto a evaluar.
        - x2 (float): Segundo punto a evaluar.

        Retorna:
        - float, float: Nuevos límites de región (rangomin, rangomax).
        """
        if self.funcion(x1) > self.funcion(x2):
            rangomin = rangomin
            rangomax = x2
        elif self.funcion(x1) < self.funcion(x2):
            rangomin = x1
            rangomax = rangomax
        elif self.funcion(x1) == self.funcion(x2):
            rangomin = x1
            rangomax = x2
        return rangomin, rangomax

    def fibonacci_iterativo(self, n):
        """
        Genera una secuencia iterativa de números Fibonacci hasta el enésimo término.

        Parámetros:
        - n (int): Número de términos Fibonacci a generar.

        Retorna:
        - list: Lista de números Fibonacci generados.
        """
        fibonacci = [0, 1]
        for i in range(2, n):
            fibonacci.append(fibonacci[i - 1] + fibonacci[i - 2])
        return fibonacci

    def calculo_lk(self, fibonacci, n, k):
        """
        Calcula el valor de L_k usando la secuencia Fibonacci para el método.

        Parámetros:
        - fibonacci (list): Secuencia Fibonacci generada.
        - n (int): Número de términos Fibonacci generados.
        - k (int): Índice k para calcular L_k.

        Retorna:
        - float: Valor calculado de L_k.
        """
        indice1 = n - (k + 1)
        indice2 = n + 1
        return fibonacci[indice1] / fibonacci[indice2]

    def optimize(self):
        """
        Es el metodo principal de clase para optimizar

        Retorna:
        - float: Punto óptimo encontrado.
        """
        a, b = self.valor_inicial, self.limite
        n = self.iteraciones
        l = b - a
        serie_fibonacci = self.fibonacci_iterativo(n * 10)
        k = 2
        lk = self.calculo_lk(serie_fibonacci, n, k)
        x1 = a + lk
        x2 = b - lk
        while k != n:
            if k % 2 == 0:
                evalx1 = self.funcion(x1)
                a, b = self.findregions(a, b, evalx1, x2)
            else:
                evalx2 = self.funcion(x2)
                a, b = self.findregions(a, b, x1, evalx2)
            k += 1
        return (a + b) / 2



**Ejemplo**


.. code-block::python
    from optimizador_joshua_8.metodos_univariables.univariablefunction import univariablefunction
    from optimizador_joshua_8.metodos_univariables.fibonacci import fibonacci


    funciones = univariablefunction(name='funcion1')
    funcion_optimizar = funciones.get_function()

    optimizador = fibonacci(x_inicial=0.1, x_limite=5, f=funcion_optimizar, iter=1000)
    resultado = optimizador.optimize()

    print(f"El mínimo aproximado de la función es: {resultado}")
Método de busqueda dorada
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    import numpy as np
    class goldensearch(by_regions_elimination):
    def __init__(self, x_inicial, x_limite, f, epsilon):
        """
        Inicializa el método de búsqueda dorada con el punto inicial, el límite de búsqueda, la función objetivo y la tolerancia.

        Parámetros:
        - x_inicial (float): Punto inicial para la búsqueda.
        - x_limite (float): Límite superior del intervalo de búsqueda.
        - f (function or funcion): Función a optimizar. Puede ser una función ordinaria o un objeto de la clase 'funcion'.
        - epsilon (float): Tolerancia para la convergencia del método.
        """
        super().__init__(x_inicial, x_limite, f, epsilon)
    
    def findregions(self, x1, x2, fx1, fx2, a, b):
        """
        Encuentra las regiones de búsqueda en base a la evaluación de dos puntos.

        Parámetros:
        - x1 (float): Primer punto de evaluación.
        - x2 (float): Segundo punto de evaluación.
        - fx1 (float): Valor de la función en el punto x1.
        - fx2 (float): Valor de la función en el punto x2.
        - a (float): Límite inferior del intervalo de búsqueda.
        - b (float): Límite superior del intervalo de búsqueda.

        Retorna:
        - float, float: Nuevos límites de región (a, b).
        """
        if fx1 > fx2:
            return x1, b
        if fx1 < fx2:
            return a, x2
        return x1, x2 

    def w_to_x(self, w, a, b):
        """
        Convierte un valor w (en el intervalo [0, 1]) a un valor en el intervalo [a, b].

        Parámetros:
        - w (float): Valor w en el intervalo [0, 1].
        - a (float): Límite inferior del intervalo de búsqueda.
        - b (float): Límite superior del intervalo de búsqueda.

        Retorna:
        - float: Valor convertido en el intervalo [a, b].
        """
        return w * (b - a) + a 

    def optimize(self):
        """
        Es el metodo principal de la clase.

        Retorna:
        - float: Punto óptimo encontrado.
        """
        a, b = self.valor_inicial, self.limite
        phi = (1 + np.sqrt(5)) / 2 - 1
        aw, bw = 0, 1
        Lw = 1
        k = 1
        while Lw > self.epsilon:
            w2 = aw + phi * Lw
            w1 = bw - phi * Lw
            aw, bw = self.findregions(w1, w2, self.funcion(self.w_to_x(w1, a, b)), self.funcion(self.w_to_x(w2, a, b)), aw, bw)
            k += 1
            Lw = bw - aw
        return (self.w_to_x(aw, a, b) + self.w_to_x(bw, a, b)) / 2



**Ejemplo**


.. code-block::python
    from optimizador_joshua_8.metodos_univariables.univariablefunction import univariablefunction
    from optimizador_joshua_8.metodos_univariables.golden import goldensearch


    funciones = univariablefunction(name='funcion1')
    funcion_optimizar = funciones.get_function()

    optimizador = goldensearch(x_inicial=0.1, x_limite=5, f=funcion_optimizar, iter=10)
    resultado = optimizador.optimize()

    print(f"El mínimo aproximado de la función es: {resultado}")
Método de intervalos a la mitad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block::python
    class interval(by_regions_elimination):
        def __init__(self, x_inicial, x_limite, f, epsilon):
            """
            Inicializa el método de búsqueda por intervalos con el punto inicial, el límite de búsqueda, la función objetivo y la tolerancia.

            Parámetros:
            - x_inicial (float): Punto inicial para la búsqueda.
            - x_limite (float): Límite superior del intervalo de búsqueda.
            - f (function or funcion): Función a optimizar. Puede ser una función ordinaria o un objeto de la clase 'funcion'.
            - epsilon (float): Tolerancia para la convergencia del método.
            """
            super().__init__(x_inicial, x_limite, f, epsilon)
        
        def findregions(self, rangomin, rangomax, x1, x2):
            """
            Encuentra las regiones de búsqueda basadas en la evaluación de dos puntos.

            Parámetros:
            - rangomin (float): Límite inferior de la región.
            - rangomax (float): Límite superior de la región.
            - x1 (float): Primer punto a evaluar.
            - x2 (float): Segundo punto a evaluar.

            Retorna:
            - float, float: Nuevos límites de región (rangomin, rangomax).
            """
            if self.funcion(x1) > self.funcion(x2):
                rangomin = rangomin
                rangomax = x2
            elif self.funcion(x1) < self.funcion(x2):
                rangomin = x1
                rangomax = rangomax
            elif self.funcion(x1) == self.funcion(x2):
                rangomin = x1
                rangomax = x2
            return rangomin, rangomax

        def intervalstep3(self, b, x1, xm):
            """
            Realiza la validicion del valor actual de x1 para actualizar o no el valor de b y xm

            Parámetros:
            - b (float): Límite superior actual del intervalo.
            - x1 (float): Punto de búsqueda a la izquierda de xm.
            - xm (float): Punto medio del intervalo actual.

            Retorna:
            - float, float, bool: Nuevos límites (b, xm) y un indicador booleano que indica si se realizó un cambio.
            """
            if self.funcion(x1) < self.funcion(xm):
                b = xm
                xm = x1
                return b, xm, True
            else:
                return b, xm, False

        def intervalstep4(self, a, x2, xm):
            """
            Realiza la validicion del valor actual de x2 para actualizar o no el valor de a y xm

            Parámetros:
            - a (float): Límite inferior actual del intervalo.
            - x2 (float): Punto de búsqueda a la derecha de xm.
            - xm (float): Punto medio del intervalo actual.

            Retorna:
            - float, float, bool: Nuevos límites (a, xm) y un indicador booleano que indica si se realizó un cambio.
            """
            if self.funcion(x2) < self.funcion(xm):
                a = xm
                xm = x2
                return a, xm, True
            else:
                return a, xm, False

        def intervalstep5(self, b, a):
            """
            Esta funcion es la que valida si se cumple o no la condicion de paro

            Parámetros:
            - b (float): Límite superior del intervalo actual.
            - a (float): Límite inferior del intervalo actual.

            Retorna:
            - bool: Indicador booleano que indica si se debe continuar con las iteraciones (True) o no (False).
            """
            l = b - a
            if abs(l) < self.epsilon:
                return False
            else:
                return True

        def optimize(self):
            """
            Es el metodo princiapl de la funcion para optimizar.

            Retorna:
            - float: Punto óptimo encontrado.
            """
            a, b = self.valor_inicial, self.limite
            xm = (a + b) / 2
            l = b - a
            x1 = a + (l / 4)
            x2 = b - (l / 4)
            a, b = self.findregions(a, b, x1, x2)
            endflag = self.intervalstep5(a, b)
            l = b - a
            while endflag:
                x1 = a + (l / 4)
                x2 = b - l / 4
                b, xm, flag3 = self.intervalstep3(b, x1, xm)
                a, xm, flag4 = self.intervalstep4(a, x2, xm)
                if flag3:
                    endflag = self.intervalstep5(a, b)
                elif not flag3:
                    a, xm, flag4 = self.intervalstep4(a, x2, xm)

                if flag4:
                    endflag = self.intervalstep5(a, b)
                elif not flag4:
                    a = x1
                    b = x2
                    endflag = self.intervalstep5(a, b)
            return xm



**Ejemplo**


.. code-block::python
    from optimizador_joshua_8.metodos_univariables.univariablefunction import univariablefunction
    from optimizador_joshua_8.metodos_univariables.interval import interval


    funciones = univariablefunction(name='funcion1')
    funcion_optimizar = funciones.get_function()

    optimizador = interval(x_inicial=0.1, x_limite=5, f=funcion_optimizar, iter=10)
    resultado = optimizador.optimize()

    print(f"El mínimo aproximado de la función es: {resultado}")
Método de newton -raphson 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block::python
    class newton_raphson(derivative_methods):
        def __init__(self, x_inicial, xlimite=1, f=None, epsilon=0.1, iter=100):
            """
            Inicializa el método de Newton-Raphson con el punto inicial, el límite de búsqueda opcional, la función objetivo opcional y la tolerancia.

            Parámetros:
            - x_inicial (float): Punto inicial para la búsqueda.
            - xlimite (float, opcional): Límite superior del intervalo de búsqueda. Por defecto es 1.
            - f (function or None, opcional): Función a optimizar. Puede ser una función ordinaria o un objeto de la clase 'funcion'. Por defecto es None.
            - epsilon (float, opcional): Tolerancia para la convergencia del método. Por defecto es 0.1.
            - iter (int, opcional): Número máximo de iteraciones. Por defecto es 100.
            """
            super().__init__(x_inicial, xlimite, f, epsilon, iter)

        def primeraderivadanumerica(self, x_actual):
            """
            Calcula la primera derivada numérica de la función en un punto dado.

            Parámetros:
            - x_actual (float): Punto en el cual se evalúa la derivada.

            Retorna:
            - float: Valor de la primera derivada numérica en x_actual.
            """
            delta = 0.0001
            numerador = self.funcion(x_actual + delta) - self.funcion(x_actual - delta)
            return numerador / (2 * delta)

        def segundaderivadanumerica(self, x_actual):
            """
            Calcula la segunda derivada numérica de la función en un punto dado.

            Parámetros:
            - x_actual (float): Punto en el cual se evalúa la derivada.

            Retorna:
            - float: Valor de la segunda derivada numérica en x_actual.
            """
            delta = 0.0001
            numerado = self.funcion(x_actual + delta) - 2 * self.funcion(x_actual) + self.funcion(x_actual - delta)
            return numerado / (delta ** 2)

        def optimize(self):
            """
            Es la funcion principal de la clase

            Retorna:
            - float: Punto óptimo encontrado.
            """
            k = 1
            x_actual = self.valor_inicial
            
            # Calcula la primera y segunda derivada numérica en el punto inicial
            xderiv = self.primeraderivadanumerica(x_actual)
            xderiv2 = self.segundaderivadanumerica(x_actual)
            
            # Calcula el siguiente punto usando la fórmula de Newton-Raphson
            xsig = x_actual - (xderiv / xderiv2)
            
            # Itera hasta que se cumpla el criterio de convergencia
            while abs(self.primeraderivadanumerica(xsig)) > self.epsilon:
                x_actual = xsig
                xderiv = self.primeraderivadanumerica(x_actual)
                xderiv2 = self.segundaderivadanumerica(x_actual)
                xsig = x_actual - (xderiv / xderiv2)
            
            return xsig



**Ejemplo**


.. code-block::python
    from optimizador_joshua_8.metodos_univariables.univariablefunction import univariablefunction
    from optimizador_joshua_8.metodos_univariables.newtonraphson import newton_raphson


    funciones = univariablefunction(name='funcion1')
    funcion_optimizar = funciones.get_function()

    optimizador = newton_raphson(x_inicial=0.1, x_limite=5, f=funcion_optimizar, iter=10)
    resultado = optimizador.optimize()

    print(f"El mínimo aproximado de la función es: {resultado}")
Método de la secante
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block::python
    class secante(derivative_methods):
        def __init__(self, x_inicial, xlimite, f, epsilon, iter=100):
            """
            Inicializa el método de la secante con el punto inicial, el límite de búsqueda, la función objetivo y la tolerancia.

            Parámetros:
            - x_inicial (float): Punto inicial para la búsqueda.
            - xlimite (float): Límite superior del intervalo de búsqueda.
            - f (function): Función a optimizar.
            - epsilon (float): Tolerancia para la convergencia del método.
            - iter (int, opcional): Número máximo de iteraciones. Por defecto es 100.
            """
            super().__init__(x_inicial, xlimite, f, epsilon, iter)
        
        def primeraderivadanumerica(self, x_actual):
            """
            Calcula la primera derivada numérica de la función en un punto dado.

            Parámetros:
            - x_actual (float): Punto en el cual se evalúa la derivada.

            Retorna:
            - float: Valor de la primera derivada numérica en x_actual.
            """
            delta = 0.0001
            numerador = self.funcion(x_actual + delta) - self.funcion(x_actual - delta)
            return numerador / (2 * delta)

        def segundaderivadanumerica(self, x_actual):
            """
            Calcula la segunda derivada numérica de la función en un punto dado.

            Parámetros:
            - x_actual (float): Punto en el cual se evalúa la derivada.

            Retorna:
            - float: Valor de la segunda derivada numérica en x_actual.
            """
            delta = 0.0001
            numerado = self.funcion(x_actual + delta) - 2 * self.funcion(x_actual) + self.funcion(x_actual - delta)
            return numerado / (delta ** 2)

        def calculozensecante(self, x2, x1):
            """
            Método principal de la funcion. La velocidad de este método depende la obtencion de a y b.

            Parámetros:
            - x2 (float): Punto actual.
            - x1 (float): Punto anterior.

            Retorna:
            - float: Siguiente punto calculado con el método de la secante.
            """
            numerador = self.primeraderivadanumerica(x2)
            denominador = (self.primeraderivadanumerica(x2) - self.primeraderivadanumerica(x1)) / (x2 - x1)
            op = numerador / denominador
            return x2 - op

        def optimize(self):
            """
            Implementa el método de la secante para encontrar el punto crítico de la función objetivo.

            Retorna:
            - float: Punto óptimo encontrado.
            """
            # Inicialización de a y b con números aleatorios dentro del intervalo
            a = np.random.uniform(self.valor_inicial, self.limite)
            b = np.random.uniform(self.valor_inicial, self.limite)
            
            # Asegura que a y b estén en direcciones adecuadas
            while self.primeraderivadanumerica(a) > 0:
                a = np.random.uniform(self.valor_inicial, self.limite)
            
            while self.primeraderivadanumerica(b) < 0:
                b = np.random.uniform(self.valor_inicial, self.valor_inicial)
            
            # Inicialización de x1 y x2
            x1 = a
            x2 = b
            
            # Calcula el siguiente punto usando el método de la secante
            z = self.calculozensecante(x2, x1)
            
            # Itera hasta que se cumpla el criterio de convergencia
            while self.primeraderivadanumerica(z) > self.epsilon:
                if self.primeraderivadanumerica(z) < 0:
                    x1 = z
                    z = self.calculozensecante(x2, x1)
                if self.primeraderivadanumerica(z) > 0:
                    x2 = z
                    z = self.calculozensecante(x2, x1)
            
            return (x1 + x2) / 2


**Ejemplo**

.. code-block::python
    from optimizador_joshua_8.metodos_univariables.univariablefunction import univariablefunction
    from optimizador_joshua_8.metodos_univariables.secante import secante


    funciones = univariablefunction(name='funcion1')
    funcion_optimizar = funciones.get_function()

    optimizador = secante(x_inicial=0.1, x_limite=5, f=funcion_optimizar, iter=10)
    resultado = optimizador.optimize()

    print(f"El mínimo aproximado de la función es: {resultado}")
Métodos Multivariables
----------------------------------
Método de cauchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python 
    from ..baseopt import gradient_methods, optimizador_univariable
    from ..metodos_univariables import goldensearch, fibonacci, interval, biseccion, secante
    import numpy as np
    from ..funciones.base import funcion

    class cauchy(gradient_methods):
    def __init__(self, variables, f: funcion, epsilon, epsilon2, iter=100, opt: optimizador_univariable = goldensearch):
        """
        Inicializa el método de Cauchy para la optimización de funciones multivariables. En el caso de los métodos de gradiente, solo pueden recibir la funciones
        pertenencientes a tipo función en esta version debido a la necesidad de utilizar un espacio de busqueda

        Parámetros:
        - variables (list): Lista de variables de la función.
        - f (funcion): Función a optimizar.
        - epsilon (float): Tolerancia para la norma del gradiente.
        - epsilon2 (float): Tolerancia para la convergencia del paso.
        - iter (int, opcional): Número máximo de iteraciones. Por defecto es 100.
        - opt (optimizador_univariable, opcional): Método de optimización univariable. Por defecto es Goldensearch. Para utilizar se debe escribir:
        goldensearch 
        fibonacci 
        interval 
        biseccion 
        secante 
        """
        super().__init__(variables, f, epsilon, iter)
        self.epsilon2 = epsilon2
        self.opt = opt
        self.gradiente = []
        self.data = f

    def testalpha(self, alfa):
        """
        Evalúa la función objetivo en un punto dado por un paso alfa en dirección opuesta al gradiente. Esta función no se utiliza

        Parámetros:
        - alfa (float): Tamaño del paso alfa.

        Retorna:
        - float: Valor de la función en el punto evaluado.
        """
        return self.funcion(self.variables - (alfa * np.array(self.gradiente)))
    
    def gradiente_calculation(self, x, delta=0.0001):
        """
        Calcula el gradiente de la función en un punto dado utilizando diferencias finitas centradas.

        Parámetros:
        - x (list): Punto en el cual se evalúa el gradiente.
        - delta (float, opcional): Paso para el cálculo numérico de la derivada. Por defecto es 0.0001.

        Retorna:
        - list: Lista con los valores del gradiente en el punto x.
        """
        if delta is None:
            delta = 0.00001
        
        vector_f1_prim = []
        x_work = np.array(x)
        x_work_f = x_work.astype(np.float64)
        
        if isinstance(delta, int) or isinstance(delta, float):
            for i in range(len(x_work_f)):
                point = np.array(x_work_f, copy=True)
                vector_f1_prim.append(self.primeraderivadaop(point, i, delta))
            return vector_f1_prim
        else:
            for i in range(len(x_work_f)):
                point = np.array(x_work_f, copy=True)
                vector_f1_prim.append(self.primeraderivadaop(point, i, delta[i]))
            return vector_f1_prim

    def primeraderivadaop(self, x, i, delta):
        """
        Calcula la primera derivada parcial de la función en un punto dado utilizando diferencias finitas centradas.

        Parámetros:
        - x (list): Punto en el cual se evalúa la derivada.
        - i (int): Índice de la variable respecto a la cual se calcula la derivada.
        - delta (float): Paso para el cálculo numérico de la derivada.

        Retorna:
        - float: Valor de la primera derivada parcial en el punto x respecto a la variable i.
        """
        mof = x[i]
        p = np.array(x, copy=True)
        p2 = np.array(x, copy=True)
        nump1 = mof + delta
        nump2 = mof - delta
        p[i] = nump1
        p2[i] = nump2
        numerador = self.funcion(p) - self.funcion(p2)
        return numerador / (2 * delta)

    def segundaderivadaop(self, x, i, delta):
        """
        Calcula la segunda derivada parcial de la función en un punto dado utilizando diferencias finitas centradas.

        Parámetros:
        - x (list): Punto en el cual se evalúa la derivada.
        - i (int): Índice de la variable respecto a la cual se calcula la derivada.
        - delta (float): Paso para el cálculo numérico de la derivada.

        Retorna:
        - float: Valor de la segunda derivada parcial en el punto x respecto a la variable i.
        """
        mof = x[i]
        p = np.array(x, copy=True)
        p2 = np.array(x, copy=True)
        nump1 = mof + delta
        nump2 = mof - delta
        p[i] = nump1
        p2[i] = nump2
        numerador = self.funcion(p) - 2 * self.funcion(x) + self.funcion(p2)
        return numerador / (delta ** 2)

    def derivadadodadoop(self, x, index_principal, index_secundario, delta):
        """
        Calcula la derivada doble cruzada de la función en un punto dado utilizando diferencias finitas centradas.

        Parámetros:
        - x (list): Punto en el cual se evalúa la derivada.
        - index_principal (int): Índice de la primera variable respecto a la cual se calcula la derivada.
        - index_secundario (int): Índice de la segunda variable respecto a la cual se calcula la derivada.
        - delta (float o list): Paso para el cálculo numérico de la derivada.

        Retorna:
        - float: Valor de la derivada doble cruzada en el punto x respecto a las variables index_principal e index_secundario.
        """
        mof = x[index_principal]
        mof2 = x[index_secundario]
        p = np.array(x, copy=True)
        p2 = np.array(x, copy=True)
        p3 = np.array(x, copy=True)
        p4 = np.array(x, copy=True)
        
        if isinstance(delta, int) or isinstance(delta, float):
            mod1 = mof + delta
            mod2 = mof - delta
            mod3 = mof2 + delta
            mod4 = mof2 - delta
            
            p[index_principal] = mod1
            p[index_secundario] = mod3
            p2[index_principal] = mod1
            p2[index_secundario] = mod4
            p3[index_principal] = mod2
            p3[index_secundario] = mod3
            p4[index_principal] = mod2
            p4[index_secundario] = mod4
            
            numerador = (self.funcion(p) - self.funcion(p2) - self.funcion(p3) + self.funcion(p4))
            return numerador / (4 * delta * delta)
        
        else:
            mod1 = mof + delta[index_principal]
            mod2 = mof - delta[index_principal]
            mod3 = mof2 + delta[index_secundario]
            mod4 = mof2 - delta[index_secundario]
            
            p[index_principal] = mod1
            p[index_secundario] = mod3
            p2[index_principal] = mod1
            p2[index_secundario] = mod4
            p3[index_principal] = mod2
            p3[index_secundario] = mod3
            p4[index_principal] = mod2
            p4[index_secundario] = mod4
            
            numerador = (self.funcion(p) - self.funcion(p2) - self.funcion(p3) + self.funcion(p4))
            return numerador / (4 * delta[index_principal] * delta[index_secundario])

    def hessian_matrix(self, x, delt=0.0001):
        """
        Calcula la matriz Hessiana de la función en un punto dado utilizando diferencias finitas centradas.

        Parámetros:
        - x (list): Punto en el cual se evalúa la matriz Hessiana.
        - delt (float o list, opcional): Paso para el cálculo numérico de las derivadas. Por defecto es 0.0001.

        Retorna:
        - list of lists: Matriz Hessiana evaluada en el punto x.
        """
        matrix_f2_prim = [[0]*len(x) for i in range(len(x))]
        x_work = np.array(x)
        x_work_f = x_work.astype(np.float64)
        
        for i in range(len(x)):
            point = np.array(x_work_f, copy=True)
            for j in range(len(x)):
                if i == j:
                    matrix_f2_prim[i][j] = self.segundaderivadaop(point, i, delt)
                else:
                    matrix_f2_prim[i][j] = self.derivadadodadoop(point, i, j, delt)
        
        return matrix_f2_prim
    
    def optimizaralpha(self, test):
        """
        Optimiza el tamaño del paso alfa utilizando el método univariable especificado. Este es el método por el cual es necesario utilizar las funciones pertenencientes
        a la clase.

        Parámetros:
        - test (funcion): Función a optimizar respecto al paso alfa.

        Retorna:
        - float: Tamaño óptimo del paso alfa.
        """
        a = self.data.get_limiteinf()
        b = self.data.get_limitesup()
        opt = self.opt(a, b, test, self.epsilon)
        alfa = opt.optimize()
        return alfa
    
    def optimize(self):
        """
        Es el método principal de la funcion 

        Retorna:
        - list: Lista con los valores óptimos de las variables encontradas.
        """
        terminar = False
        xk = self.variables
        k = 0
        
        while not terminar:
            grad = np.array(self.gradiente_calculation(xk))
            
            if np.linalg.norm(grad) < self.epsilon or k >= self.iteraciones:
                terminar = True
            else:
                def alpha_funcion(alpha):
                    return self.funcion(xk - alpha * grad)
                
                alpha = self.optimizaralpha(alpha_funcion)
                xk_1 = xk - alpha * grad
                
                if np.linalg.norm(xk_1 - xk) / (np.linalg.norm(xk) + 0.0001) < self.epsilon2:
                    terminar = True
                
                xk = xk_1
                k += 1
        
        print(f"Iteraciones totales: {k}")
        return xk




**Ejemplo**
.. code-block::python
    import numpy as np
    from optimizador_joshua_8.metodos_multivariables.cauchy import cauchy
    from optimizador_joshua_8.funciones.objetivo import objetive_function
    func_name = "himmelblau"
    espaciobusqueda = np.array([[0, 0], [1, 1]])
    obj_func = objetive_function(func_name, espaciobusqueda)
    epsilon = 0.01
    inicial = [1, 1]
    c = cauchy(inicial, obj_func, epsilon=epsilon, epsilon2=epsilon)
    print("Resultado del método Cauchy:", c.optimize())


Método de fletcher-reeves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    from ..baseopt import gradient_methods, optimizador_univariable
    from ..metodos_univariables import goldensearch, fibonacci, interval, biseccion, secante
    import numpy as np
    from ..funciones.base import funcion

    class fletcher_reeves(gradient_methods):
    def __init__(self, variables, f: funcion, epsilon, epsilon2, epsilon3, iter=100, opt: optimizador_univariable = goldensearch):
        """
        Inicializa el método de Fletcher-reeves para la optimización de funciones multivariables. En el caso de los métodos de gradiente, solo pueden recibir la funciones
        pertenencientes a tipo función en esta version debido a la necesidad de utilizar un espacio de busqueda

        Parámetros:
        - variables (list): Lista de variables de la función.
        - f (funcion): Función a optimizar.
        - epsilon (float): Tolerancia para la norma del gradiente.
        - epsilon2 (float): Tolerancia para la convergencia de las variables.
        - epsilon3 (float): Tolerancia para la norma del gradiente en la condición de parada.
        - iter (int, opcional): Número máximo de iteraciones. Por defecto es 100.
        - opt (optimizador_univariable, opcional): Método de optimización univariable. Por defecto es Goldensearch. Para utilizar se debe escribir:
        goldensearch 
        fibonacci 
        interval 
        biseccion 
        secante 
        
        """
        super().__init__(variables, f, epsilon, iter)
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        self.opt = opt
        self.gradiente = []
        self.data = f

    def testalpha(self, alfa):
        """
        Evalúa la función objetivo en un punto dado por un paso alfa en dirección opuesta al gradiente.Este metodo no se utiliza

        Parámetros:
        - alfa (float): Tamaño del paso alfa.

        Retorna:
        - float: Valor de la función en el punto evaluado.
        """
        return self.funcion(self.variables - (alfa * np.array(self.gradiente)))
    
    def gradiente_calculation(self, x, delta=0.0001):
        """
        Calcula el gradiente de la función en un punto dado.

        Parámetros:
        - x (list): Punto en el cual se evalúa el gradiente.
        - delta (float, opcional): Paso para el cálculo numérico de la derivada. Por defecto es 0.0001.

        Retorna:
        - list: Lista con los valores del gradiente en el punto x.
        """
        if delta is None: 
            delta = 0.00001
        
        vector_f1_prim = []
        x_work = np.array(x)
        x_work_f = x_work.astype(np.float64)
        
        if isinstance(delta, int) or isinstance(delta, float):
            for i in range(len(x_work_f)):
                point = np.array(x_work_f, copy=True)
                vector_f1_prim.append(self.primeraderivadaop(point, i, delta))
            return vector_f1_prim
        else:
            for i in range(len(x_work_f)):
                point = np.array(x_work_f, copy=True)
                vector_f1_prim.append(self.primeraderivadaop(point, i, delta[i]))
            return vector_f1_prim

    def primeraderivadaop(self, x, i, delta):
        """
        Calcula la primera derivada parcial de la función en un punto dado.

        Parámetros:
        - x (list): Punto en el cual se evalúa la derivada.
        - i (int): Índice de la variable respecto a la cual se calcula la derivada.
        - delta (float): Paso para el cálculo numérico de la derivada.

        Retorna:
        - float: Valor de la primera derivada parcial en el punto x respecto a la variable i.
        """
        mof = x[i]
        p = np.array(x, copy=True)
        p2 = np.array(x, copy=True)
        nump1 = mof + delta
        nump2 = mof - delta
        p[i] = nump1
        p2[i] = nump2
        numerador = self.funcion(p) - self.funcion(p2)
        return numerador / (2 * delta) 
    
    def segundaderivadaop(self, x, i, delta):
        """
        Calcula la segunda derivada parcial de la función en un punto dado.

        Parámetros:
        - x (list): Punto en el cual se evalúa la derivada.
        - i (int): Índice de la variable respecto a la cual se calcula la derivada.
        - delta (float): Paso para el cálculo numérico de la derivada.

        Retorna:
        - float: Valor de la segunda derivada parcial en el punto x respecto a la variable i.
        """
        mof = x[i]
        p = np.array(x, copy=True)
        p2 = np.array(x, copy=True)
        nump1 = mof + delta
        nump2 = mof - delta
        p[i] = nump1
        p2[i] = nump2
        numerador = self.funcion(p) - 2 * self.funcion(x) + self.funcion(p2)
        return numerador / (delta**2) 

    def derivadadodadoop(self, x, index_principal, index_secundario, delta):
        """
        Calcula la derivada doble cruzada de la función en un punto dado.

        Parámetros:
        - x (list): Punto en el cual se evalúa la derivada.
        - index_principal (int): Índice de la primera variable respecto a la cual se calcula la derivada.
        - index_secundario (int): Índice de la segunda variable respecto a la cual se calcula la derivada.
        - delta (float o list): Paso para el cálculo numérico de la derivada.

        Retorna:
        - float: Valor de la derivada doble cruzada en el punto x respecto a las variables index_principal e index_secundario.
        """
        mof = x[index_principal]
        mof2 = x[index_secundario]
        p = np.array(x, copy=True)
        p2 = np.array(x, copy=True)
        p3 = np.array(x, copy=True)
        p4 = np.array(x, copy=True)
        
        if isinstance(delta, int) or isinstance(delta, float):
            mod1 = mof + delta
            mod2 = mof - delta
            mod3 = mof2 + delta
            mod4 = mof2 - delta
            
            p[index_principal] = mod1
            p[index_secundario] = mod3
            p2[index_principal] = mod1
            p2[index_secundario] = mod4
            p3[index_principal] = mod2
            p3[index_secundario] = mod3
            p4[index_principal] = mod2
            p4[index_secundario] = mod4
            
            numerador = (self.funcion(p) - self.funcion(p2) - self.funcion(p3) + self.funcion(p4))
            return numerador / (4 * delta * delta)
        
        else:
            mod1 = mof + delta[index_principal]
            mod2 = mof - delta[index_principal]
            mod3 = mof2 + delta[index_secundario]
            mod4 = mof2 - delta[index_secundario]
            
            p[index_principal] = mod1
            p[index_secundario] = mod3
            p2[index_principal] = mod1
            p2[index_secundario] = mod4
            p3[index_principal] = mod2
            p3[index_secundario] = mod3
            p4[index_principal] = mod2
            p4[index_secundario] = mod4
            
            numerador = (self.funcion(p) - self.funcion(p2) - self.funcion(p3) + self.funcion(p4))
            return numerador / (4 * delta[index_principal] * delta[index_secundario])

    def hessian_matrix(self, x, delt=0.0001):
        """
        Calcula la matriz Hessiana de la función en un punto dado.

        Parámetros:
        - x (list): Punto en el cual se evalúa la matriz Hessiana.
        - delt (float o list, opcional): Paso para el cálculo numérico de las derivadas. Por defecto es 0.0001.

        Retorna:
        - list of lists: Matriz Hessiana evaluada en el punto x.
        """
        matrix_f2_prim = [[0]*len(x) for i in range(len(x))]
        x_work = np.array(x)
        x_work_f = x_work.astype(np.float64)
        
        for i in range(len(x)):
            point = np.array(x_work_f, copy=True)
            for j in range(len(x)):
                if i == j:
                    matrix_f2_prim[i][j] = self.segundaderivadaop(point, i, delt)
                else:
                    matrix_f2_prim[i][j] = self.derivadadodadoop(point, i, j, delt)
        
        return matrix_f2_prim
    
    def optimizaralpha(self, test):
        """
        Optimiza el tamaño del paso alfa utilizando un método univariable dado. Es debido a este método que se requiere la utilizacion de la clase función.

        Parámetros:
        - test (func): Función de prueba para evaluar diferentes tamaños de paso alfa.

        Retorna:
        - float: Tamaño óptimo del paso alfa.
        """
        a = self.data.get_limiteinf()
        b = self.data.get_limitesup()
        opt = self.opt(a, b, test, self.epsilon)
        alfa = opt.optimize()
        return alfa
    
    def s_sig_gradcon(self, gradiente_ac, gradiente_ant, s):
        """
        Calcula la dirección de descenso utilizando la fórmula de la dirección de descenso de Fletcher y Reeves.

        Parámetros:
        - gradiente_ac (list): Gradiente en la iteración actual.
        - gradiente_ant (list): Gradiente en la iteración anterior.
        - s (list): Dirección de descenso en la iteración anterior.

        Retorna:
        - list: Nueva dirección de descenso calculada.
        """
        beta = np.dot(gradiente_ac, gradiente_ac) / np.dot(gradiente_ant, gradiente_ant)
        return -gradiente_ac + beta * s

    def optimize(self):
        """
        Funcion principal de la clase para optimizar.

        Retorna:
        - list: Lista con los valores óptimos de las variables encontradas.
        """
        xk = np.array(self.variables)
        grad = self.gradiente_calculation(xk)
        sk = np.array(grad * -1)
        k = 1
        
        while (np.linalg.norm(grad) >= self.epsilon3) and (k <= self.iteraciones):
            def alpha_funcion(alpha):
                return self.funcion(xk + alpha * sk)
            
            alpha = self.optimizaralpha(alpha_funcion)
            xk_1 = xk + alpha * sk
            
            if np.linalg.norm(xk_1 - xk) / np.linalg.norm(xk) < self.epsilon2:
                break
            
            grad_1 = np.array(self.gradiente_calculation(xk_1))
            sk = self.s_sig_gradcon(grad_1, grad, sk)
            xk = xk_1
            grad = np.array(grad_1)
            k += 1
        
        return xk

**Ejemplo**
.. code-block::python
    import numpy as np
    from optimizador_joshua_8.metodos_multivariables.cauchy import cauchy
    from optimizador_joshua_8.funciones.objetivo import objetive_function
    func_name = "himmelblau"
    espaciobusqueda = np.array([[0, 0], [1, 1]])
    obj_func = objetive_function(func_name, espaciobusqueda)
    epsilon = 0.01
    inicial = [1, 1]
    g = fletcher_reeves(inicial, obj_func, epsilon=epsilon, epsilon2=epsilon, epsilon3=epsilon)
    print("Resultado del método Fletcher-Reeves:", g.optimize())


Método de Hooke-Jeeves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    import numpy as np

    class hooke_jeeves(direct_methods):
    def __init__(self, variables, delta, f, epsilon, alpha=2):
        """
        Inicializa el método de Hooke-Jeeves para la optimización de funciones.

        Parámetros:
        - variables (list): Lista de variables de la función.
        - delta (float or list): Tamaño del paso inicial para la exploración.
        - f (func): Función a optimizar.
        - epsilon (float): Tolerancia para la condición de parada.
        - alpha (float, opcional): Factor de reducción para actualizar delta. Por defecto es 2.
        """
        super().__init__(variables, f, epsilon)
        self.alpha = alpha
        self.delta = delta
    
    def movexploratory(self, basepoint, delta):
        """
        Realiza un movimiento exploratorio para explorar los vecinos de un punto base.

        Parámetros:
        - basepoint (list): Punto base a partir del cual se realizan las exploraciones.
        - delta (float or list): Tamaño del paso para la exploración en cada dimensión.

        Retorna:
        - tuple: Mejor punto encontrado y un indicador booleano de si se encontró un nuevo mejor punto.
        """
        nextpoint = []
        coordinates = [basepoint]
        newvalue = True
        
        # Generación de coordenadas vecinas
        for i in range(len(basepoint)):
            point = basepoint.copy()
            point2 = basepoint.copy()
            point[i] += delta[i]
            point2[i] -= delta[i]
            coordinates.append(point)
            coordinates.append(point2)
        
        # Evaluación de las coordenadas
        for coordinate in coordinates:
            nextpoint.append(self.funcion(coordinate))
        
        # Búsqueda del mínimo
        min_index = np.argmin(nextpoint)
        
        if (coordinates[min_index] == basepoint).all():
            newvalue = False
        
        return coordinates[min_index], newvalue

    def patternmove(self, currentbestpoint, lastbestpoint):
        """
        Realiza un movimiento de patrón para explorar en una dirección específica.

        Parámetros:
        - currentbestpoint (list): Mejor punto actual.
        - lastbestpoint (list): Mejor punto anterior.

        Retorna:
        - list: Nuevo punto a explorar.
        """
        basepoint = currentbestpoint + (currentbestpoint - lastbestpoint)
        return basepoint

    def updatedelta(self, delta):
        """
        Actualiza el tamaño del paso delta para la siguiente iteración.

        Parámetros:
        - delta (float or list): Tamaño del paso actual.

        Retorna:
        - float or list: Nuevo tamaño del paso delta.
        """
        new_delta = delta / self.alpha
        return new_delta

    def optimize(self):
        """
        Metodo principal de la clase.

        Retorna:
        - list: Lista con los valores óptimos de las variables encontradas.
        """
        cont = 0
        x_initial = np.array(self.variables)
        delta = np.array(self.delta)
        x_previous = x_initial
        x_best, flag = self.movexploratory(x_initial, delta)
        
        while np.linalg.norm(delta) > self.epsilon:
            if flag:
                x_point = self.patternmove(x_best, x_previous)
                x_best_new, flag = self.movexploratory(x_point, delta)
            else:
                delta = self.updatedelta(delta)
                x_best, flag = self.movexploratory(x_best, delta)
                x_point = self.patternmove(x_best, x_previous)
                x_best_new, flag = self.movexploratory(x_point, delta)
            
            if self.funcion(x_best_new) < self.funcion(x_best):
                flag = True
                x_previous = x_best
                x_best = x_best_new
            else:
                flag = False

            cont += 1
        
        print("Número de iteraciones: {}".format(cont))
        return x_best_new

**Ejemplo**
.. code-block::python
    from optimizador_joshua_8.metodos_multivariables.hookejeeves import hooke_jeeves
    from optimizador_joshua_8.funciones.objetivo import objetive_function
    #Hookejeeves 
    func_name = "booth"
    espaciobusqueda = np.array([[-5, -2.5], [5, 2.5]])
    obj_func = objetive_function(func_name, espaciobusqueda)
    x_inicial = [-5, -2.5]
    delta = [0.5, 0.25]
    alpha = 2
    e = 0.01
    hj = hooke_jeeves(x_inicial, delta,obj_func,e)
    print(hj.funcion)
    print("Resultado del método Hooke-Jeeves:", hj.optimize())
Método de Neldear-Mead
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    from ..baseopt import direct_methods
    import numpy as np
    class neldermead(direct_methods):
    def __init__(self, variables, gamma, beta, f, epsilon, iter=100):
        """
        Inicializa el método de Nelder-Mead para optimización.

        Parámetros:
        - variables: Lista de variables iniciales.
        - gamma: Parámetro de reflexión.
        - beta: Parámetro de contracción.
        - f: Función objetivo a optimizar.
        - epsilon: Tolerancia para la condición de parada.
        - iter: Número máximo de iteraciones (por defecto, 100).
        """
        super().__init__(variables, f, epsilon, iter)
        self.variables = np.array(variables)
        self.gamma = gamma
        self.beta = beta
    
    def delta1(self, N, scale):
        """
        Calcula el tamaño delta1 para crear un simplex.

        Parámetros:
        - N: Número de dimensiones (variables).
        - scale: Escala para ajustar el tamaño.

        Retorna:
        - Valor de delta1 para la creación del simplex.
        """
        num = np.sqrt(N + 1) + N - 1
        den = N * np.sqrt(2)
        op = num / den
        return op * scale

    def delta2(self, N, scale):
        """
        Calcula el tamaño delta2 para crear un simplex.

        Parámetros:
        - N: Número de dimensiones (variables).
        - scale: Escala para ajustar el tamaño.

        Retorna:
        - Valor de delta2 para la creación del simplex.
        """
        num = np.sqrt(N + 1) - 1
        den = N * np.sqrt(2)
        op = num / den
        return op * scale

    def create_simplex(self, initial_point, scale=1.0):
        """
        Crea un simplex inicial a partir de un punto inicial.

        Parámetros:
        - initial_point: Punto inicial alrededor del cual se crea el simplex.
        - scale: Escala para ajustar el tamaño del simplex (por defecto, 1.0).

        Retorna:
        - Simplex inicial creado alrededor del punto inicial.
        """
        n = len(initial_point)
        simplex = [np.array(initial_point, dtype=float)] 
        d1 = self.delta1(n, scale)
        d2 = self.delta2(n, scale)
        for i in range(n):
            point = np.array(simplex[0], copy=True)  
            for j in range(n):
                if j == i: 
                    point[j] += d1
                else:
                    point[j] += d2
            simplex.append(point)
        
        simplex_final = np.array(simplex)

        return np.round(simplex_final, 4)

    def findpoints(self, points):
        """
        Encuentra los mejores, segundo peor y peor puntos en el simplex.

        Parámetros:
        - points: Lista de puntos del simplex.

        Retorna:
        - Índices de los mejores, segundo peor y peor puntos en el simplex.
        """
        evaluaciones = [self.funcion(p) for p in points]
        worst = np.argmax(evaluaciones)
        best = np.argmin(evaluaciones)
        indices = list(range(len(evaluaciones)))
        indices.remove(worst)
        second_worst = indices[np.argmax([evaluaciones[i] for i in indices])]
        if second_worst == best:
            indices.remove(best)
            second_worst = indices[np.argmax([evaluaciones[i] for i in indices])]
        return best, second_worst, worst

    def xc_calculation(self, x, indices):
        """
        Calcula el centroide del simplex excluyendo el peor punto.

        Parámetros:
        - x: Puntos del simplex.
        - indices: Índices de los puntos a considerar.

        Retorna:
        - Centroide calculado.
        """
        m = x[indices]
        centro = []
        for i in range(len(m[0])):
            suma = sum(p[i] for p in m)
            v = suma / len(m)
            centro.append(v)
        return np.array(centro)

    def stopcondition(self, simplex, xc):
        """
        Condición de parada basada en la distancia entre los puntos del simplex y su centroide.

        Parámetros:
        - simplex: Simplex actual.
        - xc: Centroide del simplex.

        Retorna:
        - Valor de la condición de parada.
        """
        value = 0
        n = len(simplex)
        for i in range(n):
            value += (((self.funcion(simplex[i]) - self.funcion(xc))**2) / (n + 1))
        return np.sqrt(value)

    def optimize(self):
        """
        Método principal de la clase

        Retorna:
        - Mejor punto encontrado que minimiza la función objetivo.
        """
        cont = 1
        simplex = self.create_simplex(self.variables)
        best, second_worst, worst = self.findpoints(simplex)
        indices = [best, second_worst, worst]
        indices.remove(worst)
        centro = self.xc_calculation(simplex, indices)
        x_r = (2 * centro) - simplex[worst]
        x_new = x_r
        if self.funcion(x_r) < self.funcion(simplex[best]): 
            x_new = ((1 + self.gamma) * centro) - (self.gamma * simplex[worst])
        elif self.funcion(x_r) >= self.funcion(simplex[worst]):
            x_new = ((1 - self.beta) * centro) + (self.beta * simplex[worst])
        elif self.funcion(simplex[second_worst]) < self.funcion(x_r) and self.funcion(x_r) < self.funcion(simplex[worst]):
            x_new = ((1 - self.beta) * centro) - (self.beta * simplex[worst])
        simplex[worst] = x_new
        stop = self.stopcondition(simplex, centro)
        while stop >= self.epsilon:
            best, second_worst, worst = self.findpoints(simplex)
            indices = [best, second_worst, worst]
            indices.remove(worst)
            centro = self.xc_calculation(simplex, indices)
            x_r = (2 * centro) - simplex[worst]
            x_new = x_r
            if self.funcion(x_r) < self.funcion(simplex[best]):
                x_new = ((1 + self.gamma) * centro) - (self.gamma * simplex[worst])
            elif self.funcion(x_r) >= self.funcion(simplex[worst]):
                x_new = ((1 - self.beta) * centro) + (self.beta * simplex[worst])
            elif self.funcion(simplex[second_worst]) < self.funcion(x_r) and self.funcion(x_r) < self.funcion(simplex[worst]):
                x_new = ((1 + self.beta) * centro) - (self.beta * simplex[worst])
            simplex[worst] = x_new
            stop = self.stopcondition(simplex, centro)
            cont += 1
        
        return simplex[best]

**Ejemplo**
.. code-block::python
    from optimizador_joshua_8.metodos_multivariables.neldermead import neldermead
    from optimizador_joshua_8.funciones.objetivo import objetive_function
    func_name = "rosenbrock"
    espaciobusqueda = np.array([[-5, -2.5], [5, 2.5]])  # Puedes ajustar el rango según sea necesario
    obj_func = objetive_function(func_name3)
    #Definir parámetros para Nelder-Mead
    initialpoint = [2, 1.5, 3, -1.5, -2]
    gamma = 1.1
    b = 0.1
    e = 0.5
    # Crear y probar la instancia de Nelder-Mead
    nm = neldermead(initialpoint, gamma=gamma, beta=b, f=obj_func,epsilon=e)
    best = nm.optimize()
    print("Mejor punto encontrado:", best)
Método de newton basado en gradiente 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    from ..baseopt import gradient_methods, optimizador_univariable
    from ..metodos_univariables import goldensearch  # Importa los métodos univariados necesarios
    import numpy as np
    from ..funciones.base import funcion

    class newton(gradient_methods):
    def __init__(self, variables, f: funcion, epsilon, epsilon2, iter=100, opt: optimizador_univariable = goldensearch):
        """
        Inicializa el método de Newton para optimización.

        Parámetros:
        - variables: Lista de variables iniciales.
        - f: Función objetivo a optimizar.
        - epsilon: Tolerancia para la condición de parada basada en el gradiente.
        - epsilon2: Tolerancia adicional para la condición de parada basada en el cambio en las variables.
        - iter: Número máximo de iteraciones (por defecto, 100).
            - opt (optimizador_univariable, opcional): Método de optimización univariable. Por defecto es Goldensearch. Para utilizar se debe escribir:

        goldensearch 
        fibonacci 
        interval 
        biseccion 
        secante 
        """
        super().__init__(variables, f, epsilon, iter)
        self.epsilon2 = epsilon2
        self.opt = opt
        self.gradiente = []
        self.data = f  
    
    def testalpha(self, alfa):
        """
        Evalúa la función objetivo con un valor de alfa dado.

        Parámetros:
        - alfa: Valor de alfa para evaluar.

        Retorna:
        - Resultado de la función objetivo evaluada en variables - (alfa * gradiente).
        """
        return self.funcion(self.variables - (alfa * np.array(self.gradiente)))

    def gradiente_calculation(self, x, delta=0.0001):
        """
        Calcula el gradiente de la función objetivo en el punto dado.

        Parámetros:
        - x: Punto en el cual se calcula el gradiente.
        - delta: Tamaño del paso para la aproximación numérica del gradiente (por defecto, 0.0001).

        Retorna:
        - Vector gradiente calculado.
        """
        if delta is None: 
            delta = 0.00001
        vector_f1_prim = []
        x_work = np.array(x)
        x_work_f = x_work.astype(np.float64)
        if isinstance(delta, int) or isinstance(delta, float):
            for i in range(len(x_work_f)):
                point = np.array(x_work_f, copy=True)
                vector_f1_prim.append(self.primeraderivadaop(point, i, delta))
            return vector_f1_prim
        else:
            for i in range(len(x_work_f)):
                point = np.array(x_work_f, copy=True)
                vector_f1_prim.append(self.primeraderivadaop(point, i, delta[i]))
            return vector_f1_prim

    def primeraderivadaop(self, x, i, delta):
        """
        Calcula la primera derivada parcial de la función objetivo en el punto dado.

        Parámetros:
        - x: Punto en el cual se calcula la derivada.
        - i: Índice de la variable respecto a la cual se calcula la derivada.
        - delta: Tamaño del paso para la aproximación numérica (puede ser un valor único o un arreglo).

        Retorna:
        - Valor de la primera derivada parcial calculada.
        """
        mof = x[i]
        p = np.array(x, copy=True)
        p2 = np.array(x, copy=True)
        nump1 = mof + delta
        nump2 = mof - delta
        p[i] = nump1
        p2[i] = nump2
        numerador = self.funcion(p) - self.funcion(p2)
        return numerador / (2 * delta)

    def segundaderivadaop(self, x, i, delta):
        """
        Calcula la segunda derivada parcial de la función objetivo en el punto dado.

        Parámetros:
        - x: Punto en el cual se calcula la derivada.
        - i: Índice de la variable respecto a la cual se calcula la derivada.
        - delta: Tamaño del paso para la aproximación numérica.

        Retorna:
        - Valor de la segunda derivada parcial calculada.
        """
        mof = x[i]
        p = np.array(x, copy=True)
        p2 = np.array(x, copy=True)
        nump1 = mof + delta
        nump2 = mof - delta
        p[i] = nump1
        p2[i] = nump2
        numerador = self.funcion(p) - 2 * self.funcion(x) + self.funcion(p2)
        return numerador / (delta**2)

    def derivadadodadoop(self, x, index_principal, index_secundario, delta):
        """
        Calcula la derivada doble cruzada de la función objetivo en el punto dado.

        Parámetros:
        - x: Punto en el cual se calcula la derivada.
        - index_principal: Índice de la primera variable respecto a la cual se calcula la derivada.
        - index_secundario: Índice de la segunda variable respecto a la cual se calcula la derivada.
        - delta: Tamaño del paso para la aproximación numérica (puede ser un valor único o un arreglo).

        Retorna:
        - Valor de la derivada doble cruzada calculada.
        """
        mof = x[index_principal]
        mof2 = x[index_secundario]
        p = np.array(x, copy=True)
        p2 = np.array(x, copy=True)
        p3 = np.array(x, copy=True)
        p4 = np.array(x, copy=True)
        if isinstance(delta, int) or isinstance(delta, float):
            mod1 = mof + delta
            mod2 = mof - delta
            mod3 = mof2 + delta
            mod4 = mof2 - delta
            p[index_principal] = mod1
            p[index_secundario] = mod3
            p2[index_principal] = mod1
            p2[index_secundario] = mod4
            p3[index_principal] = mod2
            p3[index_secundario] = mod3
            p4[index_principal] = mod2
            p4[index_secundario] = mod4
            numerador = (self.funcion(p) - self.funcion(p2) - self.funcion(p3) + self.funcion(p4))
            return numerador / (4 * delta * delta)
        else:
            mod1 = mof + delta[index_principal]
            mod2 = mof - delta[index_principal]
            mod3 = mof2 + delta[index_secundario]
            mod4 = mof2 - delta[index_secundario]
            p[index_principal] = mod1
            p[index_secundario] = mod3
            p2[index_principal] = mod1
            p2[index_secundario] = mod4
            p3[index_principal] = mod2
            p3[index_secundario] = mod3
            p4[index_principal] = mod2
            p4[index_secundario] = mod4
            numerador = (self.funcion(p) - self.funcion(p2) - self.funcion(p3) + self.funcion(p4))
            return numerador / (4 * delta[index_principal] * delta[index_secundario])

    def hessian_matrix(self, x, delt=0.0001):
        """
        Calcula la matriz Hessiana de la función objetivo en el punto dado.

        Parámetros:
        - x: Punto en el cual se calcula la Hessiana.
        - delt: Tamaño del paso para la aproximación numérica (por defecto, 0.0001).

        Retorna:
        - Matriz Hessiana calculada.
        """
        matrix_f2_prim = np.zeros((len(x), len(x)))
        x_work = np.array(x)
        x_work_f = x_work.astype(np.float64)
        for i in range(len(x)):
            point = np.array(x_work_f, copy=True)
            for j in range(len(x)):
                if i == j:
                    matrix_f2_prim[i][j] = self.segundaderivadaop(point, i, delt)
                else:
                    matrix_f2_prim[i][j] = self.derivadadodadoop(point, i, j, delt)
        return matrix_f2_prim

    def optimizaralpha(self, test):
        """
        Optimiza el tamaño del paso utilizando un método univariable.Por esta funcion no se puede implementar recibir cualquier funcion, al menos en esta primera version

        Parámetros:
        - test: Función que evalúa el tamaño del paso.

        Retorna:
        - Tamaño del paso optimizado.
        """
        a = self.data.get_limiteinf()
        b = self.data.get_limitesup()
        opt = self.opt(a, b, test, self.epsilon)
        alfa = opt.optimize()
        return alfa

    def optimize(self):
        """
        Funcion principal de la clase

        Retorna:
        - Punto de variables que minimiza la función objetivo.
        """
        terminar = False
        xk = self.variables
        k = 0
        while not terminar:
            gradiente = np.array(self.gradiente_calculation(xk))
            hessiana = self.hessian_matrix(xk)
            invhes = np.linalg.inv(hessiana)
            grad = np.dot(invhes, gradiente)
            if np.linalg.norm(grad) < self.epsilon or k >= self.iteraciones:
                terminar = True
            else:
                def alpha_funcion(alpha):
                    return self.funcion(xk - alpha * grad)
                alpha = self.optimizaralpha(alpha_funcion)
                xk_1 = xk - alpha * np.dot(invhes, gradiente)
                if np.linalg.norm(xk_1 - xk) / (np.linalg.norm(xk) + 0.0001) < self.epsilon2:
                    terminar = True
                xk = xk_1
                k += 1
        print(f"Iteraciones realizadas: {k}")
        return xk

**Ejemplo**
.. code-block::python
    import numpy as np
    from optimizador_joshua_8.metodos_multivariables.newtonmethod import newton
    from optimizador_joshua_8.funciones.objetivo import objetive_function
    func_name = "himmelblau"
    espaciobusqueda = np.array([[0, 0], [1, 1]])
    obj_func = objetive_function(func_name, espaciobusqueda)
    epsilon = 0.01
    inicial = [2, 3]
    n = newton(inicial, obj_func, epsilon=epsilon, epsilon2=epsilon)
    print("Resultado del método Newton:", n.optimize())


Método de caminata aleatoria
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    from ..baseopt import direct_methods
    import numpy as np
    class random_walk(direct_methods):
    def __init__(self, variables, f, epsilon, iter=100):
        """
        Inicializa el método de caminata aleatoria para optimización.

        Parámetros:
        - variables: Lista de variables iniciales.
        - f: Función objetivo a optimizar.
        - epsilon: Tolerancia para la condición de parada.
        - iter: Número máximo de iteraciones (por defecto, 100).
        """
        super().__init__(variables, f, epsilon, iter)

    def step_calculation(self, x):
        """
        Calcula el siguiente paso en la caminata aleatoria.

        Parámetros:
        - x: Punto actual en el espacio de búsqueda.

        Retorna:
        - Nuevo punto generado aleatoriamente cercano a x.
        """
        x_n = np.array(x)
        mu = 0
        stddev = 1
        random_value = np.random.normal(mu, stddev, size=len(x))
        return x_n + random_value

    def optimize(self):
        """
        Implementa el proceso de optimización mediante caminata aleatoria.

        Retorna:
        - Mejor punto encontrado durante la optimización.
        """
        x = np.array(self.variables)
        x_mejor = x
        cont = 0
        while cont < self.iteraciones:
            x_nuevo = self.step_calculation(x)
            if self.funcion(x_nuevo) < self.funcion(x_mejor):
                x_mejor = x_nuevo
            cont += 1
        return x_mejor


**Ejemplo**
.. code-block::python
    import numpy as np
    from optimizador_joshua_8.metodos_multivariables.randomwalk import random_walk
    from optimizador_joshua_8.funciones.objetivo import objetive_function
    func_name = "himmelblau"
    espaciobusqueda = np.array([[0, 0], [1, 1]])
    obj_func = objetive_function(func_name, espaciobusqueda)
    epsilon = 0.01
    inicial = [2, 3]
    rw= random_walk(inicial, obj_func, epsilon, iter=100)
    print("Resultado del método Newton:", rw.optimize())

Implentación de funciones del usuario
----------------------------------
.. code-block:: python
    def funcion_ejemplo(x)
    """
    Para la utilizacion de los métodos multivariables, se requiere un diseño de funcion donde solo reciba X.
    Esto se debe al uso de los np.array brindados por la bibloteca numpy, si el diseño de la funcion no coincide con el moestrado aqui, los metodos marcaran error.

    Parámetros:
    - x: Lista de valores de variables.
    """
    return (x[0] - 2) ** 2 + (x[1] + 3) ** 2 + 5

