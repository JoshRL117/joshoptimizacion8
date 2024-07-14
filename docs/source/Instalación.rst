Instalacion del paquete optimizador-joshua-3
=====

.. _Instalación:

Para para instalar este paquete se hace ueso de `pip`, el cual es el administrador de paquetes de Python. Los pasos para realizar de forma correcta la instalación

son los siguientes. Primero, asegurar que el pip este instalado;en el caso que no lo tengas, utilice la 

documentacion de `pip` para realizar el proceso corresondiente al sistema operativo en uso[pip](https://pip.pypa.io/en/stable/installation/).

Seguidamente, se debe poner el siguiente comando en tu terminal:

.. code-block:: bash

   pip install optimizador-joshua-3
**Requerimientos*

Antes de descargar asegurate de tener la siguiente dependencias
- Python 3.6 o superior
- pip
- numpy
Para verificar la versión de Python y pip instaladas en tu sistema, puedes usar los siguientes comandos:
Para verificar que vercion de Python y Pip tienes intaladas puedes ejecutar los siguientes comandos en Powershell o cmd:
.. code-block:: bash

    python --version
    pip --version

Para comprobar que numpy se encuentre instalado utilice el siguiente comando: 
.. code-block:: bash

    pip show numpy

Este comando mostrará información sobre el paquete numpy si está instalado, como la versión, la ubicación de la instalación, y más.
Si encuentras problemas durante la instalación, aquí tienes algunos pasos que puedes seguir para resolverlos:

1. **Actualizar pip**: Asegúrate de que `pip` esté actualizado a la última versión.
   
   .. code-block:: bash

       pip install --upgrade pip

2. **Dependencias Faltantes**: Si faltan dependencias, `pip` debería manejarlas automáticamente. Si no, instala las dependencias manualmente listadas en `requirements.txt` o según la documentación del paquete.

Si despues de seguir los paso aun tienes algunos problemas , por favor comonicate con nosotros atraves del repositorio del proyecto en GitHub.
