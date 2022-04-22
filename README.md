# TrafficDensityApp 

TrafficDensityApp es una aplicación diseñada con el propósito de ayudar a los ayuntamientos a que tengan un mayor control de la información de densidad de tráfico en sus respectivos municipios, a partir de unas imágenes aéreas de ciertas zonas de control. 

TrafficDensityApp hace uso de visión artificial para la detección de vehículos a partir de una imagen aérea. Esta aplicación, devuelve las posiciones de los vehículos, pudiendo estimar la densidad de vehículos en un área. 

<p align="center"> 

<img src="./imgs/austin1_cropped.jpg"> <img src="./imgs/imgs_results/austin1_cropped.jpg"> 

</p>

# Instalación en host

Para poder ejecutar nuestro software el usuario necesitará como mínimo las siguientes características: 

* Python3.6 o mayor 
* (opcional) CUDA 11.2 
* (opcional) Driver de CUDA (cuDNN): 460.27.04 o mayor 

El software se puede ejecutar en GPU instalando previamente las dependencias necesarias.

Los pasos necesarios para instalar la aplicación: 

**1.** Clonar el repositorio
~~~
git clone https://github.com/fkite-cs/TrafficDensityApp.git
~~~

**2.** Crear entorno virtual
~~~
python3 -m venv virtual_environment
source virtual_environment/bin/activate
~~~

**3.** Instalar librerías dentro del entorno
~~~
python -m pip install -r requirements.txt
~~~

Al terminar estos pasos, el usruario será capaz de ejecutar la demo de detección que se ha desarrollado. 

# Ejecución en host
Una imagen de prueba se descargar desde este [enlace](https://drive.google.com/drive/folders/1JGlKaW8ph1TYesDpVoNz4p6J_-94aBEd?usp=sharing).

Para lanzar la aplicación es necesario llamar a la función main con dos argumentos
* `img_path`: ruta de la imagen que se va a analizar/estudiar.
* `out_folder`: carpeta donde se guardan las detecciones.
~~~
python main.py --img_path [PATH/TO/IMG] --out_folder [OUTPUT/FOLDER]
~~~

Para ejecutar los test, es suficiente con moverse a la carpeta de test/ y ejecutar los diferentes *.py disponibles en ella.

~~~
cd test/
python [TESTFILE_NAME]
~~~

# Instalación de imagen de Docker

Se ofrece una imagen de Docker ya operativa en DockerHub. Esta imagen se puede descargar desde el siguiente [enlace](https://hub.docker.com/r/maevision/maevision_tda)

Para hacer uso de esta tecnología es necesario tener instalado Docker en el dispositivo donde se va ejecutar la aplicación. Los pasos para su instalación se pueden encontrar en la [página oficial](https://docs.docker.com/engine/install/ubuntu/). Para ejecutar docker sin usar `sudo` se sigue el siguiente [tutorial](https://docs.docker.com/engine/install/linux-postinstall/)

Con `docker pull maevision/maevision_tda` se descarga la imagen de la nube. Este proceso puede llevar unos minutos.

Se puede comprobar si la imagen se ha descargado en su sistema si se ejecuta el comando `docker images`

Otra posibilidad es crear la imagen de Docker usando el Dockerfile de este repositorio. Para ello se deben seguir los siguientes comandos:

~~~
cd docker
docker build -t maevision/maevision_tda .
~~~

# Ejecución de aplicación usando docker

El script `run.sh` lanza la aplicación. Los parámetros que recibe son:

* `path_folder`: es una carpeta compartida entre el host y el contendor.
* `img_name`: nombre de la imagen que se va a procesar. Esta imagen debe estar en la carpeta del argumento anterior.

El resultado se guardará en `path_folder`.

# Trabajos futuros 

En las siguientes fases del proyecto se diseñarán e implementarán varias mejoras como: 

* Reducir el tiempo del mapa de calor.
* Detección de calles. Posibilidad de seleccionar diferentes calles, autopistas o cualquier vía pública para calcular su densidad de tráfico. 
* ✨✨✨ Interfaz gráfica. Elegante y fácil de usar. ✨✨✨
