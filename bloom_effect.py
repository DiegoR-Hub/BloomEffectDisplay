import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import sys
from scipy import sparse
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

"""Funcion que recibe la llamada en terminal"""
def recibirLlamada():
    numpy_imagen = pngToArrayAux(sys.argv[1])
    N = int(sys.argv[2])
    color = np.array([int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])])
    total_rows = numpy_imagen.shape[0]
    total_cols = numpy_imagen.shape[1]
    return  numpy_imagen, N, color, total_rows, total_cols

"""Funcion para encontrar centros de clusters"""
def encontrarCentrosDeLuz(numpy_imagen, color):
    lista_centros = []
    total_rows = numpy_imagen.shape[0]
    total_cols = numpy_imagen.shape[1]
    for fila in range(total_rows):
        for columna in range(total_cols):
            if numpy_imagen[fila][columna][0] == color[0] and numpy_imagen[fila][columna][1] == color[1] and numpy_imagen[fila][columna][2] == color[2]:
                lista_centros.append((fila,columna))
    return lista_centros


"""Funcion que dado un png genera un numpy array que lo representa"""
def pngToArrayAux(imagen):
    img = Image.open(imagen)
    numpy_imagen = asarray(img)
    return numpy_imagen

"""Funcion que entrega un diccionario con todos los pixeles afectados por la iluminacion, indexados"""
def biyeccionAlDominio(total_rows, total_cols, lista_centros, N):
    contador = 0
    #diccionario donde guardar las posiciones de los pixeles asociados a una indexacion
    dicLlaveIndex = {}
    dicLlavePos = {}
    #A cada cluster
    for centro in lista_centros:
        centro_fila = centro[0]
        centro_columna = centro[1]
        # recorre un cuadrado de NxN centrado en el centro del cluster
        limite_izquierdo = centro_columna - N
        limite_derecho = centro_columna + N + 1
        limite_inferior = centro_fila - N
        limite_superior = centro_fila + N + 1
        for fila in range(limite_inferior, limite_superior):
            for columna in range(limite_izquierdo, limite_derecho):
                #si un pixel del cuadrado se encuentra dentro de la imagen
                if 0<fila<total_rows and 0<columna<total_cols:
                    #y ademas tiene norma 1 menor a N
                    if abs(centro_fila - fila)+abs(centro_columna - columna) <= N:
                        #y si ademas no ha sido indexado
                        if (fila, columna) not in dicLlaveIndex.values():
                            #y ademas no es el centro del cluster o sea o al menos la fila o la columna son distintas
                            if fila!=centro_fila or columna!=centro_columna:
                                #lo agrega al diccionario con su correspondiente indice
                                dicLlaveIndex[contador] = (fila,columna)
                                dicLlavePos[(fila, columna)] = contador
                                #aumenta el indice para el siguiente pixel a indexar
                                contador += 1
    return dicLlaveIndex, dicLlavePos

"""Funcion que genera la matriz A y el vector b de metodo de gauss para solucion de EDP"""
def generarAb(dicLlaveIndex, dicLlavePos, lista_centros, color):
    dimension = len(dicLlaveIndex)
    A = lil_matrix((dimension, dimension))
    #llena de -4 en la diagonal
    for i in range(dimension):
        A[i,i] = -4
    #b tiene numero de filas elementos y cada elemento es un numpy array de 3 valores (RGB)
    b = np.zeros((dimension,3))
    #Por cada indice (cada pixel en la discretizacion) lo hace centro del stencil
    for index in dicLlavePos.values():
        #Posicion asociada al index
        pos_center_actual = dicLlaveIndex.get(index)

        """Chequea pixel izquierdo, en dominio no borde, en dominio borde, fuera del dominio"""
        posx_izquierdo = pos_center_actual[0] - 1
        posy_izquierdo = pos_center_actual[1]
        #caso dominio borde, agrega el color en b
        if (posx_izquierdo,posy_izquierdo) in lista_centros:
            b[index] = color
        #caso dominio no borde, marca en la fila de index, un 1 en el indice del pixel
        elif (posx_izquierdo, posy_izquierdo) in dicLlaveIndex.values():
            A[index, dicLlavePos.get((posx_izquierdo, posy_izquierdo))] = 1
            #caso fuera del dominio, no hace nada

        """Chequea pixel derecho, en dominio no borde, en dominio borde, fuera del dominio"""
        posx_derecha = pos_center_actual[0] + 1
        posy_derecha = pos_center_actual[1]
        # caso dominio borde, agrega el color en b
        if (posx_derecha, posy_derecha) in lista_centros:
            b[index] = color
            # caso dominio no borde, marca en la fila de index, un 1 en el indice del pixel
        elif (posx_derecha, posy_derecha) in dicLlaveIndex.values():
            A[index, dicLlavePos.get((posx_derecha, posy_derecha))] = 1
            # caso fuera del dominio, no hace nada

        """Chequea pixel arriba, en dominio no borde, en dominio borde, fuera del dominio"""
        posx_arriba = pos_center_actual[0]
        posy_arriba = pos_center_actual[1] + 1
        # caso dominio borde, agrega el color en b
        if (posx_arriba, posy_arriba) in lista_centros:
            b[index] = color
            # caso dominio no borde, marca en la fila de index, un 1 en el indice del pixel
        elif (posx_arriba, posy_arriba) in dicLlaveIndex.values():
            A[index, dicLlavePos.get((posx_arriba, posy_arriba))] = 1
            # caso fuera del dominio, no hace nada

        """Chequea pixel abajo, en dominio no borde, en dominio borde, fuera del dominio"""
        posx_abajo = pos_center_actual[0]
        posy_abajo = pos_center_actual[1] - 1
        # caso dominio borde, agrega el color en b
        if (posx_abajo, posy_abajo) in lista_centros:
            b[index] = color
            # caso dominio no borde, marca en la fila de index, un 1 en el indice del pixel
        elif (posx_abajo, posy_abajo) in dicLlaveIndex.values():
            A[index, dicLlavePos.get((posx_abajo, posy_abajo))] = 1
            # caso fuera del dominio, no hace nada
    return A, b

"""Funcion que dada una A y una b genera la solucion al sistema matricial"""
def solucionEcuacionMatricial(A, b):
    #conversion de A y b a matrices sparse para ahorrar memoria
    A_sparse = sparse.csc_matrix(A)
    b_sparse = sparse.csc_matrix(b)
    #genera el vector solucion, un csc_matrix
    vector_solucion = spsolve(A_sparse, b_sparse)
    #conviernte csc_matrix a numpy matrix
    vector_solucion = vector_solucion.todense()
    #convierte numpy matrix a numpy array
    vector_solucion = np.asarray(vector_solucion)
    #trunca valores float a int
    vector_solucion = vector_solucion.astype(int)
    #retorna - el resultado
    vector_solucion = -vector_solucion
    return vector_solucion

"""Funcion que genera la imagen tras aplicado efecto Bloom"""
def modificarImagenDadaSolucion(imagen, vector_solucion, dicLlaveIndex):
    for index in range(len(dicLlaveIndex)):
        posicion = dicLlaveIndex.get(index)
        posfila = posicion[0]
        poscol = posicion[1]
        #suma la variacion de color a cada canal de color, RGB
        #suma a canal rojo
        if imagen[posfila][poscol][0] + vector_solucion[index][0] > 255:
            imagen[posfila][poscol][0] = 255
        else:
            imagen[posfila][poscol][0] += vector_solucion[index][0]
        #suma a canal verde
        if imagen[posfila][poscol][1] + vector_solucion[index][1] > 255:
            imagen[posfila][poscol][1] = 255
        else:
            imagen[posfila][poscol][1] += vector_solucion[index][1]
        #suma a canal azul
        if imagen[posfila][poscol][2] + vector_solucion[index][2] > 255:
            imagen[posfila][poscol][2] = 255
        else:
            imagen[posfila][poscol][2] += vector_solucion[index][2]
    imagenModificada = imagen

    return imagenModificada

"""Funcion main"""
def funcionPrincipal():
    imagen_aux, N, color, total_rows, total_cols = recibirLlamada()
    print("Valor de N: " + str(N))
    print("Valor de color: ", end="")
    print(color)
    # imagen_aux era read only, numpy_imagen es write y read
    numpy_imagen = np.copy(imagen_aux)
    #para el ejemplo pone un punto de luz en el centro
    numpy_imagen[100][200] = np.array([255, 255, 254])
    #busca los puntos de la imagen que tienen el color dado
    lista_centros = encontrarCentrosDeLuz(numpy_imagen, color)
    print("Lista de centros: ", end="")
    print(lista_centros)
    #genera dos diccionarios para la biyeccion entre posiciones del dominio y numeros naturales
    dicLlaveIndex, dicLlavePos = biyeccionAlDominio(total_rows, total_cols, lista_centros, N)
    #genera A y b de la ecuacion matricial a resolver
    A, b = generarAb(dicLlaveIndex, dicLlavePos, lista_centros, color)
    #genera el vector solucion de la ecuacion matricial Ax = b
    vector_solucion = solucionEcuacionMatricial(A, b)
    #genera la nueva imagen tras realizado el efecto bloom
    nueva_imagen = modificarImagenDadaSolucion(numpy_imagen, vector_solucion, dicLlaveIndex)
    #muestra la imagen tras aplicar el efecto bloom
    plt.title("Imagen despues de aplicar efecto Bloom")
    plt.imshow(nueva_imagen)
    plt.show()
    #muestra la imagen antes de aplicar el efecto bloom
    plt.title("Imagen original")
    plt.imshow(imagen_aux)
    plt.show()

funcionPrincipal()