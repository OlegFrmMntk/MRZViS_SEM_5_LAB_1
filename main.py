import math
import random
import numpy as np
import matplotlib.pylab as plt

from PIL import Image


class ImageCompressor:
    """
       Данный класс  реализует модель линейной рециркуляционной сети
       с адаптивным шагом обучения с нормированными весами.
    """

    # Параметры изображения
    S = 3
    c_max = 255
    n = 8
    m = 8
    N = n * m * S

    def __init__(self, img, max_e=500, p=48):
        # Загрузка изображения
        self.image = Image.open(img)
        self.pix = self.image.load()

        # Размеры изображения
        self.H = self.image.height
        self.W = self.image.width

        # Количество блоков изображения
        self.L = int(self.H / self.n * self.W / self.m)

        # Максимальная ошибка
        self.e = max_e

        # Число нейронов на втором слое
        self.p = p

        # Коэффициент сжатия
        self.Z = (self.N * self.L) / ((self.N + self.L) * self.p + 2)

        # Количество итераций
        self.iteration_counter = 0

    # Вывод информации о параметрах сети
    def __str__(self):
        print("Число нейронов второго слоя = {}".format(self.p))
        print("Коэффициент сжатия = {}".format(self.Z))

    # Подсчет адаптивного шага обучения
    def adaptive_learning_step(self, matrix):
        tmp = np.dot(matrix, np.transpose(matrix))
        s = 0
        for i_f in range(len(tmp)):
            s += tmp[i_f]

        return 1.0 / (s * 10)

    # Нормирование матрицы
    def get_normalized_matrix(self, matrix):
        for i_f in range(len(matrix[0])):
            s = 0
            for j_f in range(len(matrix)):
                s += matrix[j_f][i_f] * matrix[j_f][i_f]
            s = math.sqrt(s)
            for j_f in range(len(matrix)):
                matrix[j_f][i_f] = matrix[j_f][i_f] / s

    # Представление одного блока
    def get_implementation_of_square(self, weight, height):
        Xqhw = np.empty(self.N)
        for j in range(self.n):
            for k in range(self.m):
                for i in range(self.S):  # S - rgb
                    Xqhw[i + self.S * (j + k * self.n)] = 2 * self.pix[j + height, k + weight][i] / self.c_max - 1

        return Xqhw

    def divide_on_squares(self):
        Xq = []
        for height in range(0, self.H, self.n):
            for weight in range(0, self.W, self.m):
                Xq.append(self.get_implementation_of_square(weight, height))

        return Xq

    # Создание матрицы для изображения
    def create_image_matrix(self, Xq, X_out):
        # Инициализация матриц
        image_restored = np.empty((self.H, self.W, self.S))
        image_origin = np.empty((self.H, self.W, self.S))

        # Удаление ненужных осей из массивов
        Xq = np.squeeze(Xq, axis=1)
        X_out = np.squeeze(X_out, axis=1)

        # Создание матрицы изображения из X_out
        le = self.H / self.n
        for h in range(0, self.H, self.n):
            for w in range(0, self.W, self.m):
                xq = Xq[int((h / self.n) * le + (w / self.m))]
                x_out = X_out[int((h / self.n) * le + (w / self.m))]
                for j in range(self.n):
                    for k in range(self.m):
                        for i in range(self.S):
                            image_restored[j + h, k + w, i] = (x_out[
                                                                   i + self.S * (j + k * self.n)] + 1) * self.c_max / 2
                            image_origin[j + h, k + w, i] = (xq[i + self.S * (j + k * self.n)] + 1) * self.c_max / 2

        return image_restored, image_origin

    # Главная функция
    def image_compressor(self):
        Xq = self.divide_on_squares()

        # initialize matrices
        W_first = np.empty((self.N, self.p))
        W_second = np.empty((self.p, self.N))
        X_out = np.empty((self.L, self.N))
        X_delta = np.empty((self.L, self.N))

        # adding axis to array (for easier matrix transposing)
        Xq = np.expand_dims(Xq, axis=1)
        X_out = np.expand_dims(X_out, axis=1)
        X_delta = np.expand_dims(X_delta, axis=1)

        # fielding weight matrices with random values
        random.seed()
        for i in range(self.N):
            W_first[i] = np.random.uniform(-1, 1, self.p)

        W_second = np.transpose(W_first)

        E = self.e + 1
        while E > self.e:
            E = 0

            for k in range(self.L):
                # counting Y
                Y = np.dot(Xq[k], W_first)

                # Counting X'
                X_out[k] = np.dot(Y, W_second)

                # Counting delta X
                X_delta[k] = X_out[k] - Xq[k]

                # Counting W'
                alpha_second = self.adaptive_learning_step(Y)
                # print(alpha_second)
                W_second = W_second - alpha_second * np.dot(np.transpose(Y), X_delta[k])

                # Counting W
                alpha_first = self.adaptive_learning_step(Xq[k])
                # print(alpha_first)
                W_first = W_first - alpha_first * np.dot(np.dot(np.transpose(Xq[k]), X_delta[k]),
                                                         np.transpose(W_second))

                # normalizing matrices
                self.get_normalized_matrix(W_first)
                self.get_normalized_matrix(W_second)

                # Подсчет текущей ошибки
                for i in range(self.N):
                    E += X_delta[k][0][i] * X_delta[k][0][i]

            self.iteration_counter += 1

            # Вывод информации о текущей итерации
            print("Iteration number: {}, error {}".format(self.iteration_counter, E))

        # Вывод инфорации о количество итераций и текущей ошибке
        print("Final iteration count: {}, final error {}".format(self.iteration_counter, E))

        # Сжатие и восстановление изображения
        for k in range(self.L):
            Y = np.dot(Xq[k], W_first)
            X_out[k] = np.dot(Y, W_second)

        # Вычисление матриц изображения
        image_restored, image_origin = self.create_image_matrix(Xq, X_out)

        fig = plt.figure()

        # Показать исходное изображение
        fig.add_subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image_origin.astype(np.int32))

        # Показать восстановленое изображение
        fig.add_subplot(1, 2, 2)
        plt.title("Reconstructed Image")
        plt.imshow(image_restored.astype(np.int32))
        plt.show()


imageComp = ImageCompressor("image/test5.jpg")
imageComp.image_compressor()
