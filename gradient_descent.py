import numpy as np

def gradient_descent(x, y, theta, learning_rate, iterations):
    """
    Выполняет градиентный спуск для нахождения коэффициентов линейной регрессии.
    
    :param x: Массив признаков (numpy array)
    :param y: Вектор истинных значений (numpy array)
    :param theta: Вектор параметров (numpy array)
    :param learning_rate: Скорость обучения (float)
    :param iterations: Количество итераций (int)
    :return: Оптимизированные параметры theta и история значений функции потерь
    """
    m = len(y)  # Количество наблюдений
    cost_history = []

    for i in range(iterations):
        # Предсказание
        predictions = np.dot(x, theta)
        
        # Градиент функции потерь
        error = predictions - y
        gradients = (1 / m) * np.dot(x.T, error)
        
        # Обновление параметров
        theta -= learning_rate * gradients
        
        # Функция потерь (MSE)
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        cost_history.append(cost)

        if i % 100 == 0:  # Логирование каждые 100 шагов
            print(f"Итерация {i}: Потеря = {cost}")

    return theta, cost_history


if __name__ == "__main__":

    
    # Пример данных
    x = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])  # Признаки (добавлен bias)
    y = np.array([2, 2.5, 3.5, 4])  # Целевая переменная
    theta = np.zeros(x.shape[1])  # Начальные параметры
    learning_rate = 0.1
    iterations = 1000


    # Запуск градиентного спуска
    optimal_theta, loss_history = gradient_descent(x, y, theta, learning_rate, iterations)
    print(f"Оптимизированные параметры theta: {optimal_theta}")









