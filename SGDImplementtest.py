import unittest
import numpy as np

"""
SGDImplementtest.py
Created by Koo Hyong Mo

3. 선택한 최적화 알고리즘의 구체화
4. 각 구체화 모듈의 단위 테스트 작업
5. 선택한 최적화 알고리즘의 검증

로젠브룩 함수를 이용해서 검증을 진행하며 numpy 레벨에서 구현되었습니다.

아래 함수/클래스들은 SGDImplement.py 파일의 SGDImplement 클래스를 복사 하였습니다.

"""


class SGDImplement:
    def __init__(self, x_start, x_end, x_size, y_start, y_end, y_size):
        self.x_lin = np.linspace(x_start, x_end, x_size)  # x_start 부터 x_end 까지 x_size 개의 선형 데이터 생성
        self.y_lin = np.linspace(y_start, y_end, y_size)  # y_start 부터 y_end 까지 y_size 개의 선형 데이터 생성

        self.x_mesh, self.y_mesh = np.meshgrid(self.x_lin, self.y_lin)

        self.h = 1.8e-3  # gradient 값을 예측할때 쓰이는 step

        self.epoch = 25  # 반복 횟수

    """ (x, y) 에서의 f 값"""

    def rosenbrock(self, x, y):
        return (1 - x) ** 2 + 100.0 * (y - x ** 2) ** 2

    """ (x, y) 에서 f의 미분값 """

    def rosenbrock_d(self, x0, y0):
        grad_x = (self.rosenbrock(x0 + self.h, y0) - self.rosenbrock(x0, y0)) / self.h
        grad_y = (self.rosenbrock(x0, y0 + self.h) - self.rosenbrock(x0, y0)) / self.h
        return [grad_x, grad_y]

    """ epoch 만큼 반복해서 최적화 하는 함수 """

    def train(self, start_x, start_y):
        train_history = []

        # 시작 좌표
        x = start_x
        y = start_y
        train_history.append([x, y])

        for i in range(self.epoch):
            g = self.rosenbrock_d(x, y)
            x = x - self.h * g[0]
            y = y - self.h * g[1]
            train_history.append([x, y])

        return train_history


class MyTestCase(unittest.TestCase):
    def test_rosenbrock(self):
        x_range_start = -2
        x_range_end = 3
        x_range_size = 800

        # y 값의 범위
        y_range_start = -1
        y_range_end = 3
        y_range_size = 600

        optimizer = SGDImplement(x_range_start,
                                 x_range_end,
                                 x_range_size,
                                 y_range_start,
                                 y_range_end,
                                 y_range_size)

        x = -2
        y = 2
        result = 409.0

        self.assertEqual(optimizer.rosenbrock(x, y), result)

    def test_rosenbrock_derivative(self):

        x_range_start = -2
        x_range_end = 3
        x_range_size = 800

        # y 값의 범위
        y_range_start = -1
        y_range_end = 3
        y_range_size = 600

        optimizer = SGDImplement(x_range_start,
                                 x_range_end,
                                 x_range_size,
                                 y_range_start,
                                 y_range_end,
                                 y_range_size)

        x = -2
        y = 2
        result = [-1602.4007914168692, -399.81999999996435]
        self.assertEqual(optimizer.rosenbrock_d(x, y), result)

    def test_train(self):

        x_range_start = -2
        x_range_end = 3
        x_range_size = 800

        # y 값의 범위
        y_range_start = -1
        y_range_end = 3
        y_range_size = 600

        optimizer = SGDImplement(x_range_start,
                                 x_range_end,
                                 x_range_size,
                                 y_range_start,
                                 y_range_end,
                                 y_range_size)

        start_x = 2
        start_y = 1.4

        train_data = optimizer.train(start_x, start_y)

        print(train_data)
        # 마지막 데이터가 (1,1) 에 가까울수록 최적화가 잘되었다고 볼 수 있음.
        print("optimized data :", train_data[-1])




if __name__ == '__main__':
    unittest.main()
