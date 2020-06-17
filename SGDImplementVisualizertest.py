import unittest
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
import platform

"""
SGDImplementVisualizertest.py
Created by Koo Hyong Mo

4. 각 구체화 모듈의 단위 테스트 작업
5. 선택한 최적화 알고리즘의 검증

SGDImplementVisualizer.py 의 단위테스트 입니다.

아래 함수/클래스들은 SGDImplementVisualizer.py 파일의 SGDVisualizer 클래스를 복사 하였습니다.
"""


class SGDVisualizer:
    def __init__(self, x_start, x_end, x_size, y_start, y_end, y_size):

        self.x_lin = np.linspace(x_start, x_end, x_size)  # x_start 부터 x_end 까지 x_size 개의 선형 데이터 생성
        self.y_lin = np.linspace(y_start, y_end, y_size)  # y_start 부터 y_end 까지 y_size 개의 선형 데이터 생성

        self.x_mesh, self.y_mesh = np.meshgrid(self.x_lin, self.y_lin)
        self.z = self.rosenbrock(self.x_mesh, self.y_mesh)  # rosenbrock 함수 활용해서 contour 그릴때 사용

        self.levels = np.logspace(-1, 4, 25)  # contour 해상도 (등고선 사이 간격)

        self.h = 1.8e-3  # gradient 값을 예측할때 쓰이는 step

        self.epoch = 25  # 반복 횟수

        self.startX, self.startY = 2, 1.4  # 시작점

    """ (x, y) 에서의 f 값"""

    def rosenbrock(self, x, y):
        return (1 - x) ** 2 + 100.0 * (y - x ** 2) ** 2

    """ (x, y) 에서 f의 gradient 값 (미분) """

    def rosenbrock_d(self, x0, y0):
        grad_x = (self.rosenbrock(x0 + self.h, y0) - self.rosenbrock(x0, y0)) / self.h
        grad_y = (self.rosenbrock(x0, y0 + self.h) - self.rosenbrock(x0, y0)) / self.h
        return [grad_x, grad_y]

    """ 한글 폰트 OS 별로 깨지는 현상 해결하는 함수"""

    def set_fonts(self):
        if platform.system() == 'Windows':
            # 윈도우인 경우
            font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
            rc('font', family=font_name)
        else:
            # Mac 인 경우
            rc('font', family='AppleGothic')

        # 그래프에서 마이너스 기호가 표시되도록 하는 설정입니다.
        plt.rcParams['axes.unicode_minus'] = False

    """ X, Y의 한계값 이나 Title 등을 plot 에 표시하는 함수"""

    def setPlotAttributes(self):
        plt.xlim(-1, 3)
        plt.ylim(-1, 3)
        plt.xticks(np.linspace(-2, 2, 5))
        plt.yticks(np.linspace(-1, 2, 4))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("확률적 경사 하강법을 사용한 2차함수의 최적화 (진동 현상)")

    """ 시작지점 초록색으로 표시하는 함수 """

    def markStartPoint(self):
        plt.text(self.startX + 0.02, self.startY + 0.02, 'start')
        plt.plot(self.startX, self.startY, 'go', markersize=7)

    """ 끝나는 지점 빨간색으로 표시하는 함수 """

    def markEndPoint(self):
        plt.plot(1, 1, 'ro', markersize=7)

    """ 등고선 Plot 에 표시하는 함수 """

    def drawContour(self):
        plt.contourf(self.x_mesh, self.y_mesh, self.z, alpha=0.25, levels=self.levels)
        plt.contour(self.x_mesh, self.y_mesh, self.z, colors="blue", levels=self.levels, zorder=0)

    """ epoch 만큼 반복해서 최적화 하는 함수 """

    def train(self):
        s = 0.95  # 화살표 크기
        x = self.startX
        y = self.startY
        for i in range(self.epoch):
            g = self.rosenbrock_d(x, y)
            plt.arrow(x, y, -s * self.h * g[0], -s * self.h * g[1],
                      head_width=0.04, head_length=0.04, fc='k', ec='k', lw=1)
            x = x - self.h * g[0]
            y = y - self.h * g[1]

    """ 시각화 해주는 함수 class 외부에서 이 함수만 호출 하면 될 수 있도록 함 """

    def visualize(self):
        self.set_fonts()
        self.drawContour()
        self.markStartPoint()
        self.markEndPoint()
        self.train()
        self.setPlotAttributes()

    def show(self):
        plt.show()


class MyTestCase(unittest.TestCase):

    """
    validating rosenbrock function
    """
    def test_rosenbrock(self):
        # need for instantiation
        # x 값의 범위
        x_range_start = -2
        x_range_end = 3
        x_range_size = 800

        # y 값의 범위
        y_range_start = -1
        y_range_end = 3
        y_range_size = 600

        optimizer = SGDVisualizer(x_range_start,
                                  x_range_end,
                                  x_range_size,
                                  y_range_start,
                                  y_range_end,
                                  y_range_size)

        x = -2
        y = 2
        result = 409.0
        self.assertEqual(optimizer.rosenbrock(x, y), result)

    """ 
    validating derivative of rosenbrock function 
    """
    def test_rosenbrock_derivative(self):
        # need for instantiation
        # x 값의 범위
        x_range_start = -2
        x_range_end = 3
        x_range_size = 800

        # y 값의 범위
        y_range_start = -1
        y_range_end = 3
        y_range_size = 600

        optimizer = SGDVisualizer(x_range_start,
                                  x_range_end,
                                  x_range_size,
                                  y_range_start,
                                  y_range_end,
                                  y_range_size)

        x = -2
        y = 2
        result = [-1602.4007914168692, -399.81999999996435]
        self.assertEqual(optimizer.rosenbrock_d(x, y), result)

    """ 
    1. 한글 폰트가 적용이 잘되는지
    2. Plot attributes(title, x, y)
    등등이 잘 그려지는지 테스트 합니다.
    """

    def test_run_korean_font_and_plot_attributes(self):
        # need for instantiation
        # x 값의 범위
        x_range_start = -2
        x_range_end = 3
        x_range_size = 800

        # y 값의 범위
        y_range_start = -1
        y_range_end = 3
        y_range_size = 600

        optimizer = SGDVisualizer(x_range_start,
                                  x_range_end,
                                  x_range_size,
                                  y_range_start,
                                  y_range_end,
                                  y_range_size)

        optimizer.set_fonts()
        optimizer.setPlotAttributes()
        optimizer.show()

    """ 
    등고선이 잘 그려지는 지 테스트 합니다.
    """

    def test_draw_contour(self):
        # need for instantiation
        # x 값의 범위
        x_range_start = -2
        x_range_end = 3
        x_range_size = 800

        # y 값의 범위
        y_range_start = -1
        y_range_end = 3
        y_range_size = 600

        optimizer = SGDVisualizer(x_range_start,
                                  x_range_end,
                                  x_range_size,
                                  y_range_start,
                                  y_range_end,
                                  y_range_size)

        optimizer.drawContour()
        optimizer.show()

    """ 
    시작점과 끝나는점이 각각 잘 표시 되는지 테스트 합니다 
    """
    def test_legends(self):
        # need for instantiation
        # x 값의 범위
        x_range_start = -2
        x_range_end = 3
        x_range_size = 800

        # y 값의 범위
        y_range_start = -1
        y_range_end = 3
        y_range_size = 600

        optimizer = SGDVisualizer(x_range_start,
                                  x_range_end,
                                  x_range_size,
                                  y_range_start,
                                  y_range_end,
                                  y_range_size)

        optimizer.markStartPoint()
        optimizer.markEndPoint()

        optimizer.show()

    """ 
    SGD 알고리즘이 잘 작동하는지 테스트 합니다.
    """
    def test_train_and_visualize(self):
        # need for instantiation
        # x 값의 범위
        x_range_start = -2
        x_range_end = 3
        x_range_size = 800

        # y 값의 범위
        y_range_start = -1
        y_range_end = 3
        y_range_size = 600

        optimizer = SGDVisualizer(x_range_start,
                                  x_range_end,
                                  x_range_size,
                                  y_range_start,
                                  y_range_end,
                                  y_range_size)

        optimizer.visualize()
        optimizer.show()



if __name__ == '__main__':
    unittest.main()
