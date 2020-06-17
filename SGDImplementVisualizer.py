import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
import platform
"""
SGDImplementVisualizer.py
Created by Koo Hyong Mo

3. 선택한 최적화 알고리즘의 구체화
5. 선택한 최적화 알고리즘의 검증

SGDImplement.py 내의 class SGDImplement를 개량 하여 visualizer 기능을 추가하였습니다.
SGD 알고리즘 구체화와 검증 그리고 시각화 까지 동시에 진행합니다.

SGDImplementVisualizertest.py 에서 단위 테스트를 진행할 수 있습니다.

"""

class SGDVisualizer:
    def __init__(self, x_start, x_end, x_size, y_start, y_end, y_size):

        self.x_lin = np.linspace(x_start, x_end, x_size)  # x_start 부터 x_end 까지 x_size 개의 선형 데이터 생성
        self.y_lin = np.linspace(y_start, y_end, y_size)  # y_start 부터 y_end 까지 y_size 개의 선형 데이터 생성

        self.x_mesh, self.y_mesh = np.meshgrid(self.x_lin, self.y_lin)
        self.z = self.rosenbrock(self.x_mesh, self.y_mesh) # rosenbrock 함수 활용해서 contour 그릴때 사용

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
        plt.show()


if __name__ == '__main__':
    optimizer = SGDVisualizer(-2, 3, 800, -1, 3, 600)
    optimizer.visualize()
