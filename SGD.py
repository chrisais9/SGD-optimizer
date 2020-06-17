from scipy.optimize import minimize
import numpy as np
"""
SGD.py
Created by Koo Hyong Mo

2. 선택한 최적화 알고리즘의 동작 코드와 단위 테스트

Stochastic Gradient Descent 알고리즘을 scipy optimize를 이용해서
구현하였습니다.

단위테스트는 SGDtest.py 를 참고해 주세요.
"""

""" 로젠 브룩 함수 """
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


""" 로젠 브룩 함수 미분"""
def rosenbrock_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der

def run(start_point):

    result = minimize(rosenbrock, start_point, jac=rosenbrock_der)
    print(result)

if __name__ == '__main__':
    start_point = np.array([2, 1.4])
    run(start_point)