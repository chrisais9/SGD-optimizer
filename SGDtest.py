import unittest
from scipy.optimize import minimize
import numpy as np
import numpy.testing as nptest

"""
SGDtest.py
Created by Koo Hyong Mo

2.선택한 최적화 알고리즘의 동작 코드와 단위 테스트

SGD.py 의 단위테스트 입니다.

아래 함수들은 SGD 파일의 함수를 그대로 복사하였습니다.
"""


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


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


class MyTestCase(unittest.TestCase):
    def test_rosenbrock(self):
        # validating rosenbrock function
        x = (-2, 2)
        y = 409.0
        self.assertEqual(rosenbrock(x), y)

    def test_rosenbrock_der(self):
        # validating derivative of rosenbrock function
        x = np.array([-2, 2])
        y = np.array([-1606, -400])
        nptest.assert_array_equal(rosenbrock_der(x), y)

    def test_run_sgd(self):
        # validating SGD optimize algorithm with rosenbrock function
        start_point = np.array([2, 1.4])
        run(start_point)


if __name__ == '__main__':
    unittest.main()
