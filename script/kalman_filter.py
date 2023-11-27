import numpy as np


# 卡尔曼滤波器
class KalmanFilter1D:
    def __init__(
        self, process_variance, measurement_variance, est_error, initial_value=0
    ):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimate_error = est_error

    def update(self, measurement):
        # 预测
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance

        # 更新
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error

        return self.estimate


def use_kalman_filter(RAW_data):
    process_variance = 0.0001
    measurement_variance = 1000  # 测量误差
    est_error = 2
    initial_value = RAW_data[0]
    kf = KalmanFilter1D(
        process_variance, measurement_variance, est_error, initial_value
    )
    filtered_values = []
    for measurement in RAW_data:
        filtered_value = kf.update(measurement)
        filtered_values.append(filtered_value)
    return np.array(filtered_values)
