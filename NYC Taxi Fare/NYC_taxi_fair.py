import numpy as np
import pandas as pd
import csv
import matplotlib
from datetime import datetime
from math import radians, cos, sin, asin, sqrt


def distance(lat1, lat2, lon1, lon2):
    lon1 = radians(float(lon1))
    lon2 = radians(float(lon2))
    lat1 = radians(float(lat1))
    lat2 = radians(float(lat2))

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return (c * r)


def gradL_mse(phi, w, y):
    a = np.dot(phi.T, phi)
    a = pd.DataFrame(a)
    b = np.dot(phi.T, y)
    b = pd.DataFrame(b)
    c = np.dot(a, w)
    c = pd.DataFrame(c)
    return 2 * (c - b)


def normliz(phi, y):
    n = phi.shape[1]
    for i in range(1, n):
        phi.iloc[:, i] = (phi.iloc[:, i] - np.mean(phi.iloc[:, i])) / np.std(phi.iloc[:, i])
        # phi[i] = (phi[i]-min(phi[i]))/(max(phi[i])-min(phi[i]))
    phi = pd.DataFrame(phi)
    #y = (y - np.mean(y)) / np.std(phi[i])
    # y = (y - min(y)) / (max(y)-min(y))
    y = pd.DataFrame(y)
    return phi, y


def get_features(train):
    n = train.shape[0]
    dt = []
    for i in range(0, n):
        dt.append(datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S").timestamp())
    pick_long = []
    for i in range(0, n):
        pick_long.append(train.iloc[i, 1])
    pick_lat = []
    for i in range(0, n):
        pick_lat.append(train.iloc[i, 2])
    drop_long = []
    for i in range(0, n):
        drop_long.append(train.iloc[i, 3])
    drop_lat = []
    for i in range(0, n):
        drop_lat.append(train.iloc[i, 4])
    pass_count = []
    for i in range(0, n):
        pass_count.append(train.iloc[i, 5])
    const = np.ones(n)
    phi = [const, dt, pick_long, pick_lat, drop_long, drop_lat, pass_count]
    phi = pd.DataFrame(phi).T
    y = []
    for i in range(0, n):
        y.append(train.iloc[i, 6])
    y = pd.DataFrame(y)
    [phi, y] = normliz(phi, y)
    return phi, y


def get_features_basis1_1(train):
    n = train.shape[0]
    dt = []
    for i in range(0, n):
        dt.append(datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S").timestamp())
    pick_long = []
    for i in range(0, n):
        pick_long.append(train.iloc[i, 1])
    pick_lat = []
    for i in range(0, n):
        pick_lat.append(train.iloc[i, 2])
    drop_long = []
    for i in range(0, n):
        drop_long.append(train.iloc[i, 3])
    drop_lat = []
    for i in range(0, n):
        drop_lat.append(train.iloc[i, 4])
    pass_count = []
    for i in range(0, n):
        pass_count.append(train.iloc[i, 5])
    weekends = []
    for i in range(0, n):
        if datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S").weekday() < 5:
            weekends.append(0)
        else:
            weekends.append(1)
    month = []
    for i in range(0, n):
        month.append(datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S").month)
    const = np.ones(n)
    weekday = []
    for i in range(0, n):
        weekday.append(datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S").weekday())
    time = []
    for i in range(0, n):
        a = datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S")
        time.append(3600 * a.hour + 60 * a.minute + a.second)
    dist = []
    for i in range(0, n):
        dist.append(distance(pick_lat[i], drop_lat[i], pick_long[i], drop_long[i]))
    phi = [const, dt, time, pick_long, pick_lat, drop_long, drop_lat, dist]
    # time, month, dt,
    phi = pd.DataFrame(phi).T
    [phi, y] = normliz(phi, phi[0])
    return phi


def get_features_basis1(train):
    n = train.shape[0]
    dt = []
    for i in range(0, n):
        dt.append(datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S").timestamp())
    pick_long = []
    for i in range(0, n):
        pick_long.append(train.iloc[i, 1])
    pick_lat = []
    for i in range(0, n):
        pick_lat.append(train.iloc[i, 2])
    drop_long = []
    for i in range(0, n):
        drop_long.append(train.iloc[i, 3])
    drop_lat = []
    for i in range(0, n):
        drop_lat.append(train.iloc[i, 4])
    pass_count = []
    for i in range(0, n):
        pass_count.append(train.iloc[i, 5])
    weekends = []
    for i in range(0, n):
        if datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S").weekday() < 5:
            weekends.append(0)
        else:
            weekends.append(1)
    month = []
    for i in range(0, n):
        month.append(datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S").month)
    const = np.ones(n)
    weekday = []
    for i in range(0, n):
        weekday.append(datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S").weekday())
    time = []
    for i in range(0, n):
        a = datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S")
        time.append(3600 * a.hour + 60 * a.minute + a.second)
    dist=[]
    for i in range(0, n):
        dist.append(distance(pick_lat[i], drop_lat[i], pick_long[i], drop_long[i]))
    phi = [const, dt, time, pick_long, pick_lat, drop_long, drop_lat, dist]
    '''weekday, time, month, dt,'''
    phi = pd.DataFrame(phi)
    phi = phi.T
    y = []
    for i in range(0, n):
        y.append(train.iloc[i, 6])
    y = pd.DataFrame(y)
    [phi, y] = normliz(phi, y)
    return phi, y


def get_features_basis2(train):
    n = train.shape[0]
    dt = []
    for i in range(0, n):
        dt.append(datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S").timestamp())
    pick_long = []
    for i in range(0, n):
        pick_long.append(train.iloc[i, 1])
    pick_lat = []
    for i in range(0, n):
        pick_lat.append(train.iloc[i, 2])
    drop_long = []
    for i in range(0, n):
        drop_long.append(train.iloc[i, 3])
    drop_lat = []
    for i in range(0, n):
        drop_lat.append(train.iloc[i, 4])
    pass_count = []
    for i in range(0, n):
        pass_count.append(train.iloc[i, 5])
    weekday = []
    for i in range(0, n):
        weekday.append(datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S").weekday())
    time = []
    for i in range(0, n):
        a = datetime.strptime(train.iloc[i, 0][:-4], "%Y-%m-%d %H:%M:%S")
        time.append(3600 * a.hour + 60 * a.minute + a.second)
    const = np.ones(n)
    phi = [const, weekday, time, dt, pick_long, pick_lat, drop_long, drop_lat, pass_count]
    phi = pd.DataFrame(phi)
    phi = phi.T
    y = []
    for i in range(0, n):
        y.append(train.iloc[i, 6])
    y = pd.DataFrame(y)
    [phi, y] = normliz(phi, y)
    return phi, y


def compute_RMSE(phi, w, y):
    error = (y - np.dot(phi, w))
    return np.linalg.norm(error)


def generate_output(phi_test, w):
    yp = np.matmul(phi_test, w)
    return yp


def closed_soln(phi, y):
    return np.linalg.pinv(phi).dot(y)


def gradient_descent(phi, y):
    n = phi.shape[1]
    lr = 0.000006
    w = np.random.rand(n, 1)
    w = pd.DataFrame(w)
    while np.linalg.norm(gradL_mse(phi, w, y)) > 0.01:
        w = w - lr * gradL_mse(phi, w, y)
    return w


def sgd(phi, y):
    n = phi.shape[1]
    lr = 0.00001
    w = np.random.rand(n, 1)
    # w = pd.DataFrame(w)
    # y = y.T
    phi = phi.to_numpy()
    y = y.to_numpy()
    while True:
        i = np.random.randint(phi.shape[0], size=1)
        grad = np.matmul(np.transpose(phi[i, :]), np.matmul(phi[i, :], w) - y[i])
        w = w - lr * grad
        # print(np.linalg.norm(grad))
        if np.linalg.norm(grad) < 0.00001:
            return w


def pnorm(phi, y, p):
    n = phi.shape[1]
    lr = 0.000006
    w = np.zeros(n)
    w = pd.DataFrame(w)
    lda = 0
    while np.linalg.norm(gradL_mse(phi, w, y) + p * lda * np.power(w, p - 1)) > 0.00001:
        w = w - lr * (gradL_mse(phi, w, y) + p * lda * np.power(w, p - 1))
        print(w)
    return w


def main():
    train = pd.read_csv('train.csv')
    dev = pd.read_csv('dev.csv')

    [phi_t, y_t] = get_features(train)
    [phi_t1, y_t1] = get_features_basis1(train)
    #[phi_t2, y_t2] = get_features_basis2(train)
    [phi_d1, y_d1] = get_features_basis1(dev)
    #[phi_d2, y_d2] = get_features_basis2(dev)
    w_b1 = pnorm(phi_t1, y_t1, 2)
    #w_b2 = pnorm(phi_t2, y_t2, 2)
    rmse_basis1 = compute_RMSE(phi_d1, w_b1, y_d1)
    #rmse_basis2 = compute_RMSE(phi_d2, w_b2, y_d2)
    print('Task 3: basis1')
    print(rmse_basis1)
    print('Task 3: basis2')
    #print(rmse_basis2)
    test = pd.read_csv('test.csv')
    phi_test = get_features_basis1_1(test)
    y = generate_output(phi_test.to_numpy(), w_b1.to_numpy())
    #y = (float(np.std(y_d1))*y+float(np.mean(y_d1)))
    #print(float(np.mean(y_d1)))
    np.savetxt('./sub.csv', y, delimiter=",")
    return


main()


