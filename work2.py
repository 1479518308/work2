import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
def cal_gmm(mean,std,xi):
    result = (1 / (math.sqrt(2 * math.pi) * std)) * \
             math.exp(-1 * (xi - mean) * (xi - mean) / (2 * std ** 2))
    return result


def e_step(a,mean,std,x):
    Expectations = np.zeros((len(a),len(x)))
    for k in range(len(x)):
        sump = sum(a[i]*cal_gmm(mean[i],std[i],x[k]) for i in range(len(a)))
        for i in range(len(a)):
            Expectations[i,k] = a[i]*cal_gmm(mean[i],std[i],x[k])/sump
    return Expectations

def m_step(Expectations,x,a):
    mean_list = []
    std_list = []
    a_list = []
    for i in range(len(a)):
        sume = sum(Expectations[i,:])
        a = sume/len(x)
        a_list.append(a)
        summ = sum(Expectations[i,k]*x[k] for k in range(len(x)))
        mean = summ/sume
        mean_list.append(mean)
        sums = sum(Expectations[i,k]*(x[k]-mean)**2 for k in range(len(x)))
        std = math.sqrt(sums/sume)
        std_list.append(std)
    return [mean_list,std_list,a_list]

def find(Expectations,x):

    x1 = []
    x2 = []
    for k in range(len(x)):
        if Expectations[0,k] >= 0.5:
            x1.append(x[k])
        else:
            x2.append(x[k])
    return x1,x2


def main(a,mean,std,x):
    count = 0
    while(1):
        Expectations=e_step(a,mean,std,x)
        #print(Expectations)
        if abs(mean[0] - m_step(Expectations, x, a)[0][0]) < 0.0001:
            break
        else:
            [mean, std, a] = m_step(Expectations, x, a)
        count += 1
    print("a:",a)
    print("std:", std)
    print("mean:", mean)
    print(count)
    x1,x2 = find(Expectations, x)
    return x1,x2


def draw(x1,x2):
    result = np.histogram(x1, bins=15)
    plt.plot(result[1][0:-1], result[0])
    result = np.histogram(x2, bins=20)
    plt.plot(result[1][0:-1], result[0])
    plt.xlabel('Height (cm)')
    plt.ylabel('Count')
    plt.title('Distribution of Heights')
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("height_data.csv")
    x = data['height'].values.tolist()
    a = [0.3,0.7]
    mean = [160,180]
    std = [3,5]
    x1,x2 = main(a,mean,std,x)
    #draw(x[:500],x[500:2000])
    #draw(x1, x2)
    #print(len(x1),len(x2))
    '''wrong = 0
    for xi in x1:
        if xi not in x[:500]:
            wrong += 1
    print(wrong)
    for xi in x2:
        if xi not in x[500:2000]:
            wrong += 1
    print(wrong)'''
