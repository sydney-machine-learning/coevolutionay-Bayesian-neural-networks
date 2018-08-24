
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math



def main():

    weights = np.loadtxt("weightsout.txt")
    train = np.loadtxt("trainout.txt")
    test = np.loadtxt("testout.txt")


    x_test = np.arange(len(test[0]))
    x_train = np.arange(len(train[0]))

    fx_mu = test.mean(axis=0)
    fx_high = np.percentile(test, 95, axis=0)
    fx_low = np.percentile(test, 5, axis=0)

    fx_mu_tr = train.mean(axis=0)
    fx_high_tr = np.percentile(train, 95, axis=0)
    fx_low_tr = np.percentile(train, 5, axis=0)

    # rmse_tr = np.mean(rmse_train[int(burnin):])
    # rmsetr_std = np.std(rmse_train[int(burnin):])
    # rmse_tes = np.mean(rmse_test[int(burnin):])
    # rmsetest_std = np.std(rmse_test[int(burnin):])
    # print rmse_tr, rmsetr_std, rmse_tes, rmsetest_std
    # np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')
    #
    # ytestdata = testdata[:, input]
    # ytraindata = traindata[:, input]

    plt.plot(x_test, test[0], label='actual')
    plt.plot(x_test, fx_mu, label='pred. (mean)')
    plt.plot(x_test, fx_low, label='pred.(5th percen.)')
    plt.plot(x_test, fx_high, label='pred.(95th percen.)')
    plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
    plt.legend(loc='upper right')

    plt.title("Plot of Test Data")
    plt.savefig('test.png')
    plt.savefig('test.svg', format='svg', dpi=600)
    plt.clf()
    # -----------------------------------------
    plt.plot(x_train, train[0], label='actual')
    plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
    plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
    plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
    plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
    plt.legend(loc='upper right')

    plt.title("Plot of Train Data")
    plt.savefig('train.png')
    plt.savefig('train.svg', format='svg', dpi=600)
    plt.clf()


    #Boxplot of weights
    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)

    ax.boxplot(weights)

    ax.set_xlabel('[W1] [B1] [W2] [B2]')
    ax.set_ylabel('Posterior')

    plt.legend(loc='upper right')

    plt.title("Boxplot of Posterior W (weights and biases)")
    plt.savefig('weights.png')
    plt.savefig('weights.svg', format='svg', dpi=600)

    plt.clf()


if __name__ == "__main__": main()
