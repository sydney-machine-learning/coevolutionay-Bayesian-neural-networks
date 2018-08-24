
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
import os


def main():

	root=os.getcwd()
	root += "\PythonDatasets"
	print root
	dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]


	for problem in dirlist:

		directory = "PythonDatasets/" + problem + "/Results/ChartData/";

		weights = np.loadtxt(directory + "weightsout.txt")
		train = np.loadtxt(directory + "trainout.txt")
		test = np.loadtxt(directory + "testout.txt")

		x_test = np.arange(len(test[0]))
		x_train = np.arange(len(train[0]))

		fx_mu = test.mean(axis=0)
		fx_high = np.percentile(test, 95, axis=0)
		fx_low = np.percentile(test, 5, axis=0)

		fx_mu_tr = train.mean(axis=0)
		fx_high_tr = np.percentile(train, 95, axis=0)
		fx_low_tr = np.percentile(train, 5, axis=0)



		plt.plot(x_test, fx_mu, label='pred. (mean)', alpha=0.4)
		plt.plot(x_test, fx_low, label='pred.(5th percen.)', alpha=0.4)
		plt.plot(x_test, fx_high, label='pred.(95th percen.)', alpha=0.4)
		plt.plot(x_test, test[0], label='actual', color='black')
		plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Plot of Test Data for " + problem)
		plt.savefig(directory + 'test.png')
		plt.savefig(directory +  'test.svg', format='svg', dpi=600)

		# Save to all graphs folder
		plt.savefig("GRAPHS/" + problem + '_test.png')
		plt.savefig("GRAPHS/" + problem + '_test.svg', format='svg', dpi=600)


		plt.clf()
		# -----------------------------------------

		plt.plot(x_train, fx_mu_tr, label='pred. (mean)', alpha=0.4)
		plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)', alpha=0.4)
		plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)', alpha=0.4)
		plt.plot(x_train, train[0], label='actual', color='black')
		plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Plot of Train Data for " + problem)
		plt.savefig(directory + 'train.png')
		plt.savefig(directory + 'train.svg', format='svg', dpi=600)

		#Save to all graphs folder
		plt.savefig("GRAPHS/" + problem +'_train.png')
		plt.savefig("GRAPHS/" + problem +'_train.svg', format='svg', dpi=600)


		plt.clf()

		# Boxplot of weights
		mpl_fig = plt.figure()
		ax = mpl_fig.add_subplot(111)

		ax.boxplot(weights)

		ax.set_xlabel('Weights & Biases')
		ax.set_ylabel('Values')

		plt.legend(loc='upper right')

		plt.title("Boxplot of all W (weights and biases) for " + problem)
		plt.savefig(directory + 'weights.png')
		plt.savefig(directory + 'weights.svg', format='svg', dpi=600)

		# Save to all graphs folder
		plt.savefig("GRAPHS/" + problem + '_weights.png')
		plt.savefig("GRAPHS/" + problem + '_weights.svg', format='svg', dpi=600)
		plt.clf()


if __name__ == "__main__": main()
