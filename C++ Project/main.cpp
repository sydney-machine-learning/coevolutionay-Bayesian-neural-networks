/*
--Cooperative Cooperative Coevolution of Feed-Foward Neural Networks(with G3-PCX)--
Developed by: Rohitash Chandra
Cleaned, Documented and Modified by: Shelvin Chand and Ravneil Nand

*/

#define _USE_MATH_DEFINES


#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <ctime>
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<ctype.h>
#include <time.h>
#include <windows.h>


time_t TicTime;
time_t TocTime;
using namespace::std;
//type definitions for vectors
typedef vector<double> Layer;
typedef vector<double> Nodes;
typedef vector<double> Frame;
typedef vector<int> Sizes;
typedef vector<vector<double> > Weight;
typedef vector<vector<double> > Data;
typedef vector<vector<double> > TwoDVector_Double;
typedef vector<double> OneDVector_Double;


TwoDVector_Double pos_w;
OneDVector_Double pos_tau;

TwoDVector_Double fxtrain_samples;
TwoDVector_Double fxtest_samples;

OneDVector_Double rmse_train;
OneDVector_Double rmse_test;


#define UseCCMCMC
//#define usememe
//#define bcicn
//#define UseLocalCooperativeNeuroEvolution
//#define UseLocalCooperativeNeuroEvolutionPartial
//#define UseBayesianMemetic
//#define UseBayesianMemetic_MCNEINMCMC
//#define UseMCNE
//#define UseBayesianCooperativeNeuroEvolution
//#define UseBCNE1Step
//#define UseBCNEGenerateOnly
//#define UseBCNELoop
//
//#define UseSGD
//#define UseBCNE
//#define UseCC
//#define Memory
//#define UseMCMC
//#define LSMaster
//#define neuronlevel
const int LayersNumber = 3; //total number of layers.

							//ofstream fcout("Data.txt");


string DataResultLocation = "";

//constants
const int culturalPopSize = 20;
const int CCPOPSIZE = 300;//300;//population size // Got optimal result with this size
const int minhidden = 1;//minimum number of hidden layers
const int maxhidden = 3;//maximum number of hidden layers
const int depthbegin = 10;
const int depthend = 10;
const int depthinc = 5;
const int fixedhidden = 4;

int maxgen = 10000;//number of function evaluations
int maxgenLS = 10000;//number of function evaluations
int localmaxgen = 5000;
int localminimum = 2000;

int MaxNumNeurons = 0;
int HiddenIncrementBy = 0;
int HiddenNeuronsStart = 0;



//int MaxModularity = 0.5 * (maxgen - localmaxgen);
int run = 1;
//int countfe = 1;

//data set size and number of neurons involved
const double mintrain = 95;

int trainsize = 194;//size of training set
int testsize = 166;//size of test set
int validsize = 166;

int input = 3; //input dimensions
const int output = 1; //output dimensions
double MinimumError = 0.0478;//generalisation best previous experiments RNN (Laser)


string trainfile = "train.txt";//training file
string testfile = "test.txt"; //test file
string validfile = "validation.txt"; //test file

									 //Decomposition method used
									 //#define neuronlevel   //hiddenNeuron, weightlevel, neuronlevel,networklevel -->Decomposition Method


#define sigmoid   //tanh, sigmoid --> For positive values use sigmoid else use tanh

const double alpha = 0.00001;
const double MaxErrorTollerance = 0.20;//error tolerance --> used to round up/down values
const double MomentumRate = 0;
const double Beta = 0;
int row;
int col;
int layer;
int r;
int x;
int y;

double weightdecay = 0.005;

#define rosen          // choose the function:
#define EPSILON 1e-50

#define EPSILONVALID 1e-3

#define MAXFUN 50000000  //upper bound for number of function evaluations
#define MINIMIZE 1      //set 1 to minimize and -1 to maximize
#define LIMIT 1e-20     //accuracy of best solution fitness desired
#define KIDS 2          //pool size of kids to be formed (use 2,3 or 4)
#define M 1             //M+2 is the number of parents participating in xover (use 1)
#define family 2        //number of parents to be replaced by good individuals(use 1 or 2)
#define sigma_zeta 0.1
#define sigma_eta 0.1   //variances used in PCX (best if fixed at these values)
#define  NoSpeciesCons 300
#define NPSize KIDS + 2   //new pop size
#define RandParent M+2     //number of parents participating in PCX
#define MAXRUN 10      //number of runs each with different random initial population

double d_not[CCPOPSIZE];
double seed, basic_seed;
int RUN;

class ChartData
{
public:
	void WriteAccuracyData(OneDVector_Double data, ofstream &file, int size) {

		for (int i = 0; i < size; i++)
		{
			file << data[i] << " ";
		}
		file << "\n";
	}

	void WriteAccuracyData(TwoDVector_Double data, ofstream &file, int size) {

		for (int i = 0; i < size; i++)
		{
			for (int y = 0; y < data[i].size(); y++)
			{
				file << data[i][y] << " ";
			}
		}
		file << "\n";
	}

};


/*
--InitialiseData--
Function is used to setup the dataset, input and output vectors.
1.The dataset vector is initialised with the data from the file
2.The dataset vector is then used to initialize the output/input vectors
*/
class TrainingExamples {
	friend class NeuralNetwork;

protected:
	//class variables
	Data  InputValues;
	Data  DataSet;
	Data  OutputValues;
	string FileName;
	int Datapoints;
	int colSize;
	int inputcolumnSize;
	int outputcolumnSize;
	int datafilesize;

public:
	//function declarations
	TrainingExamples()
	{
		//constructor
	};
	//overriden constructor
	TrainingExamples(string File, int size, int length, int inputsize, int outputsize) {
		//initialize functions and class variables
		inputcolumnSize = inputsize;
		outputcolumnSize = outputsize;
		datafilesize = inputsize + outputsize;
		colSize = length;
		Datapoints = size;

		FileName = File;
		InitialiseData();
	}

	void printData();

	void InitialiseData();

	void RandomizeDataOrder();

	int GetSize() {
		return Datapoints;
	}
	Data GetOutputValues() {
		return OutputValues;
	}

};

/*
--InitialiseData--
Function is used to setup the dataset, input and output vectors.
1.The dataset vector is initialised with the data from the file
2.The dataset vector is then used to initialize the output/input vectors
*/
void TrainingExamples::RandomizeDataOrder()
{

	int randomPosition = 0;

	//Shuffle data
	for (int i = 0; i < Datapoints; i++)
	{
		randomPosition = rand() % ((Datapoints - 1) - 1);
		swap(InputValues[i], InputValues[randomPosition]);
		swap(OutputValues[i], OutputValues[randomPosition]);
	}

}

/*
--InitialiseData--
Function is used to setup the dataset, input and output vectors.
1.The dataset vector is initialised with the data from the file
2.The dataset vector is then used to initialize the output/input vectors
*/
void TrainingExamples::InitialiseData()
{

	ifstream in(FileName.c_str());
	if (!in) {
		cout << endl << "failed to open file" << endl;//error message for reading from file
	}

	//initialise dataset vectors
	for (int r = 0; r < Datapoints; r++)
		DataSet.push_back(vector<double>());//create a matrix for holding the data

	for (int row = 0; row < Datapoints; row++) {
		for (int col = 0; col < colSize; col++)
			DataSet[row].push_back(0);//initialize dataset with 0s
	}
	// cout<<"printing..."<<endl;
	for (int row = 0; row < Datapoints; row++)
		for (int col = 0; col < colSize; col++)
			in >> DataSet[row][col];//fill dataset matrix with values from the training examples file
									//-------------------------
									//initialise intput vectors
	for (int r = 0; r < Datapoints; r++)
		InputValues.push_back(vector<double>());

	for (int row = 0; row < Datapoints; row++)
		for (col = 0; col < inputcolumnSize; col++)
			InputValues[row].push_back(0);//initialise with 0s

	for (int row = 0; row < Datapoints; row++)
		for (int col = 0; col < inputcolumnSize; col++)
			InputValues[row][col] = DataSet[row][col];//read values from the dataset vector

													  //initialise output vectors
	for (int r = 0; r < Datapoints; r++)
		OutputValues.push_back(vector<double>());//create matrix

	for (int row = 0; row < Datapoints; row++)
		for (int col = 0; col < outputcolumnSize; col++)
			OutputValues[row].push_back(0);//initialse with 0s

	for (int row = 0; row < Datapoints; row++)
		for (int col = 0; col < outputcolumnSize; col++)
			OutputValues[row][col] = DataSet[row][col + inputcolumnSize];//read values from last layer of dataset

	in.close();//close connection
}
/*
--printData--
Displays to the user the entire training set including all the input values and the expected output values

*/
void TrainingExamples::printData()
{
	//ofstream outf;
	//outf.open("inputvalues.txt");
	//cout << "printing...." << endl;
	//cout << "Entire Data Set.." << endl;
	//for (int row = 0; row < Datapoints; row++) {
	//	for (int col = 0; col < colSize; col++)
	//		cout << DataSet[row][col] << " ";//output entire set
	//	cout << endl;
	//}

	//outf << endl << "Input Values.." << endl;
	//for (int row = 0; row < Datapoints; row++) {
	//	outf << "row  " << row << endl;
	//	for (int col = 0; col < inputcolumnSize; col++)
	//		outf << InputValues[row][col] << " ";//output only input values
	//	outf << endl;
	//}
	////*************************************
	cout << endl << "Expected Output Values.." << endl;

	for (int row = 0; row < Datapoints; row++) {
		for (int col = 0; col < outputcolumnSize; col++)
			cout << OutputValues[row][col] << " ";//print output values
		cout << endl;
	}
	//outf.close();
}

/*  --Layer Class--
Layer Class is used to manage the different layers of the neural network
It manages the weights, neuron values, bias factor and changes in weights and bais.
Each element is a vector or vector containing other vectors forming a multi-dimensional matrix
This allows for managing of corresponding weights and values
*/
class Layers {
	friend class NeuralNetwork;

protected:
	//class variables
	Weight Weights;//neural network weights
	Weight WeightChange;//change in weight after each iteration
	Weight H;
	Layer Outputlayer;
	Layer Bias;//keeping track of the bias factor in the network
	Layer B;
	Layer Gates;
	Layer BiasChange;//keeping track of change in Bias factor
	Layer Error;//error after each iteration

public:
	//functions
	Layers()
	{
		//contructor method
	}
};

class VectorManipulation
{
public:
	TwoDVector_Double Generate2DVector_Double(int rows, int columns, double initialvalue)
	{
		vector<vector<double> > vect;

		for (int i = 0; i < rows; i++)
		{
			vector<double> newvec;

			for (int y = 0; y < columns; y++)
			{
				newvec.push_back(initialvalue);
			}

			vect.push_back(newvec);
		}

		return vect;

	}

	void Print2DVector(vector<vector<double> > vec)
	{
		cout << "\nPrinting 2D Vector" << endl;

		for (int i = 0; i < vec.size(); i++)
		{


			for (int y = 0; y < vec[i].size(); y++)
			{
				cout << vec[i][y] << " ";
			}

			cout << endl;
		}
		cout << "\nEnd Printing 2D Vector" << endl;
	}

	void Print1DVector(vector<double> vec)
	{
		cout << "\nPrinting 1D Vector" << endl;
		for (int i = 0; i < vec.size(); i++)
		{
			cout << vec[i] << " " << endl;
		}
		cout << "\nEnd Printing 1D Vector" << endl;

	}

	OneDVector_Double Generate1DVector_Double(int rows, double initialvalue)
	{

		vector<double> vect;

		for (int i = 0; i < rows; i++)
		{
			vect.push_back(initialvalue);
		}

		return vect;

	}
};

/*  --NeuralNetwork Class--
This class is used to represent the neural network structure and its functionality
It makes use of the layer class to structure the network

*/
class NeuralNetwork {
	friend class GeneticAlgorithmn;
	friend class CoEvolution;

protected:
	//class variables
	Layers nLayer[LayersNumber];
	double Heuristic;
	int StringSize;
	Layer ChromeNeuron;
	int NumEval;
	Data Output;
	Sizes layersize;
	double NMSE, ValidNMSE;

public:
	//class functions
	NeuralNetwork()
	{
		//constructor
	}

	NeuralNetwork(Sizes layer)
	{
		layersize = layer;
		StringSize = (layer[0] * layer[1]) + (layer[1] * layer[2]) + (layer[1] + layer[2]);
	}

	Data GetOutput()
	{
		return Output;

	}
	double Random();

	double Sigmoid(double ForwardOutput);

	double NMSError() { return NMSE; }

	double ValidNMSError() { return ValidNMSE; }

	void CreateNetwork(Sizes Layersize, TrainingExamples TraineeSamples);

	void ForwardPass(TrainingExamples TraineeSamp, int patternNum, Sizes Layersize);

	void BackwardPass(TrainingExamples TraineeSamp, double LearningRate, int patternNum, Sizes Layersize);
	void BackwardPassBGD(TrainingExamples TraineeSamp, double LearningRate, int patternNum, Sizes Layersize);

	void PrintWeights(Sizes Layersize);// print  all weights

	bool ErrorTolerance(TrainingExamples TraineeSamples, Sizes Layersize, double TrainStopPercent);

	double SumSquaredError(TrainingExamples TraineeSamples, Sizes Layersize);

	TwoDVector_Double BackPropogationCC(TrainingExamples TraineeSamples, double LearningRate, Sizes Layersize, string Savefile, bool load, int epochs, int individualsize);
	int BackPropogation(TrainingExamples TraineeSamples, double LearningRate, Sizes Layersize, string Savefile, bool load);
	int BackPropogationExhausted(TrainingExamples TraineeSamples, double LearningRate, Sizes Layersize, string Savefile, bool load);

	int BackPropogationMemory(TrainingExamples TraineeSamples, double LearningRate, Sizes Layersize, string Savefile, bool load, double *CurrentBestGradient, int *CurrentBestIndex, int CurrentIndex, double *CurrentBestError, int *CurrentBestErrorIndex, ofstream &bcout);

	Layer BackpropogationCICC(Layer NeuronChrome, TrainingExamples TraineeSamples, double LearningRate, Sizes Layersize, string Savefile, bool load, int EpochMax, int Island);

	void SaveLearnedData(Sizes Layersize, string filename);
	void SaveLearnedDataMemory(Sizes Layersize, string filename, vector<double>* memorySnapShot);

	double  NormalisedMeanSquaredError(TrainingExamples TraineeSamples, Sizes Layersize);

	double Rate(TrainingExamples TraineeSamples, Sizes Layersize);

	void LoadSavedData(Sizes Layersize, string filename);
	void LoadSavedDataMemory(Sizes Layersize, vector<double>* memorySnapShot);

	void SaveErrorToFile(Sizes Layersize, string filename, double error);


	double TestLearnedData(Sizes Layersize, string filename, int  size, string load, int inputsize, int outputsize);

	Layer  Neurons_to_chromes(int Island);
	double  CountLearningData(TrainingExamples TraineeSamples, int temp, Sizes Layersize);

	double  TestTrainingData(Sizes Layersize, string filename, int  size, string load, int inputsize, int outputsize, ofstream & out2);

	double CountTestingData(TrainingExamples TraineeSamples, int temp, Sizes Layersize);

	double LearningRate(TrainingExamples TraineeSamples, Sizes Layersize, int pattern);

	bool CheckOutput(TrainingExamples TraineeSamples, int pattern, Sizes Layersize);

	//void  ChoromesToNeurons(  Layer NeuronChrome);
	void  ChoromesToNeurons(Layer NeuronChrome, int Island);

	double ForwardFitnessPass(Layer NeuronChrome, TrainingExamples Test, int Island);

	OneDVector_Double Evaluate_Proposal(TrainingExamples data, OneDVector_Double w, Sizes Layersize)
	{
		int end = static_cast<int>(Layersize.size()) - 1; //know the last layer

		VectorManipulation vectormanipulate;
		OneDVector_Double fxout = vectormanipulate.Generate1DVector_Double(data.GetSize(), 0);

		//Load w into network
		ChoromesToNeurons(w, 1);//encode into neural network the best cc individual

								//Load data for testing

								//For each test data row
		for (int pattern = 0; pattern < data.GetSize(); pattern++) {
			ForwardPass(data, pattern, Layersize); //Do forward pass save output to fx
		}

		//write to file the desired output, actual output and error
		for (int pattern = 0; pattern < data.GetSize(); pattern++)
		{
			for (int output = 0; output < Layersize[end]; output++)
			{

				fxout[pattern] = Output[pattern][output];

			}
		}

		return fxout;

	}
};

/*
--Random--
Is used to generate random numbers which are used to initialize weights and neurons when the network is created
*/
double NeuralNetwork::Random()//method for assigning random weights to Neural Network connections
{
	int chance;
	double randomWeight;
	double NegativeWeight;
	chance = rand() % 2;//randomise between positive and negative

	if (chance == 0) {
		randomWeight = rand() % 100;
		return randomWeight*0.05;//assign positive weight
	}

	else {
		NegativeWeight = rand() % 100;
		return NegativeWeight*-0.05;//assign negative weight
	}
}

/*
--Sigmoid--
Function to convert weighted_sum into a value between  -1 and 1
*/
double NeuralNetwork::Sigmoid(double ForwardOutput)
{
	double ActualOutput;
#ifdef sigmoid
	ActualOutput = (1.0 / (1.0 + exp(-1.0 * (ForwardOutput))));
#endif

#ifdef tanh
	ActualOutput = (exp(2 * ForwardOutput) - 1) / (exp(2 * ForwardOutput) + 1);
#endif
	return  ActualOutput;
}

/*
--CreateNetwork--
This function is responsible for setting the overall network structure and initially the structure with random weights, random bais factors and 0s for each neuron value.
*/
void NeuralNetwork::CreateNetwork(Sizes Layersize, TrainingExamples TraineeSamples)//create network and initialize the weights
{
	int end = static_cast<int>(Layersize.size()) - 1;

	for (layer = 0; layer < Layersize.size() - 1; layer++) {//go through each layer
		for (int r = 0; r < Layersize[layer]; r++)//go through the number of connections betwneen layers i.e the weights for the connections between layers
			nLayer[layer].Weights.push_back(vector<double>()); //each layer will have matrix of vectors representing  the weights

		for (int row = 0; row < Layersize[layer]; row++)
			for (int col = 0; col < Layersize[layer + 1]; col++)
				nLayer[layer].Weights[row].push_back(Random());//initialize weights to random values for all layers

		for (int r = 0; r < Layersize[layer]; r++)// the weight change vector will have a vecctor(of type double) inside it for each of the elements
			nLayer[layer].WeightChange.push_back(vector<double>()); // this will be used to keep track of the change in weights after each iteration
																	//again this depends on number of connections(weights) we have between layers

		for (int row = 0; row < Layersize[layer]; row++)
			for (col = 0; col < Layersize[layer + 1]; col++)
				nLayer[layer].WeightChange[row].push_back(0);//initialize all the elements in weighchange array with 0 as initially we have no change

		for (int r = 0; r < Layersize[layer]; r++)
			nLayer[layer].H.push_back(vector<double>());//create matrix

		for (int row = 0; row < Layersize[layer]; row++)
			for (int col = 0; col < Layersize[layer + 1]; col++)
				nLayer[layer].H[row].push_back(0);//initialize all the elements in H with 0 for each layer
	}

	for (layer = 0; layer < Layersize.size(); layer++) {
		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].Outputlayer.push_back(0);//initialize neurons of each layer with 0s

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].Bias.push_back(Random());//the bias for each each layer and connection will be a random value

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].Gates.push_back(0);//initialize gates vector with 0s

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].B.push_back(0);//initialize with 0s

		for (row = 0; row < Layersize[layer]; row++)//for each connection we will also keep track of change in bias factor
			nLayer[layer].BiasChange.push_back(0);//initially it will be all 0

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].Error.push_back(0);// intialize error vector for each layer with 0s
	}

	for (r = 0; r < TraineeSamples.Datapoints; r++)
		Output.push_back(vector<double>());//the output vector will have nested vectors(of type double) inside it forming a matrix..
										   //depending on the number of training examples

	for (row = 0; row < TraineeSamples.Datapoints; row++)
		for (col = 0; col < Layersize[end]; col++)
			Output[row].push_back(0);// intialize all the rows in the output vector with 0s
	for (row = 0; row < StringSize; row++)
		ChromeNeuron.push_back(0); //intialise the ChromeNeuron vector with 0s
}

/*  --Forward Pass--
1. Feed the network values from the training set through the input layer
2. Apply the given weights to the input values for each layer of the network
3. Apply the bias factor to the computed weighted sum
4. Convert the resulting values using the sigmodi function to a value between -1 and 1
3. Once the final layer is reached output the computed values from the network.
*/
void NeuralNetwork::ForwardPass(TrainingExamples TraineeSamples, int patternNum, Sizes Layersize)
{
	//declaring essential variables
	double WeightedSum = 0;
	double ForwardOutput = 0;//to hold output value between -1 and 1
	int end = static_cast<int>(Layersize.size()) - 1; //know the last layer

	for (int row = 0; row < Layersize[0]; row++)
		nLayer[0].Outputlayer[row] = TraineeSamples.InputValues[patternNum][row];//initializing first layer with values from training examples

	for (int layer = 0; layer < Layersize.size() - 1; layer++) {//go layer by layer calculating forward pass...i.e multiplying neuron values by the connection weights
		for (int y = 0; y < Layersize[layer + 1]; y++) {
			for (int x = 0; x < Layersize[layer]; x++) {
				WeightedSum += (nLayer[layer].Outputlayer[x] * nLayer[layer].Weights[x][y]);//multiplying weights from the weights matrix by the value in the corresponding neuron  to determine value for neuron in next layer

				ForwardOutput = WeightedSum - nLayer[layer + 1].Bias[y]; //subtracting the bias weight from the weighted sum giving the new value for the neuron in the next layer to which these weights are connected
			}

			nLayer[layer + 1].Outputlayer[y] = Sigmoid(ForwardOutput);//convert the weighted sum to a value between 1 and -1 which will be the new value in the neuron

			WeightedSum = 0;//reset weighted sum to 0 for next neuron
		}
		WeightedSum = 0;//reset weighted sum to 0 for next layer
	}//end layer

	 //--------------------------------------------
	for (int output = 0; output < Layersize[end]; output++) {
		Output[patternNum][output] = nLayer[end].Outputlayer[output];//setting the values for the final output in the last layer.
	}
}

/*  --Backward Pass--
Go backwards in the network trying to adjust weights in order to reach the desired output.
1. Find out the error gradient for the output layer
2. Find out the error gradient for the hidden layers
3. Compute the change in weight for each layer
4. Update the weights
*/
void NeuralNetwork::BackwardPassBGD(TrainingExamples TraineeSamp, double LearningRate, int patternNum, Sizes Layersize)
{
	//patternNum = 0;
	int end = static_cast<int>(Layersize.size()) - 1;// know the end layer
	double temp = 0;
	double totalGradientError = 0;

	// compute error gradient for output neurons

	//----- Batch Gradient Descent
	for (int output = 0; output < Layersize[end]; output++) {

		for (int pattern = 0; pattern < TraineeSamp.InputValues.size(); pattern++)
		{
			ForwardPass(TraineeSamp, pattern, Layersize); //forward pass through the network to calculate output values
			totalGradientError += (Output[pattern][output] * (1 - Output[pattern][output]))*(TraineeSamp.OutputValues[pattern][output] - Output[pattern][output]);
		}

		nLayer[end].Error[output] = totalGradientError / TraineeSamp.InputValues.size();
	}
	//----------------------------------------

	for (int layer = static_cast<int>(Layersize.size()) - 2; layer != 0; layer--) {
		for (x = 0; x < static_cast<int>(Layersize[layer]); x++) {  //inner layer
			for (y = 0; y < static_cast<int>(Layersize[layer + 1]); y++) { //outer layer
				temp += (nLayer[layer + 1].Error[y] * nLayer[layer].Weights[x][y]);
			}
			nLayer[layer].Error[x] = nLayer[layer].Outputlayer[x] * (1 - nLayer[layer].Outputlayer[x]) * temp;//compute error gradient for each hidden neuron

			temp = 0.0;//reset temp for the next neuron
		}
		temp = 0.0; //reset temp for the next layer
	}

	double tmp;
	int layer = 0;
	for (layer = static_cast<int>(Layersize.size()) - 2; layer != -1; layer--) {//go through all layers
		for (x = 0; x < Layersize[layer]; x++) {  //inner layer
			for (y = 0; y < Layersize[layer + 1]; y++) { //outer layer
				tmp = ((LearningRate * nLayer[layer + 1].Error[y] * nLayer[layer].Outputlayer[x]));// calculate change in weight i.e error correctection
				nLayer[layer].Weights[x][y] += (tmp - (alpha * tmp));//update weight
			}
		}
	}

	double tmp1;

	for (layer = static_cast<int>(Layersize.size()) - 1; layer != 0; layer--) {//go through all layers
		for (y = 0; y < Layersize[layer]; y++) {
			tmp1 = ((-1 * LearningRate * nLayer[layer].Error[y]));//calculate change in bias
			nLayer[layer].Bias[y] += (tmp1 - (alpha * tmp1));//updated bias of layer
		}
	}
}

/*  --Backward Pass--
Go backwards in the network trying to adjust weights in order to reach the desired output.
1. Find out the error gradient for the output layer
2. Find out the error gradient for the hidden layers
3. Compute the change in weight for each layer
4. Update the weights
*/
void NeuralNetwork::BackwardPass(TrainingExamples TraineeSamp, double LearningRate, int patternNum, Sizes Layersize)
{
	int end = static_cast<int>(Layersize.size()) - 1;// know the end layer
	double temp = 0;

	// compute error gradient for output neurons
	for (int output = 0; output < Layersize[end]; output++) {
		nLayer[end].Error[output] = (Output[patternNum][output] * (1 - Output[patternNum][output]))*(TraineeSamp.OutputValues[patternNum][output] - Output[patternNum][output]);
	}
	//----------------------------------------

	for (int layer = static_cast<int>(Layersize.size()) - 2; layer != 0; layer--) {
		for (x = 0; x < static_cast<int>(Layersize[layer]); x++) {  //inner layer
			for (y = 0; y < static_cast<int>(Layersize[layer + 1]); y++) { //outer layer
				temp += (nLayer[layer + 1].Error[y] * nLayer[layer].Weights[x][y]);
			}
			nLayer[layer].Error[x] = nLayer[layer].Outputlayer[x] * (1 - nLayer[layer].Outputlayer[x]) * temp;//compute error gradient for each hidden neuron

			temp = 0.0;//reset temp for the next neuron
		}
		temp = 0.0; //reset temp for the next layer
	}

	double tmp;
	int layer = 0;
	for (layer = static_cast<int>(Layersize.size()) - 2; layer != -1; layer--) {//go through all layers
		for (x = 0; x < Layersize[layer]; x++) {  //inner layer
			for (y = 0; y < Layersize[layer + 1]; y++) { //outer layer
				tmp = ((LearningRate * nLayer[layer + 1].Error[y] * nLayer[layer].Outputlayer[x]));// calculate change in weight i.e error correctection
				nLayer[layer].Weights[x][y] += (tmp - (alpha * tmp));//update weight
			}
		}
	}

	double tmp1;

	for (layer = static_cast<int>(Layersize.size()) - 1; layer != 0; layer--) {//go through all layers
		for (y = 0; y < Layersize[layer]; y++) {
			tmp1 = ((-1 * LearningRate * nLayer[layer].Error[y]));//calculate change in bias
			nLayer[layer].Bias[y] += (tmp1 - (alpha * tmp1));//updated bias of layer
		}
	}
}

/*
--ErrorTolerance--
Go through all the calculated output and if the output falls within a certain range then round up or round down the value to get a whole number such as 1 or 0
This means that the output falls within an acceptable range to be classified under a specific category
*/
bool NeuralNetwork::ErrorTolerance(TrainingExamples TraineeSamples, Sizes Layersize, double TrainStopPercent)
{
	//declare essential variables
	double count = 0;
	int total = TraineeSamples.Datapoints;
	double accepted = total;
	double desiredoutput;
	double actualoutput;
	double Error;
	int end = static_cast<int>(Layersize.size()) - 1;

	//go through all training samples
	for (int pattern = 0; pattern < TraineeSamples.Datapoints; pattern++) {
		Layer Desired;
		Layer Actual;

		for (int i = 0; i < Layersize[end]; i++)
			Desired.push_back(0);//initialize vector for desired output with 0s
		for (int j = 0; j < Layersize[end]; j++)
			Actual.push_back(0);//initialize vector for actual output with 0s

		for (int output = 0; output < Layersize[end]; output++) {
			desiredoutput = TraineeSamples.OutputValues[pattern][output];
			actualoutput = Output[pattern][output];

			Desired[output] = desiredoutput;
			//if output falls within a certain range you can round it up or down for classification
			if ((actualoutput >= 0) && (actualoutput <= 0.2))
				actualoutput = 0;//round down
			else if ((actualoutput <= 1) && (actualoutput >= 0.8))
				actualoutput = 1;//round up

			Actual[output] = actualoutput;
		}
		int confirm = 0;

		for (int b = 0; b < Layersize[end]; b++) {
			if (Desired[b] == Actual[b])
				confirm++;//there is match so move on to next neuron

			if (confirm == Layersize[end])
				count++;//if an instance is correctly predicted meaning all the values of the output layer neurons match the desired output
			confirm = 0;//reset for next layer
		}
	}
	if (count == accepted)
		return false;

	return true;
}

/*
--SumSquardedError--
1.Calculated error
2.Square the error
3.Add to cumulative overall error and repeat steps 1 to 3 for all output values
4. Square root (sum/(no. of training samples * no. of neurons in output layer)) and return result-RMSE

*/
double NeuralNetwork::SumSquaredError(TrainingExamples TraineeSamples, Sizes Layersize)
{
	int end = static_cast<int>(Layersize.size()) - 1;
	double Sum = 0;
	double Error = 0;
	double ErrorSquared = 0;

	for (int pattern = 0; pattern < TraineeSamples.Datapoints; pattern++) {
		for (int output = 0; output < Layersize[end]; output++) {
			Error = fabs(TraineeSamples.OutputValues[pattern][output]) - fabs(Output[pattern][output]);
			ErrorSquared += (Error * Error);//square the error
		}

		Sum += (ErrorSquared);//add to cumulative error
		ErrorSquared = 0;//set error squared variable to 0 for next layer
	}
	return sqrt(Sum / TraineeSamples.Datapoints*Layersize[end]); //return square root of sum / (no. of training samples * no. of neurons in output layer)
}

/*
--Normalised Mean Squared Error--

*/
double NeuralNetwork::NormalisedMeanSquaredError(TrainingExamples TraineeSamples, Sizes Layersize)
{
	//variable declaration
	int end = static_cast<int>(Layersize.size()) - 1;
	double Sum = 0;
	double Sum2 = 0;
	double Error = 0;
	double ErrorSquared = 0;
	double Error2 = 0;
	double ErrorSquared2 = 0;
	double meany = 0;

	for (int pattern = 0; pattern < TraineeSamples.Datapoints; pattern++) {// go through all data points
		for (int input = 0; input < Layersize[0]; input++) { //input layer
			meany += fabs(TraineeSamples.InputValues[pattern][input]); //calculate mean
		}
		meany /= (Layersize[0] * TraineeSamples.Datapoints);

		for (int output = 0; output < Layersize[end]; output++) { //output layer
			Error2 = fabs(TraineeSamples.OutputValues[pattern][output]) - meany; //error
			Error = fabs(TraineeSamples.OutputValues[pattern][output]) - fabs(Output[pattern][output]);// error between expected and actual output
			ErrorSquared += (Error * Error);//square the error
			Sum += (ErrorSquared);//add to cumulative error
			ErrorSquared = 0;
			ErrorSquared2 += (Error2 * Error2);//square the value
		}

		meany = 0;
		Sum += (ErrorSquared);
		Sum2 += (ErrorSquared2);
		ErrorSquared = 0;
		ErrorSquared2 = 0;
	}

	return Sum / Sum2;//normalise
}

/*
--Rate--
1. Calculate Error for each output
2. Square the Error
3. Add error to cumulative error and repeat steps 1 to 3 for all output values
4. Divide the sum/(no. of training samples * no. of neurons in output layer) and return results
*/
double NeuralNetwork::Rate(TrainingExamples TraineeSamples, Sizes Layersize)
{
	//variable declaration
	int end = static_cast<int>(Layersize.size()) - 1;
	double Sum = 0;
	double Error = 0;
	double ErrorSquared = 0;
	double rate = 0;

	for (int pattern = 0; pattern < TraineeSamples.Datapoints; pattern++) {
		for (int output = 0; output < Layersize[end]; output++) {
			Error = TraineeSamples.OutputValues[pattern][output] - Output[pattern][output];//calculate difference between expected and actual output
			ErrorSquared += (Error * Error);//square the error
		}
		Sum += (ErrorSquared);//add to cumulative error
		ErrorSquared = 0;//reset to 0 for next training sample
	}

	rate = Sum / (TraineeSamples.Datapoints *(Layersize[end]));//divide sum_error_squared by (no. of training samples * no. of neurons in output layer)
	return rate;
}

/*
--PrintWeights--

Output the weights, bias factor, error term and neuron values
*/
void NeuralNetwork::PrintWeights(Sizes Layersize)//output the values of all the connection weights
{
	int end = static_cast<int>(Layersize.size()) - 1;

	for (int layer = 0; layer < Layersize.size() - 1; layer++) {
		cout << layer << "  Weights::" << endl << endl;
		for (int row = 0; row < Layersize[layer]; row++) {
			for (int col = 0; col < Layersize[layer + 1]; col++)
				cout << nLayer[layer].Weights[row][col] << " "; //output all values from the weights matrix for all layers
			cout << endl;
		}
		cout << endl << layer << "  WeightsChange::" << endl << endl;

		for (int row = 0; row < Layersize[layer]; row++) {
			for (int col = 0; col < Layersize[layer + 1]; col++)
				cout << nLayer[layer].WeightChange[row][col] << " ";//output all values from the weightchange matrix for all layers
			cout << endl;
		}

		cout << "--------------" << endl;
	}

	for (int layer = 0; layer < Layersize.size(); layer++) {
		cout << endl << layer << "  Outputlayer::" << endl << endl;//output values from outputlayer
		for (int row = 0; row < Layersize[layer]; row++)
			cout << nLayer[layer].Outputlayer[row] << " ";

		cout << endl << layer << "  Bias::" << endl << endl;
		for (int row = 0; row < Layersize[layer]; row++)
			cout << nLayer[layer].Bias[row] << " ";//output values from the Bias Matrix for each layer

		cout << endl << layer << "  Error::" << endl << endl;
		for (int row = 0; row < Layersize[layer]; row++)
			cout << nLayer[layer].Error[row] << " "; //output values from the error matrix for each layer
		cout << "----------------" << endl;
	}
}

/*
--SaveLearnedData--
Save the network weights and bias values which were able to achieve optimal results to file
*/
void NeuralNetwork::SaveLearnedData(Sizes Layersize, string filename)//save data to file
{
	ofstream out;
	out.open(filename.c_str());
	if (!out) {
		cout << endl << "failed to save file - SaveLearnedData" << endl;//error in writing to file
		return;
	}

	for (int layer = 0; layer < Layersize.size() - 1; layer++) {//ouput weights
		for (int row = 0; row < Layersize[layer]; row++) {
			for (int col = 0; col < Layersize[layer + 1]; col++)
				out << nLayer[layer].Weights[row][col] << " ";
			out << endl;
		}
		out << endl;//blank line
	}

	// output bias.
	for (int layer = 1; layer < Layersize.size(); layer++) {
		for (int y = 0; y < Layersize[layer]; y++) {
			out << nLayer[layer].Bias[y] << "  ";
			out << endl << endl;
		}
		out << endl;
	}

	out.close();//data saved--close connection

	return;
}

/*
--SaveLearnedData--
Save the network weights and bias values which were able to achieve optimal results to file
*/
void NeuralNetwork::SaveLearnedDataMemory(Sizes Layersize, string filename, vector<double>* memorySnapShot)//save data to file
{
	int index = 0;

	for (int layer = 0; layer < Layersize.size() - 1; layer++) {//ouput weights
		for (int row = 0; row < Layersize[layer]; row++) {
			for (int col = 0; col < Layersize[layer + 1]; col++) {
				(*memorySnapShot)[index] = nLayer[layer].Weights[row][col];
				index++;
			}

		}

	}

	// output bias.
	for (int layer = 1; layer < Layersize.size(); layer++) {
		for (int y = 0; y < Layersize[layer]; y++) {
			(*memorySnapShot)[index] = nLayer[layer].Bias[y];
			index++;
		}

	}
	return;
}

/*
--LoadSavedData--
Load the network weights and bias values which were able to achieve optimal results from file
*/
void NeuralNetwork::LoadSavedDataMemory(Sizes Layersize, vector<double>* memorySnapShot)//load saved data from file
{
	int index = 0;

	for (int layer = 0; layer < Layersize.size() - 1; layer++)//read weights
		for (int row = 0; row < Layersize[layer]; row++)
			for (int col = 0; col < Layersize[layer + 1]; col++) {
				nLayer[layer].Weights[row][col] = (*memorySnapShot)[index];
				index++;
			}



	for (int layer = 1; layer < Layersize.size(); layer++)//read bias
		for (int y = 0; y < Layersize[layer]; y++)
		{
			nLayer[layer].Bias[y] = (*memorySnapShot)[index];
			index++;
		}


	//cout << endl << "data loaded for testing" << endl;//data read...close connection
	return;
}

void NeuralNetwork::SaveErrorToFile(Sizes Layersize, string filename, double error)//save data to file
{

	ofstream out;
	out.open(filename.c_str());
	if (!out) {
		cout << endl << "failed to save file - SaveErrorToFile" << endl;//error in writing to file
		return;
	}

	out << error << endl;

	out.close();//data saved--close connection

	return;
}

/*
--LoadSavedData--
Load the network weights and bias values which were able to achieve optimal results from file
*/
void NeuralNetwork::LoadSavedData(Sizes Layersize, string filename)//load saved data from file
{
	ifstream in(filename.c_str());
	if (!in) {
		cout << endl << "failed to save file" << endl;//error reading from file
		return;
	}

	for (int layer = 0; layer < Layersize.size() - 1; layer++)//read weights
		for (int row = 0; row < Layersize[layer]; row++)
			for (int col = 0; col < Layersize[layer + 1]; col++)
				in >> nLayer[layer].Weights[row][col];

	for (int layer = 1; layer < Layersize.size(); layer++)//read bias
		for (int y = 0; y < Layersize[layer]; y++)
			in >> nLayer[layer].Bias[y];

	in.close();
	//cout << endl << "data loaded for testing" << endl;//data read...close connection
	return;
}

/*
--CountTestingData--
Count no. of correctly predicted instances when working with the testing data
*/
double NeuralNetwork::CountTestingData(TrainingExamples TraineeSamples, int temp, Sizes Layersize)
{
	//variable declaration
	double count = 0;
	int total = TraineeSamples.Datapoints;
	double accepted = temp * 1;
	double desiredoutput;
	double actualoutput;
	double Error;
	int end = static_cast<int>(Layersize.size()) - 1;

	for (int pattern = 0; pattern < temp; pattern++) {
		//variable declaration
		Layer Desired;//to hold desired output from dataset
		Layer Actual;//to hold actual calculated output

		for (int i = 0; i < Layersize[end]; i++)
			Desired.push_back(0);//initiliaze with 0s
		for (int j = 0; j < Layersize[end]; j++)
			Actual.push_back(0);//intialize with 0s

		for (int output = 0; output < Layersize[end]; output++) {//go through all output neurons
			desiredoutput = TraineeSamples.OutputValues[pattern][output];//get desired output values
			actualoutput = Output[pattern][output];//get calculated output value

			Desired[output] = desiredoutput;//assign to desired output vector

			if ((actualoutput >= 0) && (actualoutput <= 0.5))
				actualoutput = 0;//if its between 0-0.5 then round it down to 0

			else if ((actualoutput <= 1) && (actualoutput >= 0.5))
				actualoutput = 1;//it its between 1 and 0.5 then round it up to 1

			Actual[output] = actualoutput;//store new actual output value
		}

		int confirm = 0;

		for (int b = 0; b < Layersize[end]; b++) {
			if (Desired[b] == Actual[b])//check if the actual and desired output match i.e if the prediction/classification was correct
				confirm++;
		}

		if (confirm == Layersize[end])
			count++;//if an instance is correctly predicted meaning all the values of the output layer neurons match the desired output then increase count

		confirm = 0;//reset for next set
	}

	return count;//return count of correctly predicted instances
}

/*
--CountLearningData--
Get Accuracy for no. of correctly predicted instances when testing with training set
*/
double NeuralNetwork::CountLearningData(TrainingExamples TraineeSamples, int temp, Sizes Layersize)
{
	//variable declaration
	double count = 0;
	int total = TraineeSamples.Datapoints;
	double accepted = temp * 1;
	double desiredoutput;
	double actualoutput;
	double Error;
	int end = static_cast<int>(Layersize.size()) - 1;
	double error_tolerance = 0.2;

	for (int pattern = 0; pattern < temp; pattern++) {
		Layer Desired;//to hold desired output from dataset
		Layer Actual;//to hold calculated output values

		for (int i = 0; i < Layersize[end]; i++)
			Desired.push_back(0);//initialize with 0s
		for (int j = 0; j < Layersize[end]; j++)
			Actual.push_back(0);//initialize with 0s

		for (int output = 0; output < Layersize[end]; output++) {
			desiredoutput = TraineeSamples.OutputValues[pattern][output];//get desired output values from training set
			actualoutput = Output[pattern][output];//get actual output values from calculated set

			Desired[output] = desiredoutput;

			//depending on the error tolerance threshold-->round up/down the actual output values
			if ((actualoutput >= 0) && (actualoutput <= (0 + error_tolerance)))
				actualoutput = 0;

			else if ((actualoutput <= 1) && (actualoutput >= (1 - error_tolerance)))
				actualoutput = 1;

			Actual[output] = actualoutput;//set new actual output values
		}

		int confirm = 0;

		for (int b = 0; b < Layersize[end]; b++) {
			if (Desired[b] == Actual[b])
				confirm++;//match
		}

		if (confirm == Layersize[end])
			count++;//if an instance is correctly predicted meaning all the values of the output layer neurons match the desired output then increase count

		confirm = 0;
	}

	return count;
}

/*
--CheckOutput--
To see if actual and desired output values match
*/
bool NeuralNetwork::CheckOutput(TrainingExamples TraineeSamples, int pattern, Sizes Layersize)
{
	//variable declaration
	int end = static_cast<int>(Layersize.size()) - 1;//know last layer
	double desiredoutput;
	double actualoutput;
	Layer Desired; //to hold desired output from dataset
	Layer Actual; //to hold actual calculated output

	for (int i = 0; i < Layersize[end]; i++)
		Desired.push_back(0);//initialize with 0s
	for (int j = 0; j < Layersize[end]; j++)
		Actual.push_back(0);//initialize with 0s

	int count = 0;

	for (int output = 0; output < Layersize[end]; output++) {//check for all neurons in output layer
		desiredoutput = TraineeSamples.OutputValues[pattern][output];//get desired output values from training set
		actualoutput = Output[pattern][output];//get actual calculated output
		Desired[output] = desiredoutput;
		cout << "desired : " << desiredoutput << "      " << actualoutput << endl;

		//round up or round down---if actual output is between 0-0.5[round down] else if it is between 0.5-1 then round down
		if ((actualoutput >= 0) && (actualoutput <= 0.5))
			actualoutput = 0;

		else if ((actualoutput <= 1) && (actualoutput >= 0.5))
			actualoutput = 1;

		Actual[output] = actualoutput;//new actual output value
	}

	cout << "---------------------" << endl;

	for (int b = 0; b < Layersize[end]; b++) {
		if (Desired[b] != Actual[b])//if the actual and intended output do not match then return false
			return false;
	}
	return true;
}

/*
--TestTrainingData--
Test the trained network with testing data
*/
double NeuralNetwork::TestTrainingData(Sizes Layersize, string filename, int  size, string load, int inputsize, int outputsize, ofstream & out2)
{
	//variable declaration
	bool valid;
	double count = 1;
	int total;
	double accuracy;

	int end = static_cast<int>(Layersize.size()) - 1;
	//load testing data
	TrainingExamples Test(load, size, inputsize + outputsize, inputsize, outputsize);
	//initialize network
	//CreateNetwork(Layersize, Test);
	//load saved training data
	LoadSavedData(Layersize, filename);
	//get testing values and do a forward pass
	for (int pattern = 0; pattern < size; pattern++) {
		ForwardPass(Test, pattern, Layersize);
	}
	//write to file the desired output, actual output and error
	for (int pattern = 0; pattern < size; pattern++) {
		for (int output = 0; output < Layersize[end]; output++) {
			out2 << Output[pattern][output] << " " << Test.OutputValues[pattern][output] << " " << fabs(fabs(Test.OutputValues[pattern][output]) - fabs(Output[pattern][output])) << endl;
		}
	}
	out2 << endl;
	out2 << endl;
	//output rmse and nmse to file
	accuracy = SumSquaredError(Test, Layersize);
	out2 << " RMSE:  " << accuracy << endl;
	//cout << "RMSE: " << accuracy << " %" << endl;
	NMSE = NormalisedMeanSquaredError(Test, Layersize);
	//out2 << " NMSE:  " << NMSE << endl;

	return accuracy;
}

/*
--TestLearnedData--
Test the network using the training data
*/
double NeuralNetwork::TestLearnedData(Sizes Layersize, string filename, int  size, string load, int inputsize, int outputsize)
{
	//variable declaration
	bool valid;
	double count = 1;
	int total;
	double accuracy;
	//get testing data set
	TrainingExamples Test(filename, size, inputsize + outputsize, inputsize, outputsize);

	total = static_cast<int>(Test.InputValues.size()); //how many samples to test?
													   //initialize network
													   //CreateNetwork(Layersize, Test);
													   //load saved network data
	LoadSavedData(Layersize, load);
	//do forward pass and calculate output
	for (int pattern = 0; pattern < total; pattern++) {
		ForwardPass(Test, pattern, Layersize);
	}
	//get number of correctly predicted instances
	count = CountTestingData(Test, size, Layersize);

	accuracy = (count / total) * 100; //get accuracy percentage
	cout << "The sucessful count is " << count << " out of " << total << endl;
	cout << "The accuracy of test is: " << accuracy << " %" << endl;
	return accuracy;
}

/*
--BackpropogationCICC--

Backpropagation function that takes in a network individual as parameter which is passed to network to be refined.
1. Pass Individual to network using previous winner decomposition method
2. Loop till a max Epoch (Island evolution time)
2.1 Perform Backpropagation
3. Return refined individual

*/

Layer NeuralNetwork::BackpropogationCICC(Layer ind, TrainingExamples TraineeSamples, double LearningRate, Sizes Layersize, string Savefile, bool load, int EpochMax, int Island)
{
	/// ---------------- Backpropagation Algorithm island procedure ---------------------- ///
	double SumErrorSquared;

	ChoromesToNeurons(ind, Island); //Populate network with individual

	int Epoch = 0; //Keeps track of epoch iteration count
	bool Learn = true; //Loop end variable

	while (Learn == true)
	{
		//---- FOR SGD
		for (int pattern = 0; pattern < TraineeSamples.InputValues.size(); pattern++)
		{//For all training examples perform forward and backward pass.
			ForwardPass(TraineeSamples, pattern, Layersize);
			BackwardPass(TraineeSamples, 0.1, pattern, Layersize); //Backward pass with learning rate set to 0.1
		}

		//---- FOR BGD
		//BackwardPassBGD(TraineeSamples, LearningRate, 0, Layersize);//do backward pass to update the weights

		Epoch++;

		//normalisedMeanSquaredError = NormalisedMeanSquaredError(TraineeSamples, Layersize); //Get current network error
		//cout << normalisedMeanSquaredError << " : is NormalisedMeanSquaredError" << endl;

		SumErrorSquared = SumSquaredError(TraineeSamples, Layersize); //Get current network error (RMSE)
		cout << SumErrorSquared << " : is RMSE" << endl;

		//ecout << SumErrorSquared << "\n";



		if (Epoch == EpochMax)
			Learn = false;
	}

	return  Neurons_to_chromes(Island); //Return refined individual
}

/*
--BackPropagation--
Main algorithm for neural network.
1. Do forward pass to calculate output
2. Do backward pass to adjust weights in order to reach desired output
3. Calculate error/difference between actual and desired output
4. Repeat 1 and 2 until desired accuracy is reached
*/
int NeuralNetwork::BackPropogationMemory(TrainingExamples TraineeSamples, double LearningRate, Sizes Layersize, string Savefile, bool load, double *CurrentBestGradient, int *CurrentBestGradientIndex, int CurrentIndex, double *CurrentBestError, int *CurrentBestErrorIndex, ofstream &bcout)
{

	//variable declaration
	double SumErrorSquared;
	int Id = 0;
	int Epoch = 0;
	bool Learn = true;
	//initialize network
	//CreateNetwork(Layersize, TraineeSamples);

	double Y2 = 0;
	double Y1 = 0;
	double X2 = 0;
	double X1 = 0;
	double CurrentGradient = 0;

	double minimumSumSquaredError = 99;
	int minimumErrorCounter = 1;

	//while learning not finished
	while (Learn == true) {
		//for all training data
		TraineeSamples.RandomizeDataOrder();
		//-- SGD
		for (int pattern = 0; pattern < TraineeSamples.InputValues.size(); pattern++)
		{
			ForwardPass(TraineeSamples, pattern, Layersize); //forward pass through the network to calculate output values

			BackwardPass(TraineeSamples, LearningRate, pattern, Layersize);//do backward pass to update the weights
		}



		//-- BGD

		//BackwardPassBGD(TraineeSamples, LearningRate, 0, Layersize);//do backward pass to update the weights

		//increase number of iterations
		Epoch++;
		cout << Epoch << " : is Epoch    *********************    " << endl;
		//error calculation
		SumErrorSquared = SumSquaredError(TraineeSamples, Layersize);
		cout << SumErrorSquared << " : is SumErrorSquared" << endl;

		//ecout << SumErrorSquared << "\n";
		//SaveErrorToFile(Layersize, file, SumErrorSquared);
		//save network weights
		SaveLearnedData(Layersize, Savefile);
		//double trained = 0;
		////train for 98% accuracy
		//if (trained >= 98) {
		//	Learn = false;
		//}
		//or 4000 iterations


		//Keep track of global minimum
		if (SumErrorSquared < *CurrentBestError) {
			*CurrentBestError = SumErrorSquared;
			*CurrentBestErrorIndex = CurrentIndex;
			minimumErrorCounter = 1;


			SaveLearnedData(Layersize, "CurrentMinimumWeights.txt");

			ofstream bestError;
			bestError.open("CurrentMinimumTestError.txt");


			bestError << *CurrentBestError << endl;

			bestError.close();

			//Should I save this solution
		}
		else {
			//Check if this is the 100th epoch where the minimum error has not changed. Stuck in minima return
			if (minimumErrorCounter == 100)
			{
				bcout << "Quit Run - Local Minimum = " << SumErrorSquared << " Global Minimum = " << *CurrentBestError << endl;

				Learn = false;
			}
			else {
				minimumErrorCounter++;
			}

		}



		if (Epoch == maxgenLS)
			Learn = false;

		//########### Calculate Gradient #############//
		//Get starting point - run BP for one pass and get error

		//|(Y2-Y1)/(X2-X1)|

		//Only start from epoch 2 else you risk divide by 0 error where X2 = 1 and X1 = 1
		//X1 and Y1 are the base point to reference

		//X1 = 1; //First point is always 1 epoch
		//X2 = Epoch;

		//if (Epoch == 2)
		//{
		//	Y1 = SumSquaredError(TraineeSamples, Layersize);

		//}
		//else if (Epoch == maxgenLS)
		//{
		//	Y2 = SumSquaredError(TraineeSamples, Layersize);
		//	CurrentGradient = abs((Y2 - Y1) / (X2 - X1));

		//	if (CurrentGradient < *CurrentBestGradient)
		//	{
		//		*CurrentBestGradient = CurrentGradient;
		//		*CurrentBestIndex = CurrentIndex;
		//	}
		//	else {
		//		return Epoch; //break this solution may not be promising
		//	}
		//}





	}

	//system("pause");

	//return number of iterations
	return Epoch;
}

/*
--BackPropagation--
Main algorithm for neural network.
1. Do forward pass to calculate output
2. Do backward pass to adjust weights in order to reach desired output
3. Calculate error/difference between actual and desired output
4. Repeat 1 and 2 until desired accuracy is reached
*/

int NeuralNetwork::BackPropogation(TrainingExamples TraineeSamples, double LearningRate, Sizes Layersize, string Savefile, bool load)
{

	//variable declaration
	double previous = 0;
	double SumErrorSquared;
	int repeatcount = 0;

	int Id = 0;
	int Epoch = 0;
	bool Learn = true;
	//initialize network
	//CreateNetwork(Layersize, TraineeSamples);double

	//while learning not finished
	while (Learn == true) {
		//for all training data
		TraineeSamples.RandomizeDataOrder();
		//-- SGD
		for (int pattern = 0; pattern < TraineeSamples.InputValues.size(); pattern++)
		{
			ForwardPass(TraineeSamples, pattern, Layersize); //forward pass through the network to calculate output values

			BackwardPass(TraineeSamples, LearningRate, pattern, Layersize);//do backward pass to update the weights
		}

		//-- BGD

		//BackwardPassBGD(TraineeSamples, LearningRate, 0, Layersize);//do backward pass to update the weights

		//increase number of iterations
		Epoch++;
		cout << Epoch << " : is Epoch    *********************    " << endl;
		//error calculation
		SumErrorSquared = SumSquaredError(TraineeSamples, Layersize);
		cout << SumErrorSquared << " : is SumErrorSquared" << endl;

		//ecout << SumErrorSquared << "\n";
		//SaveErrorToFile(Layersize, file, SumErrorSquared);
		//save network weights
		SaveLearnedData(Layersize, Savefile);
		double trained = 0;
		//train for 98% accuracy
		/*if (trained >= 98) {
		Learn = false;
		}*/
		//or 4000 iterations
		/*if (Epoch == maxgenLS)
		Learn = false;*/
		if (Epoch > 1)
			Learn = false;

		if (SumErrorSquared > previous || SumErrorSquared == previous) {
			repeatcount = repeatcount + 1;

			if (repeatcount > 10)
			{
				Learn = false;

			}
		}
		else
		{
			repeatcount = 0; // reset repeat count
			previous = SumErrorSquared;

		}


	}

	//system("pause");

	//return number of iterations
	return Epoch;
}
int NeuralNetwork::BackPropogationExhausted(TrainingExamples TraineeSamples, double LearningRate, Sizes Layersize, string Savefile, bool load)
{

	//variable declaration
	double previous = 99;
	double SumErrorSquared;
	int repeatcount = 0;

	int Id = 0;
	int Epoch = 0;
	bool Learn = true;
	//initialize network
	//CreateNetwork(Layersize, TraineeSamples);double

	//while learning not finished
	while (repeatcount < 100) {
		//for all training data
		TraineeSamples.RandomizeDataOrder();
		//-- SGD
		for (int pattern = 0; pattern < TraineeSamples.InputValues.size(); pattern++)
		{
			ForwardPass(TraineeSamples, pattern, Layersize); //forward pass through the network to calculate output values

			BackwardPass(TraineeSamples, LearningRate, pattern, Layersize);//do backward pass to update the weights
		}

		//-- BGD

		//BackwardPassBGD(TraineeSamples, LearningRate, 0, Layersize);//do backward pass to update the weights

		//increase number of iterations
		Epoch++;
		cout << Epoch << " : is Epoch    *********************    " << endl;
		//error calculation
		SumErrorSquared = SumSquaredError(TraineeSamples, Layersize);
		cout << SumErrorSquared << " : is SumErrorSquared" << endl;

		//ecout << SumErrorSquared << "\n";
		//SaveErrorToFile(Layersize, file, SumErrorSquared);
		//save network weights
		SaveLearnedData(Layersize, Savefile);
		double trained = 0;
		//train for 98% accuracy
		/*if (trained >= 98) {
		Learn = false;
		}*/
		//or 4000 iterations
		/*if (Epoch == maxgenLS)
		Learn = false;*/
		//		if (Epoch > maxgen)
		//			Learn = false;

		if (SumErrorSquared > previous || SumErrorSquared == previous) {
			repeatcount = repeatcount + 1;
			cout << "\nRepeat Count: " << repeatcount << endl;
			//			if (repeatcount > 10)
			//			{
			//				
			//			
			//			}
		}
		else
		{
			repeatcount = 0; // reset repeat count
			previous = SumErrorSquared;

		}


	}

	//system("pause");

	//return number of iterations
	return Epoch;
}


TwoDVector_Double NeuralNetwork::BackPropogationCC(TrainingExamples TraineeSamples, double LearningRate, Sizes Layersize, string Savefile, bool load, int epochs, int individualsize)
{
	VectorManipulation vectormanipulate;
	//TwoDVector_Double savedsamples = vectormanipulate.Generate2DVector_Double(epochs, individualsize, 0);
	TwoDVector_Double savedsamples;

	//variable declaration
	double previous = 0;
	double SumErrorSquared;
	int repeatcount = 0;

	int Id = 0;
	int Epochs = 0;
	bool Learn = true;

	double currentbest = 99;

	int accepted = 0;

	while (accepted < epochs)
		//for (int u = 0; u < epochs; u++)
	{
		TraineeSamples.RandomizeDataOrder();
		for (int pattern = 0; pattern < TraineeSamples.InputValues.size(); pattern++)
		{
			ForwardPass(TraineeSamples, pattern, Layersize); //forward pass through the network to calculate output values

			BackwardPass(TraineeSamples, LearningRate, pattern, Layersize);//do backward pass to update the weights
		}

		Epochs++;
		SumErrorSquared = SumSquaredError(TraineeSamples, Layersize);
		cout << "\nEpoch: " << Epochs << " Error: " << SumErrorSquared;



		if (SumErrorSquared > currentbest || SumErrorSquared == currentbest)
		{
			repeatcount++;
			cout << "\nBP repeat count: " << repeatcount;

			if (repeatcount == 100)
			{
				break;
			}
		}
		else {

			repeatcount = 0;
			currentbest = SumErrorSquared;

			savedsamples.push_back(Neurons_to_chromes(1)); //only if accepted add to samples
			accepted++;
		}
	}

	return savedsamples;
}

/*
--Neurons_to_chromes--
Convert neurons to chromes which can be processed by the genetic algorithm
Receives Island parameter which specifies which decomposition method to follow when
extracting current network individual
*/

Layer NeuralNetwork::Neurons_to_chromes(int Island)
{
	int gene = 0;
	Layer NeuronChrome(StringSize);

	int layer = 0;

	if (Island == 1) //Neuron level
	{
		for (int neu = 0; neu < layersize[1]; neu++)
		{
			for (int row = 0; row < layersize[layer]; row++)
			{//Convert weights to chromes for hidden layer
				NeuronChrome[gene] = nLayer[layer].Weights[row][neu];
				gene++;
			}

			//Convert bias to chromes for hidden layer
			NeuronChrome[gene] = nLayer[layer + 1].Bias[neu];
			gene++;
		}

		layer = 1;

		for (int neu = 0; neu < layersize[2]; neu++)
		{
			for (int row = 0; row < layersize[layer]; row++)
			{//Convert weights to chromes for output layer
				NeuronChrome[gene] = nLayer[layer].Weights[row][neu];
				gene++;
			}

			//Convert bias to chromes for output layer
			//Mon 10-Apr-17 03:11 PM, [Modified By: Gary] [Error here ]
			NeuronChrome[gene] = nLayer[layer + 1].Bias[neu];
			gene++;
		}
	}

	gene = 0;

	if (Island == 2) //Network Level (Backpropagation)
	{
		for (int layer = 1; layer < layersize.size(); layer++) // For all 3 layers
		{
			for (row = 0; row < layersize[layer]; row++) // For current layer
			{
				NeuronChrome[gene] = nLayer[layer].Bias[row]; // Save bias

				gene++;
			}
		}

		for (int layer = 0; layer < layersize.size() - 1; layer++) // For input and hidden layers
		{
			for (row = 0; row < layersize[layer]; row++) //For current layer
			{
				for (col = 0; col < layersize[layer + 1]; col++)
				{//Convert weights to chromes for current layer
					NeuronChrome[gene] = nLayer[layer].Weights[row][col];
					gene++;
				}
			}
		}

		cout << "\n";
	}

	return   NeuronChrome;
}

/*
--ChromesToNeurons--
Convert the chromes back to the neural network structure
*/

/*void NeuralNetwork:: ChoromesToNeurons(  Layer NeuronChrome)
{
StringSize = NeuronChrome.size();
int layer = 0;
int gene = 0;
#ifdef neuronlevel
for(int neu = 0; neu < layersize[1]; neu++ ){
for( int row = 0; row<  layersize[layer] ; row++){
nLayer[layer].Weights[row][neu]=  NeuronChrome[gene]  ;//convert genes back to neurons and weights for hidden layer
gene++;
}

nLayer[layer+1 ].Bias[neu] =   NeuronChrome[gene]  ;//convert genes back to bias for hidden layer
gene++;
}

layer = 1;

for(int neu = 0; neu < layersize[2]; neu++ ){
for( int row = 0; row<  layersize[layer] ; row++){
nLayer[layer].Weights[row][neu]=  NeuronChrome[gene]  ; //convert genes back to neurons and weights for output layer
gene++;
}

nLayer[layer+1 ].Bias[neu] =   NeuronChrome[gene]  ;//convert genes back to bais for output layer
gene++;
}

#endif

#ifdef hiddenNeuron
for(int neu = 0; neu < layersize[1]; neu++ ){
for( int row = 0; row<  layersize[layer] ; row++){
nLayer[layer].Weights[row][neu]=  NeuronChrome[gene]  ; //convert genes back to neurons and weights for hidden layer
gene++;
}

nLayer[layer+1 ].Bias[neu] =   NeuronChrome[gene]  ;
gene++;

for( int row = 0; row<  layersize[layer+2] ; row++){
nLayer[layer+1].Weights[neu][row]=  NeuronChrome[gene]  ; //convert genes back to bais for hideen layer
gene++;
}
}

#endif

#ifdef weightlevel

for(int layer=0; layer <  layersize.size()-1; layer++){
for( row = 0; row<  layersize[layer] ; row++){
for( col = 0; col <  layersize[layer+1]; col++)  {
nLayer[layer].Weights[row][col]=  NeuronChrome[gene]  ; //convert genes back to neurons and weights for all layers
gene++;
}
}
}

for(int layer=0; layer <  layersize.size() ; layer++){
for( row = 0; row <  layersize[layer] ; row ++) {
nLayer[layer ].Bias[row] =   NeuronChrome[gene]  ; //convert genes back to bias for all layers
gene++;
}
}
#endif

#ifdef networklevel

for(int layer=0; layer <  layersize.size()-1; layer++){
for( row = 0; row<  layersize[layer] ; row++){
for( col = 0; col <  layersize[layer+1]; col++)  {
nLayer[layer].Weights[row][col]=  NeuronChrome[gene]  ; //convert genes back to neurons and weights for all layers
gene++;
}
}
}

for(int layer=0; layer <  layersize.size() ; layer++){
for( row = 0; row <  layersize[layer] ; row ++) {
nLayer[layer ].Bias[row] =   NeuronChrome[gene]  ; //convert genes back to bias for all layers
gene++;
}
}
#endif
}*/

/*
--ChromesToNeurons--
Convert the chromes back to the neural network structure.
Receives Island parameter which specifies which decomposition method to follow when
passing individual to network.
*/

void NeuralNetwork::ChoromesToNeurons(Layer NeuronChrome, int Island)
{
	StringSize = static_cast<int>(NeuronChrome.size());
	int layer = 0;
	int gene = 0;

	if (Island == 1) //Neuron level
	{
		for (int neu = 0; neu < layersize[1]; neu++) {
			for (int row = 0; row < layersize[layer]; row++)
			{
				nLayer[layer].Weights[row][neu] = NeuronChrome[gene];
				gene++;
			}

			nLayer[layer + 1].Bias[neu] = NeuronChrome[gene];
			gene++;
		}

		layer = 1;

		for (int neu = 0; neu < layersize[2]; neu++)
		{
			for (int row = 0; row < layersize[layer]; row++)
			{
				nLayer[layer].Weights[row][neu] = NeuronChrome[gene];
				gene++;
			}

			nLayer[layer + 1].Bias[neu] = NeuronChrome[gene];
			gene++;
		}
	}
	gene = 0;

	if (Island == 2) //Network Level (Backpropagation)
	{
		for (int layer = 1; layer < layersize.size(); layer++) {
			for (row = 0; row < layersize[layer]; row++) {
				nLayer[layer].Bias[row] = NeuronChrome[gene];
				gene++;
			}
		}
		for (int layer = 0; layer < layersize.size() - 1; layer++) {
			for (row = 0; row < layersize[layer]; row++) {
				for (col = 0; col < layersize[layer + 1]; col++) {
					nLayer[layer].Weights[row][col] = NeuronChrome[gene];
					gene++;
				}
			}
		}
	}

#ifdef hiddenNeuron
	for (int neu = 0; neu < layersize[1]; neu++) {
		for (int row = 0; row < layersize[layer]; row++) {
			nLayer[layer].Weights[row][neu] = NeuronChrome[gene];
			gene++;
		}

		nLayer[layer + 1].Bias[neu] = NeuronChrome[gene];
		gene++;

		for (int row = 0; row < layersize[layer + 2]; row++) {
			nLayer[layer + 1].Weights[neu][row] = NeuronChrome[gene];
			gene++;
		}
	}

#endif

#ifdef weightlevel

	for (int layer = 0; layer < layersize.size() - 1; layer++) {
		for (row = 0; row < layersize[layer]; row++) {
			for (col = 0; col < layersize[layer + 1]; col++) {
				nLayer[layer].Weights[row][col] = NeuronChrome[gene];
				gene++;
			}
		}
	}
	//---------------------------------------------------------

	//-----------------------------------------------
	for (int layer = 0; layer < layersize.size(); layer++) {
		for (row = 0; row < layersize[layer]; row++) {
			nLayer[layer].Bias[row] = NeuronChrome[gene];
			gene++;
		}
	}
#endif
}

/*
--ForwardFitnessPass--
Calculate the fitness of the network with the test data
*/

double NeuralNetwork::ForwardFitnessPass(Layer NeuronChrome, TrainingExamples Test, int Island)
{
	//Convert the chromes back to neurons
	ChoromesToNeurons(NeuronChrome, Island);

	NumEval++; //number of function evaluations

	double SumErrorSquared = 0;
	bool Learn = true;

	for (int pattern = 0; pattern < Test.InputValues.size(); pattern++)
	{
		ForwardPass(Test, pattern, layersize); //Do forward pass on the input values and calculate output for the test data
	}

	SumErrorSquared = SumSquaredError(Test, layersize); //Calculate sum squared error from the output on the test data

	if (Learn == false)
		return -1;

	return  SumErrorSquared; //return error
}

/*
--Individual Class--
Represents a particular chromozone
*/
class Individual {
public: //class variables
	Layer Chrome;
	double Fitness;
	Layer BitChrome;
	OneDVector_Double Values;


public://class functions
	Individual()
	{
		VectorManipulation manipulate;
		Values = manipulate.Generate1DVector_Double(2, 0);
		//Index 0 = Likelihood
		//Index 1 = PLikelihood
	}
	void print();
};
typedef vector<double> Nodes;

/*
--Genetic Algorithm--
Used to evolve a population in order to achieve a set which has an acceptable fitness level
*/
class GeneticAlgorithmn
{
public://class variables
	int PopSize;
	vector<Individual> Population;
	Sizes TempIndex;
	vector<Individual> NewPop;
	Sizes mom;
	Sizes list;
	int MaxGen;
	int NumVariable;
	double BestFit;
	double WorstFit;
	int BestIndex;
	int WorstIndex;
	int NumEval;
	int  kids;

	double BestLikelihood;
	double BestPriorLikelihood;

	int BestIndexLikelihood;
	int BestIndexPriorLikelihood;




public: //class functions
	GeneticAlgorithmn(int stringSize)
	{
		NumVariable = stringSize;
		NumEval = 0;
		BestIndex = 0;
		WorstIndex = 0;
	}

	GeneticAlgorithmn()
	{
		BestIndex = 0;
		WorstIndex = 0;
	}

	double Fitness() { return BestFit; }

	double WorstFitness() { return WorstFit; }

	double  RandomWeights();

	double  RandomAddition();

	void PrintPopulation();

	double rand_normal(double mean, double stddev);

	int GenerateNewPCX(int pass, NeuralNetwork network, TrainingExamples Sample, double Mutation, int depth);

	double Objective(Layer x);

	void  InitilisePopulation(int popsize);

	void Evaluate();

	//Mon 10-Apr-17 10:29 AM, [Modified By: Gary] [Changed [] to vector]
	double  modu(vector<double> index);
	//Mon 10-Apr-17 10:29 AM, [Modified By: Gary] [Changed [] to vector]
	double  innerprod(vector<double> Ind1, vector<double> Ind2);

	void RandomParents();

	void  my_family();   //here a random family (1 or 2) of parents is created who would be replaced by good individuals

	void  find_parents();

	void  rep_parents();  //here the best (1 or 2) individuals replace the family of parents

	void sort();
};

/*
--RandomWeights--
Generate random number  between -5 and 5(double)
*/
//Mon 10-Apr-17 03:30 PM, [Modified By: Gary] [removed other if]
double GeneticAlgorithmn::RandomWeights()
{
	int chance;
	double randomWeight;
	double NegativeWeight;
	chance = rand() % 2;

	if (chance == 0) {
		randomWeight = rand() % 100000;
		return randomWeight*0.00005;
	}
	else
	{
		NegativeWeight = rand() % 100000;
		return NegativeWeight*-0.00005;
	}

	//if (chance == 0) {
	//	randomWeight = rand() % 100000;
	//	return randomWeight*0.00005;
	//}
	//else
	//if (chance == 1) {
	//	NegativeWeight = rand() % 100000;
	//	return NegativeWeight*-0.00005;
	//}
}

/*
--rand_normal--
Box Muller Random Numbers
*/
double GeneticAlgorithmn::rand_normal(double mean, double stddev) {
	static double n2 = 0.0;
	static int n2_cached = 0;

	if (!n2_cached) {
		// choose a point x,y in the unit circle uniformly at random
		double x, y, r;
		do {
			//  scale two random integers to doubles between -1 and 1
			x = 2.0*rand() / RAND_MAX - 1;
			y = 2.0*rand() / RAND_MAX - 1;

			r = x*x + y*y;
		} while (r == 0.0 || r > 1.0);

		// Apply Box-Muller transform on x, y
		double d = sqrt(-2.0*log(r) / r);
		double n1 = x*d;
		n2 = y*d;

		// scale and translate to get desired mean and standard deviation
		double result = n1*stddev + mean;

		n2_cached = 1;
		return result;
	}
	else {
		n2_cached = 0;
		return n2*stddev + mean;
	}
}

/*
--RandomAddition--
Generate random number  between -0.9 and 0.9 (type double)
*/

double GeneticAlgorithmn::RandomAddition()
{
	int chance;
	double randomWeight;
	double NegativeWeight;
	chance = rand() % 2;

	if (chance == 0) {
		randomWeight = rand() % 100;
		return randomWeight*0.009;
	}

	else {
		NegativeWeight = rand() % 100;
		return NegativeWeight*-0.009;
	}
}

/*
--InitializsePopulation--
Intialize the vectors used in algorithm,
*/
void GeneticAlgorithmn::InitilisePopulation(int popsize)
{
	//variable declaration
	double x, y;
	Individual Indi;
	NumEval = 0;
	BestIndex = 0;
	WorstIndex = 0;
	PopSize = popsize;

	for (int i = 0; i < PopSize; i++) {
		TempIndex.push_back(0); //initialize with 0s
		mom.push_back(0); //initialize female parent
	}

	for (int i = 0; i < NPSize; i++) {
		list.push_back(0); //initialize with 0s
	}

	for (int row = 0; row < PopSize; row++)
		Population.push_back(Indi); //population made up of individuals

	for (int row = 0; row < PopSize; row++) {
		for (int col = 0; col < NumVariable; col++) {
			Population[row].Chrome.push_back(RandomWeights());//initialize population with random weights
		}
	}

	for (int row = 0; row < NPSize; row++)
		NewPop.push_back(Indi);//new population made up of individuals

	for (int row = 0; row < NPSize; row++) {
		for (int col = 0; col < NumVariable; col++)
			NewPop[row].Chrome.push_back(0);//initialise new population with 0s
	}
}

/*
--Evaluate--
Calculates the fitness of all chromozones
*/
void GeneticAlgorithmn::Evaluate()
{
	// solutions are evaluated and best id is computed

	Population[0].Fitness = Objective(Population[0].Chrome); //calculate fitness by using objective function
	BestFit = Population[0].Fitness;//initially best fitness is the fitness of the first chromozone
	BestIndex = 0;

	WorstFit = Population[0].Fitness;//initially best fitness is the fitness of the first chromozone
	WorstIndex = 0;


	for (int row = 0; row < PopSize; row++)
	{
		Population[row].Fitness = Objective(Population[row].Chrome);//evaluate fitness of chromes

		if ((MINIMIZE * BestFit) > (MINIMIZE * Population[row].Fitness))
		{
			BestFit = Population[row].Fitness; //update best fitness
			BestIndex = row;
		}

		if ((MINIMIZE * WorstFit) < (MINIMIZE * Population[row].Fitness))
		{
			WorstFit = Population[row].Fitness; //update best fitness
			WorstIndex = row;
		}
	}
}

/*
--Print--
Output fitness and chromozone values
*/
void GeneticAlgorithmn::PrintPopulation()
{
	cout << "output individuals and its chromes" << endl;
	for (int row = 0; row < PopSize / 5; row++) {
		for (int col = 0; col < NumVariable; col++)
			cout << Population[row].Chrome[col] << " "; //output individuals and its chromes

		cout << endl;
	}
	cout << "output fitness" << endl;
	for (int row = 0; row < PopSize / 5; row++)
		cout << Population[row].Fitness << endl;//output fitness

	cout << " best fitness---" << endl;
	cout << BestFit << "  " << BestIndex << endl;//best fitness
}

/*
--Objective--
Used to calculate the fitness of the chromozones. Contains 3 functions which can be used based on their need
*/
double GeneticAlgorithmn::Objective(Layer x)
{
	int i, j, k;
	double fit, sumSCH;

	fit = 0.0;

#ifdef ellip
	// Ellipsoidal function
	for (j = 0; j < NumVariable; j++)
		fit += ((j + 1)*(x[j] * x[j]));
#endif

#ifdef schwefel
	// Schwefel's function
	for (j = 0; j < NumVariable; j++)
	{
		for (i = 0, sumSCH = 0.0; i < j; i++)
			sumSCH += x[i];
		fit += sumSCH * sumSCH;
	}
#endif

#ifdef rosen
	//  Rosenbrock's function
	for (j = 0; j < NumVariable - 1; j++)
		fit += 100.0*(x[j] * x[j] - x[j + 1])*(x[j] * x[j] - x[j + 1]) + (x[j] - 1.0)*(x[j] - 1.0);

	//NumEval++;
#endif

	return(fit);
}

/*
--my_family--
Here a random family (1 or 2) of parents is created who would be replaced by good individuals
*/
void GeneticAlgorithmn::my_family()
{
	int i, j, index; //variables
	int swp;
	double u;

	for (i = 0; i < PopSize; i++)
		mom[i] = i;

	for (i = 0; i < family; i++)
	{
		index = (rand() % PopSize) + i; //randomly get index value which corresponds to an individual from the population

		if (index > (PopSize - 1)) index = PopSize - 1; //make sure index is not outside population range
		swp = mom[index];
		mom[index] = mom[i];
		mom[i] = swp; //swap values
	}
}

/*
--find_parents--
Here the parents to be replaced are added to the temporary sub-population to assess their goodness
against the new solutions formed which will be the basis of whether they should be kept or not
*/
void GeneticAlgorithmn::find_parents()
{
	int i, j, k;
	double u, v;//variables

	my_family();
	for (j = 0; j < family; j++)//go through entire family
	{
		for (i = 0; i < NumVariable; i++)
			NewPop[kids + j].Chrome[i] = Population[mom[j]].Chrome[i];//add parents to new sub-pop

		NewPop[kids + j].Fitness = Objective(NewPop[kids + j].Chrome);//store fitness of parent
	}
}

/*
--rep_parents--
Here the best (1 or 2) individuals replace the family of parents
*/
void GeneticAlgorithmn::rep_parents()
{
	int i, j;
	for (j = 0; j < family; j++)
	{
		for (i = 0; i < NumVariable; i++)
			Population[mom[j]].Chrome[i] = NewPop[list[j]].Chrome[i]; //new child added to population

		Population[mom[j]].Fitness = Objective(Population[mom[j]].Chrome);//update fitness
	}
}

/*
--sort--
Arrange individuals in order of population
*/
void GeneticAlgorithmn::sort()
{
	int i, j, temp;
	double dbest;

	for (i = 0; i < (kids + family); i++) list[i] = i;

	if (MINIMIZE)
		for (i = 0; i < (kids + family - 1); i++)
		{
			dbest = NewPop[list[i]].Fitness;
			for (j = i + 1; j < (kids + family); j++)
			{
				if (NewPop[list[j]].Fitness < dbest)
				{
					dbest = NewPop[list[j]].Fitness;//get the best fitness
					temp = list[j];
					list[j] = list[i];
					list[i] = temp;
				}
			}
		}
	else
		for (i = 0; i < (kids + family - 1); i++)
		{
			dbest = NewPop[list[i]].Fitness;
			for (j = i + 1; j < (kids + family); j++)
			{
				if (NewPop[list[j]].Fitness > dbest)
				{
					dbest = NewPop[list[j]].Fitness;//get the best fitness
					temp = list[j];
					list[j] = list[i];
					list[i] = temp;
				}
			}
		}
}

/*
--modu--
Calculate modulus
*/

//Mon 10-Apr-17 10:28 AM, [Modified By: Gary] [changed index[] to vector<double> index]
double GeneticAlgorithmn::modu(vector<double> index)
{
	//variables
	int i;
	double sum, modul;
	sum = 0.0;

	for (i = 0; i < NumVariable; i++)
		sum += (index[i] * index[i]);

	modul = sqrt(sum);

	return modul;
}

/*
--innerprod--
calculates the inner product of two vectors
*/

//Mon 10-Apr-17 10:28 AM, [Modified By: Gary] [changed [] to vector<double> index]
double GeneticAlgorithmn::innerprod(vector<double> Ind1, vector<double> Ind2)
{
	//variables
	int i;
	double sum;
	sum = 0.0;

	for (i = 0; i < NumVariable; i++)
		sum += (Ind1[i] * Ind2[i]);

	return sum;//return inner_product
}


/*
--GenerateNewPCX--
Generate new population
*/
int GeneticAlgorithmn::GenerateNewPCX(int pass, NeuralNetwork network, TrainingExamples Sample, double Mutation, int depth)
{
	//variable declaration
	int i, j, num, k;
	vector<double> Centroid(NumVariable);//double Centroid[NumVariable];
	double tempvar, tempsum, D_not, dist;
	vector<double> tempar1(NumVariable); //double tempar1[NumVariable];
	vector<double> tempar2(NumVariable); //double tempar2[NumVariable];
	double D[RandParent];
	vector<double> d(NumVariable);//double d[NumVariable];
	vector<vector<double> > diff;   //double diff[RandParent][NumVariable];

									//Mon 10-Apr-17 10:25 AM, [Modified By: Gary] [initialize diff 2 dimensional array]
	diff.resize(RandParent, vector<double>(NumVariable, 0));

	double temp1, temp2, temp3;
	int temp;

	for (i = 0; i < NumVariable; i++)
		Centroid[i] = 0.0;//initialize centroid with 0

						  // centroid is calculated here
	for (i = 0; i < NumVariable; i++)
	{
		for (j = 0; j < RandParent; j++)
			Centroid[i] += Population[TempIndex[j]].Chrome[i];

		Centroid[i] /= RandParent;
	}

	// calculate the distace (d) from centroid to the index parent arr1[0]
	// also distance (diff) between index and other parents are computed
	for (j = 1; j < RandParent; j++)
	{
		for (i = 0; i < NumVariable; i++)
		{
			if (j == 1)
				d[i] = Centroid[i] - Population[TempIndex[0]].Chrome[i];
			diff[j][i] = Population[TempIndex[j]].Chrome[i] - Population[TempIndex[0]].Chrome[i];

			if (isnan(diff[j][i]))
			{
				cout << "`diff nan   " << endl;
				diff[j][i] = 1;
				return (0);
			}

		}

		if (modu(diff[j]) < EPSILON)
		{
			cout << "RUN Points are very close to each other. Quitting this run   " << endl;

			return (0);
		}

		/*if (isnan(diff[j][i]))
		{
		cout << "`diff nan   " << endl;
		diff[j][i] = 1;
		return (0);
		}*/
	}
	dist = modu(d); // modu calculates the magnitude of the vector

	if (dist < EPSILON)
	{
		cout << "RUN Points are very close to each other. Quitting this run    " << endl;

		return (0);
	}

	// orthogonal directions are computed (see the paper)
	for (i = 1; i < RandParent; i++)
	{
		temp1 = innerprod(diff[i], d);
		if ((modu(diff[i])*dist) == 0) {
			cout << " division by zero: part 1" << endl;
			temp2 = temp1 / (1);
		}
		else {
			temp2 = temp1 / (modu(diff[i])*dist);
		}
		temp3 = 1.0 - pow(temp2, 2.0);
		D[i] = modu(diff[i])*sqrt(temp3);
	}

	D_not = 0;
	for (i = 1; i < RandParent; i++)
		D_not += D[i];

	D_not /= (RandParent - 1); //this is the average of the perpendicular distances from all other parents (minus the index parent) to the index vector

							   // Next few steps compute the child, by starting with a random vector
	for (j = 0; j < NumVariable; j++)
	{
		tempar1[j] = rand_normal(0, D_not*sigma_eta);
		tempar2[j] = tempar1[j];
	}

	for (j = 0; j < NumVariable; j++)
	{
		if (pow(dist, 2.0) == 0) {
			cout << " division by zero: part 2" << endl;
			tempar2[j] = tempar1[j] - ((innerprod(tempar1, d)*d[j]) / 1);
		}
		else
			tempar2[j] = tempar1[j] - ((innerprod(tempar1, d)*d[j]) / pow(dist, 2.0));
	}

	for (j = 0; j < NumVariable; j++)
		tempar1[j] = tempar2[j];

	for (k = 0; k < NumVariable; k++)
		NewPop[pass].Chrome[k] = Population[TempIndex[0]].Chrome[k] + tempar1[k];

	tempvar = rand_normal(0, sigma_zeta);

	for (k = 0; k < NumVariable; k++) {
		NewPop[pass].Chrome[k] += (tempvar*d[k]);
	}

	double random = rand() % 10;

	Layer Chrome(NumVariable);

	for (k = 0; k < NumVariable; k++) {
		if (!isnan(NewPop[pass].Chrome[k])) {
			Chrome[k] = NewPop[pass].Chrome[k];
		}
		else
			NewPop[pass].Chrome[k] = RandomAddition();
	}

	return (1);
}

/*
--RandomParents--
Select two parents. First one will be the one with the best fitness within the population and the second one is randomly selected
*/
void GeneticAlgorithmn::RandomParents()
{
	//variable declaration
	int i, j, index;
	int swp;
	double u;
	int delta;

	for (i = 0; i < PopSize; i++)
		TempIndex[i] = i;

	swp = TempIndex[0];
	TempIndex[0] = TempIndex[BestIndex];  // best is always included as a parent and is the index parent
										  // this can be changed for solving a generic problem
	TempIndex[BestIndex] = swp;

	for (i = 1; i < RandParent; i++)  // shuffle the other parents i.e randomly select the other parent
	{
		index = (rand() % PopSize) + i;//randomly generate index

		if (index > (PopSize - 1)) index = PopSize - 1;//ensure its within the correct range

		swp = TempIndex[index];
		TempIndex[index] = TempIndex[i];
		TempIndex[i] = swp;//swap
	}
}

void TIC(void)
{
	TicTime = time(NULL);
}

void TOC(void)
{
	TocTime = time(NULL);
}

double StopwatchTimeInSeconds()
{
	return difftime(TocTime, TicTime);
}

typedef vector<GeneticAlgorithmn> GAvector;
/*
--Table--
*/
class Table {
public:
	Layer   SingleSp;
};

typedef vector<Table> TableVector;

class LikelihoodReturnObject
{
public:
	double likelihood;
	OneDVector_Double pred_train;
	double rmse;

};
class MCMC : public GeneticAlgorithmn
{
public:
	OneDVector_Double GetLikelihoodForSubPop(OneDVector_Double parent, int h)
	{
		VectorManipulation vectormanipulate;

		clock_t start = clock();

		int hidden = static_cast<int>(h);
		int weightsize1 = (input*hidden);//number of weights between hidden and input layer
		int weightsize2 = (hidden*output);//number of weights between hidden and output layer
		int biasize = hidden + output; //bias for hidden and output layer

		TrainingExamples SamplesTrain(trainfile, trainsize, input + output, input, output); //get training data
		TrainingExamples SamplesTest(testfile, testsize, input + output, input, output); //get test data
		Sizes layersize;
		layersize.push_back(input);
		layersize.push_back(hidden);
		layersize.push_back(output);

		OneDVector_Double y_train;
		OneDVector_Double y_test;

		NeuralNetwork network(layersize); //initialize network
		network.CreateNetwork(layersize, SamplesTrain);//setup neural network

		int w_size = weightsize1 + weightsize2 + biasize;


		OneDVector_Double w = parent;


		double step_w = 0.02;
		double step_eta = 0.01;

		OneDVector_Double pred_train = network.Evaluate_Proposal(SamplesTrain, w, layersize);
		OneDVector_Double pred_test = network.Evaluate_Proposal(SamplesTest, w, layersize);

		double eta = GetEta(pred_train, SamplesTrain.GetOutputValues());
		double tau_pro = exp(eta);

		int sigma_squared = 25;
		int nu_1 = 0;
		int nu_2 = 0;


		double prior_likelihood = Prior_Likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, h);

		LikelihoodReturnObject likelihood = Likelihood_Func(network, SamplesTrain, w, tau_pro, layersize);

		OneDVector_Double likelihoods = vectormanipulate.Generate1DVector_Double(2, 0);
		likelihoods[0] = likelihood.likelihood;
		likelihoods[1] = prior_likelihood;

		return likelihoods;

		//int naccept = 0;
		////Create new w proposal


		//double eta_pro = eta + rand_normal(0, step_eta);
		//tau_pro = exp(eta_pro);

		//LikelihoodReturnObject likelihood_proposal = Likelihood_Func(network, SamplesTrain, w_proposal, tau_pro, layersize);

		//double prior_prop = Prior_Likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, h);

		//double diff_likelihood = likelihood_proposal.likelihood - likelihood.likelihood;
		//double diff_priorliklihood = prior_prop - prior_likelihood;

		//double  mh_prob = 1;

		//if (exp(diff_likelihood + diff_priorliklihood) < 1) {
		//	mh_prob = exp(diff_likelihood + diff_priorliklihood);
		//}

		///*	RandomNumber rn;*/
		//double u = randfrom(0, 1);

		//if (u < mh_prob)
		//{
		//	return true;
		//}
		//else {
		//	return false;
		//}
	}
	bool ExecuteVetting(OneDVector_Double parent, OneDVector_Double offspring, int h)
	{
		VectorManipulation vectormanipulate;

		clock_t start = clock();

		int hidden = static_cast<int>(h);
		int weightsize1 = (input*hidden);//number of weights between hidden and input layer
		int weightsize2 = (hidden*output);//number of weights between hidden and output layer
		int biasize = hidden + output; //bias for hidden and output layer

		TrainingExamples SamplesTrain(trainfile, trainsize, input + output, input, output); //get training data
		TrainingExamples SamplesTest(testfile, testsize, input + output, input, output); //get test data
		Sizes layersize;
		layersize.push_back(input);
		layersize.push_back(hidden);
		layersize.push_back(output);

		OneDVector_Double y_train;
		OneDVector_Double y_test;

		NeuralNetwork network(layersize); //initialize network
		network.CreateNetwork(layersize, SamplesTrain);//setup neural network

		int w_size = weightsize1 + weightsize2 + biasize;


		OneDVector_Double w = parent;
		OneDVector_Double w_proposal = offspring;

		double step_w = 0.02;
		double step_eta = 0.01;

		OneDVector_Double pred_train = network.Evaluate_Proposal(SamplesTrain, w, layersize);
		OneDVector_Double pred_test = network.Evaluate_Proposal(SamplesTest, w, layersize);

		double eta = GetEta(pred_train, SamplesTrain.GetOutputValues());
		double tau_pro = exp(eta);

		int sigma_squared = 25;
		int nu_1 = 0;
		int nu_2 = 0;


		double prior_likelihood = Prior_Likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, h);

		LikelihoodReturnObject likelihood = Likelihood_Func(network, SamplesTrain, w, tau_pro, layersize);

		int naccept = 0;
		//Create new w proposal


		double eta_pro = eta + rand_normal(0, step_eta);
		tau_pro = exp(eta_pro);

		LikelihoodReturnObject likelihood_proposal = Likelihood_Func(network, SamplesTrain, w_proposal, tau_pro, layersize);

		double prior_prop = Prior_Likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, h);

		double diff_likelihood = likelihood_proposal.likelihood - likelihood.likelihood;
		double diff_priorliklihood = prior_prop - prior_likelihood;

		double  mh_prob = 1;

		if (exp(diff_likelihood + diff_priorliklihood) < 1) {
			mh_prob = exp(diff_likelihood + diff_priorliklihood);
		}

		/*	RandomNumber rn;*/
		double u = randfrom(0, 1);

		if (u < mh_prob)
		{
			return true;
		}
		else {
			return false;
		}

	}

	OneDVector_Double ExecuteGenerateOnly(OneDVector_Double parent, int h, int steps, int startind, int endind)
	{
		VectorManipulation vectormanipulate;
		clock_t start = clock();

		int hidden = static_cast<int>(h);
		int weightsize1 = (input*hidden);//number of weights between hidden and input layer
		int weightsize2 = (hidden*output);//number of weights between hidden and output layer
		int biasize = hidden + output; //bias for hidden and output layer

		TrainingExamples SamplesTrain(trainfile, trainsize, input + output, input, output); //get training data
		TrainingExamples SamplesTest(testfile, testsize, input + output, input, output); //get test data
		Sizes layersize;
		layersize.push_back(input);
		layersize.push_back(hidden);
		layersize.push_back(output);

		OneDVector_Double y_train;
		OneDVector_Double y_test;

		NeuralNetwork network(layersize); //initialize network
		network.CreateNetwork(layersize, SamplesTrain);//setup neural network

		int w_size = weightsize1 + weightsize2 + biasize;
		/*TwoDVector_Double pos_w = vectormanipulate.Generate2DVector_Double(samples, w_size, 1);
		OneDVector_Double pos_tau = vectormanipulate.Generate1DVector_Double(samples, 1);

		TwoDVector_Double fxtrain_samples = vectormanipulate.Generate2DVector_Double(samples, trainsize, 1);
		TwoDVector_Double fxtest_samples = vectormanipulate.Generate2DVector_Double(samples, testsize, 1);

		OneDVector_Double rmse_train = vectormanipulate.Generate1DVector_Double(samples, 0);
		OneDVector_Double rmse_test = vectormanipulate.Generate1DVector_Double(samples, 0);*/

		double step_w = 0.02;
		double step_eta = 0.01;


		OneDVector_Double w = parent;
		OneDVector_Double w_proposal = parent;

		for (int y = startind; y < (endind + 1); y++)
		{
			w_proposal[y] = w[y] + rand_normal(0, step_w);
		}

		OneDVector_Double pred_train = network.Evaluate_Proposal(SamplesTrain, w, layersize);
		OneDVector_Double pred_test = network.Evaluate_Proposal(SamplesTest, w, layersize);



		double eta = GetEta(pred_train, SamplesTrain.GetOutputValues());
		double tau_pro = exp(eta);

		int sigma_squared = 25;
		int nu_1 = 0;
		int nu_2 = 0;


		double prior_likelihood = Prior_Likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, h);

		LikelihoodReturnObject likelihood = Likelihood_Func(network, SamplesTrain, w, tau_pro, layersize);


		int naccept = 0;
		//Create new w proposal
		while (naccept < steps)
		{

			for (int y = startind; y < (endind + 1); y++)
			{
				w_proposal[y] = w[y] + rand_normal(0, step_w);
			}


			/*vectormanipulate.Print1DVector(w_proposal);*/

			double eta_pro = eta + rand_normal(0, step_eta);
			double tau_pro = exp(eta_pro);

			LikelihoodReturnObject likelihood_proposal = Likelihood_Func(network, SamplesTrain, w_proposal, tau_pro, layersize);

			double prior_prop = Prior_Likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, h);

			double diff_likelihood = likelihood_proposal.likelihood - likelihood.likelihood;
			double diff_priorliklihood = prior_prop - prior_likelihood;

			double  mh_prob = 1;

			if (exp(diff_likelihood + diff_priorliklihood) < 1) {
				mh_prob = exp(diff_likelihood + diff_priorliklihood);
			}

			/*	RandomNumber rn;*/
			double u = randfrom(0, 1);

			if (u < mh_prob)
			{
				// Update position
				/*cout << i << " is accepted sample" << endl;*/
				naccept += 1;
				likelihood = likelihood_proposal;

				prior_likelihood = prior_prop;
				w = w_proposal;
				eta = eta_pro;
			}
			else {
				cout << "\nRejected";
			}


		}


		return w;

		/*eta = np.log(np.var(pred_train - y_train))
		tau_pro = np.exp(eta)

		sigma_squared = 25
		nu_1 = 0
		nu_2 = 0*/

	}
	OneDVector_Double ExecuteVettingSteps(OneDVector_Double parent, OneDVector_Double offspring, int h, int steps, int startind, int endind)
	{
		VectorManipulation vectormanipulate;
		clock_t start = clock();

		int hidden = static_cast<int>(h);
		int weightsize1 = (input*hidden);//number of weights between hidden and input layer
		int weightsize2 = (hidden*output);//number of weights between hidden and output layer
		int biasize = hidden + output; //bias for hidden and output layer

		TrainingExamples SamplesTrain(trainfile, trainsize, input + output, input, output); //get training data
		TrainingExamples SamplesTest(testfile, testsize, input + output, input, output); //get test data
		Sizes layersize;
		layersize.push_back(input);
		layersize.push_back(hidden);
		layersize.push_back(output);

		OneDVector_Double y_train;
		OneDVector_Double y_test;

		NeuralNetwork network(layersize); //initialize network
		network.CreateNetwork(layersize, SamplesTrain);//setup neural network

		int w_size = weightsize1 + weightsize2 + biasize;
		/*TwoDVector_Double pos_w = vectormanipulate.Generate2DVector_Double(samples, w_size, 1);
		OneDVector_Double pos_tau = vectormanipulate.Generate1DVector_Double(samples, 1);

		TwoDVector_Double fxtrain_samples = vectormanipulate.Generate2DVector_Double(samples, trainsize, 1);
		TwoDVector_Double fxtest_samples = vectormanipulate.Generate2DVector_Double(samples, testsize, 1);

		OneDVector_Double rmse_train = vectormanipulate.Generate1DVector_Double(samples, 0);
		OneDVector_Double rmse_test = vectormanipulate.Generate1DVector_Double(samples, 0);*/

		OneDVector_Double w = parent;
		OneDVector_Double w_proposal = offspring;

		double step_w = 0.02;
		double step_eta = 0.01;



		OneDVector_Double pred_train = network.Evaluate_Proposal(SamplesTrain, w, layersize);
		OneDVector_Double pred_test = network.Evaluate_Proposal(SamplesTest, w, layersize);



		double eta = GetEta(pred_train, SamplesTrain.GetOutputValues());
		double tau_pro = exp(eta);

		int sigma_squared = 25;
		int nu_1 = 0;
		int nu_2 = 0;


		double prior_likelihood = Prior_Likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, h);

		LikelihoodReturnObject likelihood = Likelihood_Func(network, SamplesTrain, w, tau_pro, layersize);


		int naccept = 0;
		//Create new w proposal
		while (naccept < steps)
		{

			for (int y = startind; y < (endind + 1); y++)
			{
				w_proposal[y] = w[y] + rand_normal(0, step_w);
			}


			/*vectormanipulate.Print1DVector(w_proposal);*/

			double eta_pro = eta + rand_normal(0, step_eta);
			double tau_pro = exp(eta_pro);

			LikelihoodReturnObject likelihood_proposal = Likelihood_Func(network, SamplesTrain, w_proposal, tau_pro, layersize);

			double prior_prop = Prior_Likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, h);

			double diff_likelihood = likelihood_proposal.likelihood - likelihood.likelihood;
			double diff_priorliklihood = prior_prop - prior_likelihood;

			double  mh_prob = 1;

			if (exp(diff_likelihood + diff_priorliklihood) < 1) {
				mh_prob = exp(diff_likelihood + diff_priorliklihood);
			}

			/*	RandomNumber rn;*/
			double u = randfrom(0, 1);

			if (u < mh_prob)
			{
				// Update position
				/*cout << i << " is accepted sample" << endl;*/
				naccept += 1;
				likelihood = likelihood_proposal;

				prior_likelihood = prior_prop;
				w = w_proposal;
				eta = eta_pro;
			}
			else {
				cout << "\nRejected";
			}


		}


		return w;

		/*eta = np.log(np.var(pred_train - y_train))
		tau_pro = np.exp(eta)

		sigma_squared = 25
		nu_1 = 0
		nu_2 = 0*/

	}

	OneDVector_Double Execute(int samples, double h)
	{
		VectorManipulation vectormanipulate;
		clock_t start = clock();

		int hidden = static_cast<int>(h);
		int weightsize1 = (input*hidden);//number of weights between hidden and input layer
		int weightsize2 = (hidden*output);//number of weights between hidden and output layer
		int biasize = hidden + output; //bias for hidden and output layer

		TrainingExamples SamplesTrain(trainfile, trainsize, input + output, input, output); //get training data
		TrainingExamples SamplesTest(testfile, testsize, input + output, input, output); //get test data
		Sizes layersize;
		layersize.push_back(input);
		layersize.push_back(hidden);
		layersize.push_back(output);

		OneDVector_Double y_train;
		OneDVector_Double y_test;

		NeuralNetwork network(layersize); //initialize network
		network.CreateNetwork(layersize, SamplesTrain);//setup neural network

		int w_size = weightsize1 + weightsize2 + biasize;
		TwoDVector_Double pos_w = vectormanipulate.Generate2DVector_Double(samples, w_size, 1);
		OneDVector_Double pos_tau = vectormanipulate.Generate1DVector_Double(samples, 1);

		TwoDVector_Double fxtrain_samples = vectormanipulate.Generate2DVector_Double(samples, trainsize, 1);
		TwoDVector_Double fxtest_samples = vectormanipulate.Generate2DVector_Double(samples, testsize, 1);

		OneDVector_Double rmse_train = vectormanipulate.Generate1DVector_Double(samples, 0);
		OneDVector_Double rmse_test = vectormanipulate.Generate1DVector_Double(samples, 0);

		OneDVector_Double w = GenerateRandomWeights(w_size);
		OneDVector_Double w_proposal = GenerateRandomWeights(w_size);

		vectormanipulate.Print1DVector(w);

		double step_w = 0.02;
		double step_eta = 0.01;



		OneDVector_Double pred_train = network.Evaluate_Proposal(SamplesTrain, w, layersize);
		OneDVector_Double pred_test = network.Evaluate_Proposal(SamplesTest, w, layersize);



		double eta = GetEta(pred_train, SamplesTrain.GetOutputValues());
		double tau_pro = exp(eta);

		int sigma_squared = 25;
		int nu_1 = 0;
		int nu_2 = 0;


		double prior_likelihood = Prior_Likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, h);

		LikelihoodReturnObject likelihood = Likelihood_Func(network, SamplesTrain, w, tau_pro, layersize);
		cout << "Begin sampling using mcmc random walk";

		int naccept = 0;
		//Create new w proposal
		for (int i = 0; i < samples; i++)
		{

			for (int y = 0; y < w_proposal.size(); y++)
			{
				w_proposal[y] = w[y] + rand_normal(0, step_w);
			}


			/*vectormanipulate.Print1DVector(w_proposal);*/

			double eta_pro = eta + rand_normal(0, step_eta);
			double tau_pro = exp(eta_pro);

			LikelihoodReturnObject likelihood_proposal = Likelihood_Func(network, SamplesTrain, w_proposal, tau_pro, layersize);

			double prior_prop = Prior_Likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, h);

			double diff_likelihood = likelihood_proposal.likelihood - likelihood.likelihood;
			double diff_priorliklihood = prior_prop - prior_likelihood;

			double  mh_prob = 1;

			if (exp(diff_likelihood + diff_priorliklihood) < 1) {
				mh_prob = exp(diff_likelihood + diff_priorliklihood);
			}

			/*	RandomNumber rn;*/
			double u = randfrom(0, 1);

			if (u < mh_prob)
			{
				// Update position
				/*cout << i << " is accepted sample" << endl;*/
				naccept += 1;
				likelihood = likelihood_proposal;

				cout << "\nRMSE: " << likelihood_proposal.rmse;

				prior_likelihood = prior_prop;
				w = w_proposal;
				eta = eta_pro;
			}


		}


		return w;

		/*eta = np.log(np.var(pred_train - y_train))
		tau_pro = np.exp(eta)

		sigma_squared = 25
		nu_1 = 0
		nu_2 = 0*/


	}

	OneDVector_Double ExecuteTillExhausted(int samples, double h)
	{
		VectorManipulation vectormanipulate;
		clock_t start = clock();

		int hidden = static_cast<int>(h);
		int weightsize1 = (input*hidden);//number of weights between hidden and input layer
		int weightsize2 = (hidden*output);//number of weights between hidden and output layer
		int biasize = hidden + output; //bias for hidden and output layer

		TrainingExamples SamplesTrain(trainfile, trainsize, input + output, input, output); //get training data
		TrainingExamples SamplesTest(testfile, testsize, input + output, input, output); //get test data
		Sizes layersize;
		layersize.push_back(input);
		layersize.push_back(hidden);
		layersize.push_back(output);

		OneDVector_Double y_train;
		OneDVector_Double y_test;

		NeuralNetwork network(layersize); //initialize network
		network.CreateNetwork(layersize, SamplesTrain);//setup neural network

		int w_size = weightsize1 + weightsize2 + biasize;
		TwoDVector_Double pos_w = vectormanipulate.Generate2DVector_Double(samples, w_size, 1);
		OneDVector_Double pos_tau = vectormanipulate.Generate1DVector_Double(samples, 1);

		TwoDVector_Double fxtrain_samples = vectormanipulate.Generate2DVector_Double(samples, trainsize, 1);
		TwoDVector_Double fxtest_samples = vectormanipulate.Generate2DVector_Double(samples, testsize, 1);

		OneDVector_Double rmse_train = vectormanipulate.Generate1DVector_Double(samples, 0);
		OneDVector_Double rmse_test = vectormanipulate.Generate1DVector_Double(samples, 0);

		OneDVector_Double w = GenerateRandomWeights(w_size);
		OneDVector_Double w_proposal = GenerateRandomWeights(w_size);

		vectormanipulate.Print1DVector(w);

		double step_w = 0.02;
		double step_eta = 0.01;



		OneDVector_Double pred_train = network.Evaluate_Proposal(SamplesTrain, w, layersize);
		OneDVector_Double pred_test = network.Evaluate_Proposal(SamplesTest, w, layersize);



		double eta = GetEta(pred_train, SamplesTrain.GetOutputValues());
		double tau_pro = exp(eta);

		int sigma_squared = 25;
		int nu_1 = 0;
		int nu_2 = 0;


		double prior_likelihood = Prior_Likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, h);

		LikelihoodReturnObject likelihood = Likelihood_Func(network, SamplesTrain, w, tau_pro, layersize);
		cout << "Begin sampling using mcmc random walk";
		double currentbest = 99;
		double repeatcount = 0;

		int naccept = 0;
		//Create new w proposal
		/*for (int i = 0; i < samples; i++)*/
		while (repeatcount < 100) //if same rmse hasnt improved in 20 cycles

		{

			for (int y = 0; y < w_proposal.size(); y++)
			{
				w_proposal[y] = w[y] + rand_normal(0, step_w);
			}


			/*vectormanipulate.Print1DVector(w_proposal);*/

			double eta_pro = eta + rand_normal(0, step_eta);
			double tau_pro = exp(eta_pro);

			LikelihoodReturnObject likelihood_proposal = Likelihood_Func(network, SamplesTrain, w_proposal, tau_pro, layersize);

			double prior_prop = Prior_Likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, h);

			double diff_likelihood = likelihood_proposal.likelihood - likelihood.likelihood;
			double diff_priorliklihood = prior_prop - prior_likelihood;

			double  mh_prob = 1;

			if (exp(diff_likelihood + diff_priorliklihood) < 1) {
				mh_prob = exp(diff_likelihood + diff_priorliklihood);
			}

			/*	RandomNumber rn;*/
			double u = randfrom(0, 1);

			if (u < mh_prob)
			{
				// Update position
				/*cout << i << " is accepted sample" << endl;*/
				naccept += 1;
				likelihood = likelihood_proposal;

				cout << "\nRMSE: " << likelihood_proposal.rmse;

				prior_likelihood = prior_prop;
				w = w_proposal;
				eta = eta_pro;

				if (likelihood_proposal.rmse > currentbest)
				{
					repeatcount++;
					cout << "\nRepeat Count: " << repeatcount;
				}
				else
				{
					currentbest = likelihood_proposal.rmse;
					repeatcount = 0;

				}

			}


		}


		return w;

		/*eta = np.log(np.var(pred_train - y_train))
		tau_pro = np.exp(eta)

		sigma_squared = 25
		nu_1 = 0
		nu_2 = 0*/


	}

	/* generate a random floating point number from min to max */
	double randfrom(double min, double max)
	{
		double range = (max - min);
		double div = RAND_MAX / range;
		return min + (rand() / div);
	}


	/*
	--rand_normal--
	Box Muller Random Numbers
	*/
	double rand_normal(double mean, double stddev) {
		static double n2 = 0.0;
		static int n2_cached = 0;

		if (!n2_cached) {
			// choose a point x,y in the unit circle uniformly at random
			double x, y, r;
			do {
				//  scale two random integers to doubles between -1 and 1
				x = 2.0*rand() / RAND_MAX - 1;
				y = 2.0*rand() / RAND_MAX - 1;

				r = x*x + y*y;
			} while (r == 0.0 || r > 1.0);

			// Apply Box-Muller transform on x, y
			double d = sqrt(-2.0*log(r) / r);
			double n1 = x*d;
			n2 = y*d;

			// scale and translate to get desired mean and standard deviation
			double result = n1*stddev + mean;

			n2_cached = 1;
			return result;
		}
		else {
			n2_cached = 0;
			return n2*stddev + mean;
		}
	}

	//likelihood_func(neuralnet, self.traindata, w, tau_pro)

	LikelihoodReturnObject Likelihood_Func(NeuralNetwork network, TrainingExamples SamplesTrain, OneDVector_Double w, double tau_pro, Sizes layersize)
	{
		VectorManipulation vectormanipulate;
		OneDVector_Double loss = vectormanipulate.Generate1DVector_Double(SamplesTrain.GetSize(), 0);
		Data y = SamplesTrain.GetOutputValues();

		OneDVector_Double fx = network.Evaluate_Proposal(SamplesTrain, w, layersize);

		double rmse = network.SumSquaredError(SamplesTrain, layersize);

		double temp = -0.5 * log(2 * M_PI * tau_pro);
		double sum = 0;
		//Find difference between y and fx
		for (int i = 0; i < fx.size(); i++)
		{
			loss[i] = temp - (0.5 * (pow((y[i][0] - fx[i]), 2) / tau_pro));
			sum += loss[i];

		}

		LikelihoodReturnObject returnobject;
		returnobject.likelihood = sum;
		returnobject.pred_train = fx;
		returnobject.rmse = rmse;

		return returnobject;
	}


	double Prior_Likelihood(double sigma_squared, double nu_1, double nu_2, OneDVector_Double w, double tausq, int hidden)
	{
		double part1 = -1 * ((input * hidden + hidden + 2) / 2) * log(sigma_squared);

		//Square each weight value

		double sum = 0;

		for (int i = 0; i < w.size(); i++)
		{
			sum += pow(w[i], 2);
		}

		double part2 = 1 / (2 * sigma_squared) * sum;
		double log_loss = part1 - part2 - (1 + nu_1) * log(tausq) - (nu_2 / tausq);

		return log_loss;
	}
	double GetEta(OneDVector_Double networkoutput, TwoDVector_Double actualoutput)
	{


		VectorManipulation vectormanipulate;


		//vectormanipulate.Print1DVector(networkoutput);
		//vectormanipulate.Print2DVector(actualoutput);


		OneDVector_Double diff = vectormanipulate.Generate1DVector_Double(networkoutput.size(), 0);
		double sum = 0;

		for (int i = 0; i < networkoutput.size(); i++)
		{
			double difference = networkoutput[i] - actualoutput[i][0];
			sum += difference;
			diff[i] = difference;
		}

		double mean = sum / networkoutput.size();

		sum = 0;

		for (int i = 0; i < diff.size(); i++)
		{
			sum += pow(diff[i] - mean, 2);

		}

		double variance = sum / (networkoutput.size() - 1);

		return log(variance);


	}


	OneDVector_Double GenerateRandomWeights(int w_size)
	{
		OneDVector_Double vec;

		for (int i = 0; i < w_size; i++)
		{
			vec.push_back(RandomWeights());
		}

		return vec;
	}








};




/*
--CoEvolution--
*/
class CoEvolution : public  NeuralNetwork, public virtual TrainingExamples, public   GeneticAlgorithmn
{
public: //class variables
	int  NoSpecies;
	GAvector Species;
	TableVector  TableSp;
	int PopSize;
	vector<bool> NotConverged;
	Sizes SpeciesSize;
	Layer   Individual;
	Layer  BestIndividual;
	double bestglobalfit;
	Data TempTable;
	int TotalEval;
	int TotalSize;
	int SingleGAsize;
	double Train;
	double Test;
	double Valid;
	int kid;

	//constructor
	CoEvolution() {
	}
	//class functions
	void UpdateSpeciesPartial(int popsize, TwoDVector_Double data);
	void UpdateSpeciesBaye(int popsize, TwoDVector_Double data);
	void InitializeSpecies(int popsize);
	void  	EvaluateSpecies(NeuralNetwork network, TrainingExamples Sample, int Island);
	void  	EvaluateSpeciesBayesian(NeuralNetwork network, TrainingExamples Sample, int Island);

	void GetBestTable(int sp);
	void PrintBestIndexes(int sp);
	void SwapLocalBestToCCRegion(int epochs);


	void PrintSpecies();
	OneDVector_Double SplitIndividualGetBackOffspring(OneDVector_Double from, int s);

	void Join();

	void Print();

	void sort(int s);

	void  	find_parents(int s, NeuralNetwork network, TrainingExamples Sample, int Island);

	void 	EvalNewPop(int pass, int s, NeuralNetwork network, TrainingExamples Sample, int Island);

	void  	rep_parents(int s, NeuralNetwork network, TrainingExamples Sample, int Island);
	void   	EvolveSubPopulations(int repetitions, double h, NeuralNetwork network, TrainingExamples Sample, double mutation, int depth, ofstream &out2, int Island);
	void   	EvolveSubPopulationsBCNESteps(int repetitions, double h, NeuralNetwork network, TrainingExamples Sample, double mutation, int depth, ofstream &out2, int Island);
	void   	EvolveSubPopulationsBCNEGenerateOnly(int repetitions, double h, NeuralNetwork network, TrainingExamples Sample, double mutation, int depth, ofstream &out2, int Island);

	void   	EvolveSubPopulationsBCNE(int repetitions, double h, NeuralNetwork network, TrainingExamples Sample, double mutation, int depth, ofstream &out2, int Island);
	void   	EvolveBCNE(string file, ofstream &trainout, ofstream &testout, ofstream &weightsout, ofstream &rmsetrainout, ofstream &rmsetestout, ofstream &acceptout, int repetitions, double h, NeuralNetwork network, TrainingExamples Sample, double mutation, int depth, ofstream &out1, ofstream &out2, int Island);


};

void  CoEvolution::UpdateSpeciesPartial(int popsize, TwoDVector_Double data)
{
	for (int s = 0; s < NoSpecies; s++) {


		for (int y = 0; y < data.size(); y++)
		{
			OneDVector_Double sampledchild = SplitIndividualGetBackOffspring(data[y], s);

			for (int o = 0; o < Species[s].NumVariable; o++)
				Species[s].Population[y + popsize].Chrome[o] = sampledchild[o];
		}

		//Species[s].BestIndex = popsize + (data.size() - 1);

	}

}

/*
--InitializeSpecies--
Initialize the sub-populations in the CC framework
*/
void  CoEvolution::UpdateSpeciesBaye(int popsize, TwoDVector_Double data)
{
	for (int s = 0; s < NoSpecies; s++) {


		for (int y = 0; y < data.size(); y++)
		{
			OneDVector_Double sampledchild = SplitIndividualGetBackOffspring(data[y], s);

			for (int o = 0; o < Species[s].NumVariable; o++)
				Species[s].Population[y].Chrome[o] = sampledchild[o];
		}

		Species[s].BestIndex = data.size() - 1;

	}

}
/*
--InitializeSpecies--
Initialize the sub-populations in the CC framework
*/
void  CoEvolution::InitializeSpecies(int popsize)
{       //variable declaration
	PopSize = popsize;
	GAvector SpeciesP(NoSpecies); //species of type Genetic Algorithm
	Species = SpeciesP;

	for (int Sp = 0; Sp < NoSpecies; Sp++) {
		NotConverged.push_back(false);
	}

	basic_seed = 0.4122;
	RUN = 2;

	seed = basic_seed + (1.0 - basic_seed)*(double)((RUN)-1) / (double)MAXRUN;
	if (seed > 1.0) printf("\n warning!!! seed number exceeds 1.0");

	/*for( int s =0; s < NoSpecies; s++){
	Species[s].randomize(seed);
	}*/

	TotalSize = 0;
	for (int row = 0; row < NoSpecies; row++)
		TotalSize += SpeciesSize[row];

	for (int row = 0; row < TotalSize; row++)
		Individual.push_back(0);//initialize with 0s

	for (int s = 0; s < NoSpecies; s++) {
		Species[s].NumVariable = SpeciesSize[s];
		Species[s].InitilisePopulation(popsize);//initialize species with 0s...that is the individuals to 0s
	}

	TableSp.resize(NoSpecies);//resize to number of sub-pops--- to store the combined species from each of the sub-pops

	for (int row = 0; row < NoSpecies; row++)
		for (int col = 0; col < SpeciesSize[row]; col++)
			TableSp[row].SingleSp.push_back(0); //resize to the size of the species in each sub-pop
}

/*
--PrintSpecies--
Print the individuals within the population
*/
void CoEvolution::PrintSpecies()
{
	cout << "print species" << endl;
	for (int s = 0; s < NoSpecies; s++) {
		Species[s].PrintPopulation();
		cout << s << endl;
	}
}

/*
--GetBestTable--
Populate TableSp with best species

*/



void  CoEvolution::SwapLocalBestToCCRegion(int epochs) //epochs = 150 - 1 = 149. 150 and above indexes swap to cc region so it won't be replaced in next iteration
{
	cout << "\n";
	//Find worst index in CC range


	double worstFit = 0;
	double worstind = 0;

	//for each species,
	for (int sN = 0; sN < NoSpecies; sN++)
	{
		for (int PIndex = 0; PIndex < epochs; PIndex++)
		{
			if ((MINIMIZE * worstFit) < (MINIMIZE * Species[sN].Population[PIndex].Fitness))
			{
				worstFit = Species[sN].Population[PIndex].Fitness;
				worstind = PIndex;
				//  cout<<Species[SpNum].Population[PIndex].Fitness<<endl;
			}
		}

		if (Species[sN].BestIndex > (epochs - 1))
		{

			int best = Species[sN].BestIndex;
			cout << "\n[Swap to CC Region] Species " << sN << ": " << Species[sN].BestIndex;

			Species[sN].BestIndex = worstind;

			//Foreach chrome, replace worst index in species
			for (int s = 0; s < SpeciesSize[sN]; s++)
				Species[sN].Population[worstind].Chrome[s] = Species[sN].Population[best].Chrome[s];//get best set of chromes

		}



	}
}


void  CoEvolution::PrintBestIndexes(int CurrentSp)
{
	cout << "\n";

	//for each species,
	for (int sN = 0; sN < CurrentSp; sN++)
	{
		cout << " Species " << sN << ": " << Species[sN].BestIndex;


	}
}

void  CoEvolution::GetBestTable(int CurrentSp)
{
	int Best;

	//for each species,
	for (int sN = 0; sN < CurrentSp; sN++) {
		Best = Species[sN].BestIndex;//best index

		for (int s = 0; s < SpeciesSize[sN]; s++)
			TableSp[sN].SingleSp[s] = Species[sN].Population[Best].Chrome[s];//get best set of chromes
	}

	for (int sN = CurrentSp; sN < NoSpecies; sN++) {
		Best = Species[sN].BestIndex;
		for (int s = 0; s < SpeciesSize[sN]; s++)
			TableSp[sN].SingleSp[s] = Species[sN].Population[Best].Chrome[s];//update
	}
}

/*
--Join--
Join all the sub-populations together
*/
OneDVector_Double   CoEvolution::SplitIndividualGetBackOffspring(OneDVector_Double extractfrom, int s)
{
	VectorManipulation vectormanipulate;

	OneDVector_Double returnobj = vectormanipulate.Generate1DVector_Double(SpeciesSize[s], 0);

	int index = 0;

	for (int row = 0; row < NoSpecies; row++)
	{
		for (int col = 0; col < SpeciesSize[row]; col++)
		{
			if (row == s)
			{
				returnobj[col] = extractfrom[index];	//join the species to form one individual
			}

			index++;
		}

	}

	return returnobj;
}

/*
--Join--
Join all the sub-populations together
*/
void   CoEvolution::Join()
{
	int index = 0;

	for (int row = 0; row < NoSpecies; row++) {
		for (int col = 0; col < SpeciesSize[row]; col++) {
			Individual[index] = TableSp[row].SingleSp[col];	//join the species to form one individual
			index++;
		}
	}
}

/*
--Print--
*/
void   CoEvolution::Print()
{
	cout << "coevolution :: Print() Single species" << endl;
	for (int row = 0; row < NoSpecies; row++) {
		for (int col = 0; col < SpeciesSize[row]; col++) {
			cout << TableSp[row].SingleSp[col] << " ";
		}
		cout << endl;
	}
	cout << endl;
	cout << "coevolution :: Print() Individual" << endl;
	for (int row = 0; row < TotalSize; row++)
		cout << Individual[row] << " ";

	cout << endl << endl;
}

/*
--EvaluateSpecies--
*/

void    CoEvolution::EvaluateSpeciesBayesian(NeuralNetwork network, TrainingExamples Sample, int Island)
{

	MCMC bayesian;

	for (int SpNum = 0; SpNum < NoSpecies; SpNum++) {
		GetBestTable(SpNum);

		//---------make the first individual in the population the best

		for (int i = 0; i < Species[SpNum].NumVariable; i++)
			TableSp[SpNum].SingleSp[i] = Species[SpNum].Population[0].Chrome[i];

		Join();


		OneDVector_Double out = bayesian.GetLikelihoodForSubPop(Individual, 4);
		Species[SpNum].Population[0].Values[0] = out[0]; //Set likelihood
		Species[SpNum].Population[0].Values[1] = out[1]; //set prior likelihood

		//Species[SpNum].Population[0].Fitness = network.ForwardFitnessPass(Individual, Sample, Island);//ObjectiveFunc(Individual);
		TotalEval++;

		Species[SpNum].BestLikelihood = Species[SpNum].Population[0].Values[0];
		Species[SpNum].BestIndexLikelihood = 0;


		/*Species[SpNum].BestFit = Species[SpNum].Population[0].Fitness;
		Species[SpNum].BestIndex = 0;

		Species[SpNum].WorstFit = Species[SpNum].Population[0].Fitness;
		Species[SpNum].WorstIndex = 0;*/
		// cout<<"g"<<endl;
		//------------do for the rest

		for (int PIndex = 0; PIndex < PopSize; PIndex++) {

			//cout << PIndex << " -- " << endl;

			for (int i = 0; i < Species[SpNum].NumVariable; i++)
				TableSp[SpNum].SingleSp[i] = Species[SpNum].Population[PIndex].Chrome[i];

			Join();
			//Print();

			Species[SpNum].Population[PIndex].Fitness = network.ForwardFitnessPass(Individual, Sample, Island);//
			TotalEval++;
			//   ObjectiveFunc(Individual);

			if ((MINIMIZE * Species[SpNum].BestFit) > (MINIMIZE * Species[SpNum].Population[PIndex].Fitness))
			{
				Species[SpNum].BestFit = Species[SpNum].Population[PIndex].Fitness;
				Species[SpNum].BestIndex = PIndex;
				//  cout<<Species[SpNum].Population[PIndex].Fitness<<endl;
			}

			if ((MINIMIZE * Species[SpNum].WorstFit) < (MINIMIZE * Species[SpNum].Population[PIndex].Fitness))
			{
				Species[SpNum].WorstFit = Species[SpNum].Population[PIndex].Fitness;
				Species[SpNum].WorstIndex = PIndex;
				//  cout<<Species[SpNum].Population[PIndex].Fitness<<endl;
			}
		}

		// cout<< Species[SpNum].BestIndex<<endl;
		cout << "SP " << SpNum + 1 << " Evaluated" << endl;
	}
}

/*
--EvaluateSpecies--
*/

void    CoEvolution::EvaluateSpecies(NeuralNetwork network, TrainingExamples Sample, int Island)
{
	for (int SpNum = 0; SpNum < NoSpecies; SpNum++) {
		GetBestTable(SpNum);

		//---------make the first individual in the population the best

		for (int i = 0; i < Species[SpNum].NumVariable; i++)
			TableSp[SpNum].SingleSp[i] = Species[SpNum].Population[0].Chrome[i];

		Join();
		Species[SpNum].Population[0].Fitness = network.ForwardFitnessPass(Individual, Sample, Island);//ObjectiveFunc(Individual);
		TotalEval++;

		Species[SpNum].BestFit = Species[SpNum].Population[0].Fitness;
		Species[SpNum].BestIndex = 0;

		Species[SpNum].WorstFit = Species[SpNum].Population[0].Fitness;
		Species[SpNum].WorstIndex = 0;
		// cout<<"g"<<endl;
		//------------do for the rest

		for (int PIndex = 0; PIndex < PopSize; PIndex++) {

			//cout << PIndex << " -- " << endl;

			for (int i = 0; i < Species[SpNum].NumVariable; i++)
				TableSp[SpNum].SingleSp[i] = Species[SpNum].Population[PIndex].Chrome[i];

			Join();
			//Print();

			Species[SpNum].Population[PIndex].Fitness = network.ForwardFitnessPass(Individual, Sample, Island);//
			TotalEval++;
			//   ObjectiveFunc(Individual);

			if ((MINIMIZE * Species[SpNum].BestFit) > (MINIMIZE * Species[SpNum].Population[PIndex].Fitness))
			{
				Species[SpNum].BestFit = Species[SpNum].Population[PIndex].Fitness;
				Species[SpNum].BestIndex = PIndex;
				//  cout<<Species[SpNum].Population[PIndex].Fitness<<endl;
			}

			if ((MINIMIZE * Species[SpNum].WorstFit) < (MINIMIZE * Species[SpNum].Population[PIndex].Fitness))
			{
				Species[SpNum].WorstFit = Species[SpNum].Population[PIndex].Fitness;
				Species[SpNum].WorstIndex = PIndex;
				//  cout<<Species[SpNum].Population[PIndex].Fitness<<endl;
			}
		}

		// cout<< Species[SpNum].BestIndex<<endl;
		cout << "\nSP " << SpNum + 1 << " Evaluated";
	}
}

/*
--find_parents--
Here the parents to be replaced are added to the temporary sub-population to assess
their goodness against the new solutions formed which will be the basis of whether they should be kept or not
*/

void CoEvolution::find_parents(int s, NeuralNetwork network, TrainingExamples Sample, int Island)
{
	int i, j, k;
	double u, v;

	Species[s].my_family();

	for (j = 0; j < family; j++)
	{
		Species[s].NewPop[Species[s].kids + j].Chrome = Species[s].Population[Species[s].mom[j]].Chrome; //Add parents to new population

		GetBestTable(s); //Get best species
		for (int i = 0; i < Species[s].NumVariable; i++)
			TableSp[s].SingleSp[i] = Species[s].NewPop[Species[s].kids + j].Chrome[i]; //Get species into table used by join() to concatenate
		Join();	//Concatenate to form one individual

		Species[s].NewPop[Species[s].kids + j].Fitness = network.ForwardFitnessPass(Individual, Sample, Island); //Evalaute fitness of new addition which are the parents
		TotalEval++;
	}
}
/*
--EvalNewPop--
Evaluate fitness of new sub-populations
*/

void CoEvolution::EvalNewPop(int pass, int s, NeuralNetwork network, TrainingExamples Sample, int Island)
{
	GetBestTable(s);
	for (int i = 0; i < Species[s].NumVariable; i++)
		TableSp[s].SingleSp[i] = Species[s].NewPop[pass].Chrome[i];
	Join();

	Species[s].NewPop[pass].Fitness = network.ForwardFitnessPass(Individual, Sample, Island);
	TotalEval++;
}

/*
--sort--
Sorts the species in  the sub-population in order of fitness
*/
void  CoEvolution::sort(int s)
{
	int i, j, temp;
	double dbest;

	for (i = 0; i < (Species[s].kids + family); i++) Species[s].list[i] = i;

	if (MINIMIZE)
		for (i = 0; i < (Species[s].kids + family - 1); i++)
		{
			dbest = Species[s].NewPop[Species[s].list[i]].Fitness;

			for (j = i + 1; j < (Species[s].kids + family); j++)
			{
				if (Species[s].NewPop[Species[s].list[j]].Fitness < dbest)
				{
					dbest = Species[s].NewPop[Species[s].list[j]].Fitness;
					temp = Species[s].list[j];
					Species[s].list[j] = Species[s].list[i];
					Species[s].list[i] = temp;
				}
			}
		}
	else
		for (i = 0; i < (Species[s].kids + family - 1); i++)
		{
			dbest = Species[s].NewPop[Species[s].list[i]].Fitness;

			for (j = i + 1; j < (Species[s].kids + family); j++)
			{
				if (Species[s].NewPop[Species[s].list[j]].Fitness > dbest)
				{
					dbest = Species[s].NewPop[Species[s].list[j]].Fitness;
					temp = Species[s].list[j];
					Species[s].list[j] = Species[s].list[i];
					Species[s].list[i] = temp;
				}
			}
		}
}

/*
--rep_parents--
Here the best (1 or 2) individuals replace the family of parents
*/

void CoEvolution::rep_parents(int s, NeuralNetwork network, TrainingExamples Sample, int Island)
{
	int i, j;
	for (j = 0; j < family; j++)
	{
		Species[s].Population[Species[s].mom[j]].Chrome = Species[s].NewPop[Species[s].list[j]].Chrome; //Update population with new species

		GetBestTable(s); //Update fitness and update best

		for (int i = 0; i < Species[s].NumVariable; i++)
			TableSp[s].SingleSp[i] = Species[s].Population[Species[s].mom[j]].Chrome[i];	//Get into one table for concatenation
		Join();	//Concatenate into one individual

		Species[s].Population[Species[s].mom[j]].Fitness = network.ForwardFitnessPass(Individual, Sample, Island); //Update fitness of new individual
		TotalEval++;
	}
}

/*
--EvolveSubPopulations--
Evolve and form new sub-populations
*/
//(1, 1, network, Samples, mutation, 0, out2, 1)
void CoEvolution::EvolveSubPopulations(int repetitions, double h, NeuralNetwork network, TrainingExamples Samples, double mutation, int depth, ofstream & out1, int Island)
{
	double tempfit;
	double tempfitWorst;
	int count = 0;
	int tag = 99999;
	kid = KIDS;

	//species are subpopulations?
	for (int s = 0; s < NoSpecies; s++) {
		if ((rand() % 100) < (h * 100)) {
			for (int r = 0; r < repetitions; r++) {
				if (NotConverged[s]) {
					tempfit = Species[s].Population[Species[s].BestIndex].Fitness; //get fitness of fittest individuals
					Species[s].kids = KIDS;
					Species[s].RandomParents();//select random parents for mating
											   //2 parents randomly selected

											   //2 kids
					for (int i = 0; i < Species[s].kids; i++)
					{

						//Modified 11/04/2017 9:15 PM Assertation fail
						tag = Species[s].GenerateNewPCX(i, network, Samples, mutation, depth); //generate a child using PCX


						if (tag == 0) break;
					}
					if (tag == 0) {
						NotConverged[s] = false;
					}

					for (int i = 0; i < Species[s].kids; i++)
						EvalNewPop(i, s, network, Samples, Island);//evaluate new sub-population for fitness

					find_parents(s, network, Samples, Island);  // form a pool from which a solution is to be
																//   replaced by the created child
					Species[s].sort();          // sort the kids+parents by fitness
					rep_parents(s, network, Samples, Island);   // a chosen parent is replaced by the child

					Species[s].BestIndex = 0;
					Species[s].WorstIndex = 0;

					tempfit = Species[s].Population[0].Fitness;
					tempfitWorst = Species[s].Population[0].Fitness;

					for (int i = 1; i < PopSize; i++) {

						if ((MINIMIZE *    Species[s].Population[i].Fitness) < (MINIMIZE * tempfit))
						{
							tempfit = Species[s].Population[i].Fitness;
							Species[s].BestFit = tempfit;
							Species[s].BestIndex = i;//update fittest
						}

						if ((MINIMIZE *    Species[s].Population[i].Fitness) > (MINIMIZE * tempfitWorst))
						{
							tempfitWorst = Species[s].Population[i].Fitness;
							Species[s].WorstFit = tempfitWorst;
							Species[s].WorstIndex = i;//update fittest
						}

					}

					GetBestTable(s);//update best species
					Join();//concatenate into one
				}
			}
		}
	}
}
void CoEvolution::EvolveBCNE(string file, ofstream &trainout, ofstream &testout, ofstream &weightsout, ofstream &rmsetrainout, ofstream &rmsetestout, ofstream &acceptout, int repetitions, double h, NeuralNetwork network, TrainingExamples Samples, double mutation, int depth, ofstream & out1, ofstream & out2, int Island)
{
	VectorManipulation vectormanipulate;

	double tempfit;
	double tempfitWorst;
	int count = 0;
	int tag = 99999;
	kid = KIDS;
	int numsamples = 20000;
	int samplescounter = 0;
	double localtrain = 0;
	double localtest = 0;

	//---------------------------------------------------
	MCMC bayesian;
	clock_t start = clock();

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	TrainingExamples SamplesTrain(trainfile, trainsize, input + output, input, output); //get training data
	TrainingExamples SamplesTest(testfile, testsize, input + output, input, output); //get test data
	Sizes layersize;
	layersize.push_back(input);
	layersize.push_back(hidden);
	layersize.push_back(output);

	OneDVector_Double y_train;
	OneDVector_Double y_test;

	int w_size = weightsize1 + weightsize2 + biasize;
	TwoDVector_Double pos_w = vectormanipulate.Generate2DVector_Double(1, w_size, 1);
	OneDVector_Double pos_tau = vectormanipulate.Generate1DVector_Double(1, 1);

	TwoDVector_Double fxtrain_samples = vectormanipulate.Generate2DVector_Double(1, trainsize, 1);
	TwoDVector_Double fxtest_samples = vectormanipulate.Generate2DVector_Double(1, testsize, 1);

	OneDVector_Double rmse_train = vectormanipulate.Generate1DVector_Double(1, 0);
	OneDVector_Double rmse_test = vectormanipulate.Generate1DVector_Double(1, 0);

	OneDVector_Double w = bayesian.GenerateRandomWeights(w_size);
	OneDVector_Double w_proposal = bayesian.GenerateRandomWeights(w_size);

	vectormanipulate.Print1DVector(w);

	double step_w = 0.02;
	double step_eta = 0.01;



	OneDVector_Double pred_train = network.Evaluate_Proposal(SamplesTrain, w, layersize);
	OneDVector_Double pred_test = network.Evaluate_Proposal(SamplesTest, w, layersize);

	double eta = bayesian.GetEta(pred_train, SamplesTrain.GetOutputValues());
	double tau_pro = exp(eta);

	int sigma_squared = 25;
	int nu_1 = 0;
	int nu_2 = 0;


	double prior_likelihood = bayesian.Prior_Likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, h);

	LikelihoodReturnObject likelihood = bayesian.Likelihood_Func(network, SamplesTrain, w, tau_pro, layersize);
	cout << "Begin sampling using mcmc random walk";


	/*cout << "\nInitial RMSE: " << likelihood.rmse;*/

	int naccept = 0;

	ChartData writedata;
	double currentbest = 99;
	double repeatcount = 0;

	//Create new w proposal
	while(samplescounter < numsamples && repeatcount < 20)
	{
		//Generate offspring
		bool generateanother = true;
		//----------------------------------------
		for (int s = 0; s < NoSpecies; s++) {
			samplescounter++;
			
			generateanother = true;
			if ((rand() % 100) < (h * 100)) {
				while(generateanother)
				{

					

					tempfit = Species[s].Population[Species[s].BestIndex].Fitness; //get fitness of fittest individuals
					Species[s].kids = KIDS;
					Species[s].RandomParents();//select random parents for mating
											   //2 parents randomly selected

					//generate 2 kids
					for (int i = 0; i < Species[s].kids; i++)
					{

						//Modified 11/04/2017 9:15 PM Assertation fail
						tag = Species[s].GenerateNewPCX(i, network, Samples, mutation, depth); //generate a child using PCX


						if (tag == 0) break;
					}
					if (tag == 0) {
						NotConverged[s] = false;
					}

					//evalaute fitness of kids
					for (int i = 0; i < Species[s].kids; i++)
						EvalNewPop(i, s, network, Samples, Island);//evaluate new sub-population for fitness

					find_parents(s, network, Samples, Island);  // form a pool from which a solution is to be
																//   replaced by the created child
					Species[s].sort();          // sort the kids+parents by fitness


					//Best solution index after sorting lies in Species[s].list[0]

					//----------------------------------------------------------------------------------------
					//Check bayesian

					GetBestTable(s);

					for (int o = 0; o < Species[s].NumVariable; o++)
						TableSp[s].SingleSp[o] = Species[s].NewPop[Species[s].list[0]].Chrome[o];
					Join();

					//get best offspring check if can be accepted into population

					OneDVector_Double offspring = Individual;

					double eta_pro = eta + rand_normal(0, step_eta);
					double tau_pro = exp(eta_pro);

					LikelihoodReturnObject likelihood_proposal = bayesian.Likelihood_Func(network, SamplesTrain, offspring, tau_pro, layersize);

					double prior_prop = bayesian.Prior_Likelihood(sigma_squared, nu_1, nu_2, offspring, tau_pro, h);

					double diff_likelihood = likelihood_proposal.likelihood - likelihood.likelihood;
					double diff_priorliklihood = prior_prop - prior_likelihood;

					double  mh_prob = 1;

					if (exp(diff_likelihood + diff_priorliklihood) < 1) {
						mh_prob = exp(diff_likelihood + diff_priorliklihood);
					}

					double u = bayesian.randfrom(0, 1);

					if (u < mh_prob)
					{

						generateanother = false;

						// Update position
						//cout << "\nAccepted offspring";
						naccept += 1;
						likelihood = likelihood_proposal;

						prior_likelihood = prior_prop;
						w = offspring;
						eta = eta_pro;

						//add to posterior
						pos_w.push_back(offspring);

						//--------------------------------------------
						//CC bit

						rep_parents(s, network, Samples, Island);   // a chosen parent is replaced by the child

						Species[s].BestIndex = 0;
						Species[s].WorstIndex = 0;

						tempfit = Species[s].Population[0].Fitness;
						tempfitWorst = Species[s].Population[0].Fitness;

						for (int i = 1; i < PopSize; i++) {

							if ((MINIMIZE *    Species[s].Population[i].Fitness) < (MINIMIZE * tempfit))
							{
								tempfit = Species[s].Population[i].Fitness;
								Species[s].BestFit = tempfit;
								Species[s].BestIndex = i;//update fittest
							}

							if ((MINIMIZE *    Species[s].Population[i].Fitness) > (MINIMIZE * tempfitWorst))
							{
								tempfitWorst = Species[s].Population[i].Fitness;
								Species[s].WorstFit = tempfitWorst;
								Species[s].WorstIndex = i;//update fittest
							}

						}


						//cout << "\nBEst Index: " << Species[s].BestIndex;

						GetBestTable(s);//update best species
						Join();//concatenate into one

						network.ChoromesToNeurons(offspring, 1);
						//--------------------------------------------------
						// Write output data ---------------------------------------------------------------------------
						//test the neural network with training data
						//For each sample
						//save weights
						//save train output
						//save test output
						//rmse train
						//rmse test
						network.SaveLearnedData(layersize, file);//save the neural network weights

						localtrain = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
						writedata.WriteAccuracyData(network.GetOutput(), trainout, trainsize); //Write out actual vs prediction train
						rmsetrainout << "\n" << localtrain; //Write out RMSE train.

						//cout << "\nLocal Train: " << localtrain;

						localtest = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
						writedata.WriteAccuracyData(network.GetOutput(), testout, testsize); //Write out actual vs prediction test
						rmsetestout << "\n" << localtest; //Write out RMSE test.

						//cout << "\nLocal Test: " << localtest;

						Layer netweights = network.Neurons_to_chromes(1);
						writedata.WriteAccuracyData(netweights, weightsout, netweights.size()); //Write out weights.

																								//--------------------------------------------------------------------------------------------
						

					}
					else {
						//cout << "\nDeclined offspring - Generate another";
					}
					//----------------------------------------------------------------------------------------


					count = count + 1;

					//for (int i = 0; i < Species[s].kids; i++)
					//	EvalNewPop(i, s, network, Samples, Island);//evaluate new sub-population for fitness

				}
			}
		}
		//----------------------------------------
	/*	for (int y = 0; y < w_proposal.size(); y++)
		{
			w_proposal[y] = w[y] + rand_normal(0, step_w);
		}*/


		cout << "\nSample:" << samplescounter << " RMSE: " << Species[NoSpecies - 1].Population[Species[NoSpecies - 1].BestIndex].Fitness;


		if (Species[NoSpecies - 1].Population[Species[NoSpecies - 1].BestIndex].Fitness > currentbest || Species[NoSpecies - 1].Population[Species[NoSpecies - 1].BestIndex].Fitness == currentbest)
		{
			repeatcount++;

			cout << "\nRepeat Count: " << repeatcount;
		}
		else
		{
			currentbest = Species[NoSpecies - 1].Population[Species[NoSpecies - 1].BestIndex].Fitness;
			repeatcount = 0;
		}



	}

	//---------------------------------------------------
	//species are subpopulations?
	acceptout << "Accept: " << naccept << "/" << numsamples;
}


void CoEvolution::EvolveSubPopulationsBCNE(int repetitions, double h, NeuralNetwork network, TrainingExamples Samples, double mutation, int depth, ofstream & out1, int Island)
{
	VectorManipulation vectormanipulate;

	double tempfit;
	double tempfitWorst;
	int count = 0;
	int tag = 99999;
	kid = KIDS;

	MCMC bayesian;

	//species are subpopulations?
	for (int s = 0; s < NoSpecies; s++) {
		if ((rand() % 100) < (h * 100)) {
			for (int r = 0; r < repetitions; r++) {
				if (NotConverged[s]) {
					tempfit = Species[s].Population[Species[s].BestIndex].Fitness; //get fitness of fittest individuals
					Species[s].kids = KIDS;
					Species[s].RandomParents();//select random parents for mating
											   //2 parents randomly selected

											   //Get fitter parent
					GetBestTable(s);
					Join();

					OneDVector_Double fitterparent = Individual;


					//2 kids
					for (int i = 0; i < Species[s].kids; i++)
					{
						bool bayesianvetted = false;

						int count = 0;
						while (bayesianvetted == false)
						{
							if (count > 0)
							{
								cout << "\nRejected Child. Count : " << count;
							}

							if (count > 10)
							{
								cout << "\n  Quitting Run" << count;
								bayesianvetted = true;
								break;
							}


							//Modified 11/04/2017 9:15 PM Assertation fail
							tag = Species[s].GenerateNewPCX(i, network, Samples, mutation, depth); //generate a child using PCX

							GetBestTable(s);
							for (int o = 0; o < Species[s].NumVariable; o++)
								TableSp[s].SingleSp[o] = Species[s].NewPop[i].Chrome[o];
							Join();

							OneDVector_Double child = Individual;

							bayesianvetted = bayesian.ExecuteVetting(fitterparent, child, h);

							count = count + 1;
						}

						if (tag == 0) break;
					}
					if (tag == 0) {
						NotConverged[s] = false;
					}

					for (int i = 0; i < Species[s].kids; i++)
						EvalNewPop(i, s, network, Samples, Island);//evaluate new sub-population for fitness

					find_parents(s, network, Samples, Island);  // form a pool from which a solution is to be
																//   replaced by the created child
					Species[s].sort();          // sort the kids+parents by fitness
					rep_parents(s, network, Samples, Island);   // a chosen parent is replaced by the child

					Species[s].BestIndex = 0;
					Species[s].WorstIndex = 0;

					tempfit = Species[s].Population[0].Fitness;
					tempfitWorst = Species[s].Population[0].Fitness;

					for (int i = 1; i < PopSize; i++) {

						if ((MINIMIZE *    Species[s].Population[i].Fitness) < (MINIMIZE * tempfit))
						{
							tempfit = Species[s].Population[i].Fitness;
							Species[s].BestFit = tempfit;
							Species[s].BestIndex = i;//update fittest
						}

						if ((MINIMIZE *    Species[s].Population[i].Fitness) > (MINIMIZE * tempfitWorst))
						{
							tempfitWorst = Species[s].Population[i].Fitness;
							Species[s].WorstFit = tempfitWorst;
							Species[s].WorstIndex = i;//update fittest
						}

					}

					GetBestTable(s);//update best species
					Join();//concatenate into one
				}
			}
		}
	}
}

void CoEvolution::EvolveSubPopulationsBCNESteps(int repetitions, double h, NeuralNetwork network, TrainingExamples Samples, double mutation, int depth, ofstream & out1, int Island)
{
	VectorManipulation vectormanipulate;

	double tempfit;
	double tempfitWorst;

	int tag = 99999;
	kid = KIDS;

	MCMC bayesian;

	//species are subpopulations?
	for (int s = 0; s < NoSpecies; s++) {
		if ((rand() % 100) < (h * 100)) {
			for (int r = 0; r < repetitions; r++) {
				if (NotConverged[s]) {
					tempfit = Species[s].Population[Species[s].BestIndex].Fitness; //get fitness of fittest individuals
					Species[s].kids = KIDS;
					Species[s].RandomParents();//select random parents for mating
											   //2 parents randomly selected

											   //Get fitter parent
					GetBestTable(s);
					Join();

					OneDVector_Double fitterparent = Individual;

					cout << "\n S: " << s;
					//2 kids
					for (int i = 0; i < Species[s].kids; i++)
					{


						if (s == 1) {
							cout << "";
						}


						//Modified 11/04/2017 9:15 PM Assertation fail
						tag = Species[s].GenerateNewPCX(i, network, Samples, mutation, depth); //generate a child using PCX

						GetBestTable(s);
						for (int o = 0; o < Species[s].NumVariable; o++)
							TableSp[s].SingleSp[o] = Species[s].NewPop[i].Chrome[o];
						Join();

						//Get start index, end index
						int index = 0;
						int startindex = 0;
						int endindex = 0;


						for (int row = 0; row < NoSpecies; row++) {
							for (int col = 0; col < SpeciesSize[row]; col++) {

								if (row == s && col == 0)
								{
									startindex = index;
								}
								else if (row == s && col == (SpeciesSize[row] - 1))
								{
									endindex = index;
									break;
								}

								index++;
							}
						}

						OneDVector_Double sampledchildindividual = bayesian.ExecuteVettingSteps(fitterparent, Individual, h, 1, startindex, endindex);
						//From individual get back species string

						//split individual get back offspring
						OneDVector_Double sampledchild = SplitIndividualGetBackOffspring(sampledchildindividual, s);

						//replace new offsping with sampled
						for (int o = 0; o < Species[s].NumVariable; o++)
							Species[s].NewPop[i].Chrome[o] = sampledchild[o];



						if (tag == 0) break;
					}
					if (tag == 0) {
						NotConverged[s] = false;
					}

					for (int i = 0; i < Species[s].kids; i++)
						EvalNewPop(i, s, network, Samples, Island);//evaluate new sub-population for fitness

					find_parents(s, network, Samples, Island);  // form a pool from which a solution is to be
																//   replaced by the created child
					Species[s].sort();          // sort the kids+parents by fitness
					rep_parents(s, network, Samples, Island);   // a chosen parent is replaced by the child

					Species[s].BestIndex = 0;
					Species[s].WorstIndex = 0;

					tempfit = Species[s].Population[0].Fitness;
					tempfitWorst = Species[s].Population[0].Fitness;

					for (int i = 1; i < PopSize; i++) {

						if ((MINIMIZE *    Species[s].Population[i].Fitness) < (MINIMIZE * tempfit))
						{
							tempfit = Species[s].Population[i].Fitness;
							Species[s].BestFit = tempfit;
							Species[s].BestIndex = i;//update fittest
						}

						if ((MINIMIZE *    Species[s].Population[i].Fitness) > (MINIMIZE * tempfitWorst))
						{
							tempfitWorst = Species[s].Population[i].Fitness;
							Species[s].WorstFit = tempfitWorst;
							Species[s].WorstIndex = i;//update fittest
						}

					}

					GetBestTable(s);//update best species
					Join();//concatenate into one
				}
			}
		}
	}
}

void CoEvolution::EvolveSubPopulationsBCNEGenerateOnly(int repetitions, double h, NeuralNetwork network, TrainingExamples Samples, double mutation, int depth, ofstream & out1, int Island)
{
	VectorManipulation vectormanipulate;

	double tempfit;
	double tempfitWorst;

	int tag = 99999;
	kid = KIDS;

	MCMC bayesian;

	//species are subpopulations?
	for (int s = 0; s < NoSpecies; s++) {
		if ((rand() % 100) < (h * 100)) {
			for (int r = 0; r < repetitions; r++) {
				if (NotConverged[s]) {
					tempfit = Species[s].Population[Species[s].BestIndex].Fitness; //get fitness of fittest individuals

					Species[s].kids = KIDS;
					Species[s].RandomParents();//select random parents for mating
											   //2 parents randomly selected

					//Get single parent as best
					GetBestTable(s);
					Join();

					OneDVector_Double fitterparent = Individual;


					//2 kids
					for (int i = 0; i < Species[s].kids; i++)
					{
						//Get start index, end index
						int index = 0;
						int startindex = 0;
						int endindex = 0;


						for (int row = 0; row < NoSpecies; row++) {
							for (int col = 0; col < SpeciesSize[row]; col++) {

								if (row == s && col == 0)
								{
									startindex = index;
								}
								else if (row == s && col == (SpeciesSize[row] - 1))
								{
									endindex = index;
									break;
								}

								index++;
							}
						}

						OneDVector_Double sampledchildindividual = bayesian.ExecuteGenerateOnly(fitterparent, h, 1, startindex, endindex);
						//From individual get back species string

						//split individual get back offspring
						OneDVector_Double sampledchild = SplitIndividualGetBackOffspring(sampledchildindividual, s);

						//replace new offsping with sampled
						for (int o = 0; o < Species[s].NumVariable; o++)
							Species[s].NewPop[i].Chrome[o] = sampledchild[o];



						if (tag == 0) break;
					}
					if (tag == 0) {
						NotConverged[s] = false;
					}

					for (int i = 0; i < Species[s].kids; i++)
						EvalNewPop(i, s, network, Samples, Island);//evaluate new sub-population for fitness

					find_parents(s, network, Samples, Island);  // form a pool from which a solution is to be
																//   replaced by the created child
					Species[s].sort();          // sort the kids+parents by fitness
					rep_parents(s, network, Samples, Island);   // a chosen parent is replaced by the child

					Species[s].BestIndex = 0;
					Species[s].WorstIndex = 0;

					tempfit = Species[s].Population[0].Fitness;
					tempfitWorst = Species[s].Population[0].Fitness;

					for (int i = 1; i < PopSize; i++) {

						if ((MINIMIZE *    Species[s].Population[i].Fitness) < (MINIMIZE * tempfit))
						{
							tempfit = Species[s].Population[i].Fitness;
							Species[s].BestFit = tempfit;
							Species[s].BestIndex = i;//update fittest
						}

						if ((MINIMIZE *    Species[s].Population[i].Fitness) > (MINIMIZE * tempfitWorst))
						{
							tempfitWorst = Species[s].Population[i].Fitness;
							Species[s].WorstFit = tempfitWorst;
							Species[s].WorstIndex = i;//update fittest
						}

					}

					GetBestTable(s);//update best species
					Join();//concatenate into one
				}
			}
		}
	}
}

/*
--CoEvolution--
*/
class CombinedEvolution : public    CoEvolution
{
public:
	//class variables
	int TotalEval;
	int TotalSize;
	double Train;
	double TrainNMSE;
	double TestNMSE;
	double TrainRMSE;
	double TestRMSE;
	double ValidRMSE;
	double ValidNMSE;
	double Test;
	double Valid;
	double Error;
	double TimeElapsed;
	//Layer testing_result;
	CoEvolution NeuronLevel;
	CoEvolution BayesianCoevolution;
	CoEvolution BPCoevolution;
	//CoEvolution WeightLevel;
	//CoEvolution OneLevel;
	int Cycles;
	bool Sucess;

	CombinedEvolution() {
		//constructor
	}
	//class functions
	int GetEval() {
		return TotalEval;
	}

	int GetCycle() {
		return Cycles;
	}

	double GetError() {
		return Test;
	}

	double NMSETrain() {
		return TrainNMSE;
	}

	double NMSETest() {
		return TestNMSE;
	}

	double getRMSETest() {
		return TestRMSE;
	}
	void setRMSETest(double result) {
		TestRMSE = result;
	}

	double getRMSETrain() {
		return TrainRMSE;
	}
	void setRMSETrain(double result) {
		TrainRMSE = result;
	}
	//validation
	double getRMSEValid() {
		return ValidRMSE;
	}
	double setRMSEValid(double result) {
		ValidRMSE = result;
	}

	double ValidNMSError() {
		return ValidNMSE;
	}
	bool GetSucess() {
		return Sucess;
	}

	//get/set time variables
	double getTIME()
	{
		return TimeElapsed;
	}

	void setTIME(double result)
	{
		TimeElapsed = result;
	}
	void   SaveBCNEData(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout, OneDVector_Double w);

	void   GSSpikesCooperativeNeuroEvolution(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout);
	void   LocalCooperativeNeuroEvolution2(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout);
	void   LocalCooperativeNeuroEvolution(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout);
	void   CCMCMC(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout,ofstream &trainout, ofstream &testout, ofstream &weightsout, ofstream &rmsetrainout, ofstream &rmseestout, ofstream &acceptout);
	void   BayesianCooperativeNeuroEvolution(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout);
	void   ProcedureBayesianCNE1Step(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout);
	void   ProcedureBayesianCNE1GenerateOnly(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout);
	void   ProcedureSGD(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout, ofstream &trainout, ofstream &testout, ofstream &weightsout);
	void   ProcedureBayesianCNE(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout);
	void   ProcedureBayesianMemetic(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout, ofstream &trainout, ofstream &testout, ofstream &weightsout);
	void   ProcedureBayesianMemetic_MCNEINMCMC(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout, ofstream &trainout, ofstream &testout, ofstream &weightsout);
	void   ProcedureMCNE(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout, ofstream &trainout, ofstream &testout, ofstream &weightsout);

	void   ProcedureCC(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout);
	void   ProcedureMemory(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout);
	void   ProcedureLSMaster(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout);
};







/*
--Procedure--
The entire algorithm
*/
void    CombinedEvolution::ProcedureLSMaster(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	ofstream out;
	out.open("Rnuuu.txt");

	//for( int row = 0; row< 50 ; row++)
	//	    testing_result.push_back(0);//initialize with 0s

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, Samples);//setup neural network

											  //bp = true;

	if (bp) {
		gene = network.BackPropogation(Samples, 0.1, layersize, file, true); //use backpropogation
	}
	else {
		// ############################################################ BCICN Code #######################################################//

		Sucess = false;
		Cycles = 0;

		// >>>>>>>>>>>>>>>>> Initialize Neuron Level Island <<<<<<<<<<<< //

		for (int n = 0; n < hidden; n++)
			NeuronLevel.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

		for (int n = 0; n < output; n++)
			NeuronLevel.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

		NeuronLevel.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).
		NeuronLevel.InitializeSpecies(CCPOPSIZE); //Generate random populations for each species.
		NeuronLevel.EvaluateSpecies(network, Samples, 1); //Get fitness of populations in species

		cout << "Hidden: " << h << " Total Sub-pops: " << NeuronLevel.NoSpecies << endl;

		for (int s = 0; s < NeuronLevel.NoSpecies; s++)
			NeuronLevel.NotConverged[s] = true;

		cout << " Evaluated Neuronlevel ----------->" << endl;


		TotalEval = 0;
		NeuronLevel.TotalEval = 0;


		int NeuronTempEval = 0;

		double BestWL, BestLL, BestNetL;

		int total_epoch = 0;

		int count = 0;
		Layer ErrorArray;
		Layer ValidRMSEArray;

		//while (NeuronLevel.TotalEval < 200) //Run CC for 1000 iterations
		//{

		//	NeuronLevel.EvolveSubPopulations(1, 1, network, Samples, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
		//	NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
		//	NeuronLevel.Join();//join the species together
		//	network.ChoromesToNeurons(NeuronLevel.Individual, 1);//encode into neural network the best cc individual
		//	network.SaveLearnedData(layersize, file);//save the neural network weights
		//	Error = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].Population[NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex].Fitness;

		//	ErrorArray.push_back(Error);
		//	//TotalEval += NeuronLevel.TotalEval;

		//	ecout << Error << "\n";

		//	out1 << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with
		//	cout << TotalEval << "/" << maxgen << " H: " << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with



		//	count++;

		//}

		//while the maximum number of function evaluations is not reached
		while (TotalEval <= maxgen)
		{
			//-- Memetic Framework

			//-- 1. Run Master - CCNL for 200 generations


			//best cc individual already encoded into the network. Just run BP using existing network weights
			gene = network.BackPropogation(Samples, 0.1, layersize, file, true); //use backpropogation
			TotalEval += NeuronLevel.TotalEval + gene;

			double LSError = network.SumSquaredError(Samples, layersize);
			bool guided = false;

			//Pass to CC

			//Blind Guidance - Replace Best Only ################################################################################
			cout << "Transferring\n";

			NeuronLevel.Individual = network.Neurons_to_chromes(1); //Set neuron level individual from current network individual

			int index = 0;

			for (int row = 0; row < NeuronLevel.NoSpecies; row++)
			{
				for (int col = 0; col < NeuronLevel.SpeciesSize[row]; col++)
				{
					NeuronLevel.TableSp[row].SingleSp[col] = NeuronLevel.Individual[index];
					index++;
				}
			}

			//---------------------------------------------------------------------------------------------------
			//transfer to best only
			/*int Best = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex;


			for (int sN = 0; sN < NeuronLevel.NoSpecies - 1; sN++)
			{
			for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
			NeuronLevel.Species[sN].Population[Best].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			NeuronLevel.Species[sN].BestIndex = Best;
			}

			for (int sN = NeuronLevel.NoSpecies - 1; sN < NeuronLevel.NoSpecies; sN++)
			{
			for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
			NeuronLevel.Species[sN].Population[Best].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			NeuronLevel.Species[sN].BestIndex = Best;
			}*/

			//---------------------------------------------------------------------------------------------------
			//transfer to worst only
			int Worst = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].WorstIndex;


			for (int sN = 0; sN < NeuronLevel.NoSpecies - 1; sN++)
			{
				for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
					NeuronLevel.Species[sN].Population[Worst].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
				NeuronLevel.Species[sN].BestIndex = Worst;
			}

			for (int sN = NeuronLevel.NoSpecies - 1; sN < NeuronLevel.NoSpecies; sN++)
			{
				for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
					NeuronLevel.Species[sN].Population[Worst].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
				NeuronLevel.Species[sN].BestIndex = Worst;
			}

			//randomly choose 5 spots and replace
			//for each species
			//			for (int sN = 0; sN < NeuronLevel.NoSpecies - 1; sN++)
			//			{
			//				//randomly replace 50 individuals in population
			//
			//				for (int p = 0; p < 5; p++) //replace 5 spots
			//				{
			//					for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
			//						NeuronLevel.Species[sN].Population[rand() % CCPOPSIZE].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			//				}
			//
			//
			//			}
			//
			//			for (int sN = NeuronLevel.NoSpecies - 1; sN < NeuronLevel.NoSpecies; sN++)
			//			{
			//
			//				for (int p = 0; p < 5; p++) //replace 5 spots
			//				{
			//					for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
			//						NeuronLevel.Species[sN].Population[rand() % CCPOPSIZE].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			//				}
			//			}

			//network.ChoromesToNeurons(OneLevel.Individual, 2);
			//network.SaveLearnedData(layersize, file); //Save learned data for test

			for (int i = 0; i < 20; i++)
			{

				NeuronLevel.EvolveSubPopulations(1, 1, network, Samples, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
				NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
				NeuronLevel.Join();//join the species together
				network.ChoromesToNeurons(NeuronLevel.Individual, 1);//encode into neural network the best cc individual
				network.SaveLearnedData(layersize, file);//save the neural network weights
				Error = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].Population[NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex].Fitness;

				ErrorArray.push_back(Error);
				//TotalEval += NeuronLevel.TotalEval;

				//ecout << Error << "\n";

				out1 << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with
				cout << TotalEval << "/" << maxgen << " H: " << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with



				count++;

			}

			//monitor test and validation with time
			double validerror = 0;
			double testerror = 0;
			validerror = network.TestTrainingData(layersize, file, validsize, validfile, input, output, out2);
			testerror = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
			ecout << validerror << "," << testerror << "\n";



			//}

			//-- 2. Run Slave LS for 200 generations


			//-- 3. Guidance - Either blind or guided


			//################################################################################################################################
			//guided = true;

			//if (guided)
			//{
			//	//Guided - Replace Best Only ################################################################################

			//	if (LSError < Error)
			//	{
			//		cout << "Transferring\n";
			//		NeuronLevel.Individual = network.Neurons_to_chromes(1); //Set neuron level individual from current network individual

			//		int index = 0;

			//		for (int row = 0; row < NeuronLevel.NoSpecies; row++)
			//		{
			//			for (int col = 0; col < NeuronLevel.SpeciesSize[row]; col++)
			//			{
			//				NeuronLevel.TableSp[row].SingleSp[col] = NeuronLevel.Individual[index];
			//				index++;
			//			}
			//		}

			//		int Best = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex;

			//		for (int sN = 0; sN < NeuronLevel.NoSpecies - 1; sN++)
			//		{
			//			for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
			//				NeuronLevel.Species[sN].Population[Best].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			//			NeuronLevel.Species[sN].BestIndex = Best;
			//		}

			//		for (int sN = NeuronLevel.NoSpecies - 1; sN < NeuronLevel.NoSpecies; sN++)
			//		{
			//			for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
			//				NeuronLevel.Species[sN].Population[Best].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			//			NeuronLevel.Species[sN].BestIndex = Best;
			//		}

			//	}
			//	else 
			//	{
			//		//LS did not improve anything
			//		//revert
			//		network.ChoromesToNeurons(NeuronLevel.Individual, 1);//encode into neural network the best from CCNL

			//		cout << "Reverting\n";

			//	}

			//	//network.ChoromesToNeurons(OneLevel.Individual, 2);
			//	network.SaveLearnedData(layersize, file); //Save learned data for test

			//	//End Guided ########################################################################################################
			//}
			//else
			//{
			//	//Blind Guidance - Replace Best Only ################################################################################
			//	cout << "Transferring\n";

			//	NeuronLevel.Individual = network.Neurons_to_chromes(1); //Set neuron level individual from current network individual

			//	int index = 0;

			//	for (int row = 0; row < NeuronLevel.NoSpecies; row++)
			//	{
			//		for (int col = 0; col < NeuronLevel.SpeciesSize[row]; col++)
			//		{
			//			NeuronLevel.TableSp[row].SingleSp[col] = NeuronLevel.Individual[index];
			//			index++;
			//		}
			//	}

			//	int Best = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex;

			//	for (int sN = 0; sN < NeuronLevel.NoSpecies - 1; sN++)
			//	{
			//		for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
			//			NeuronLevel.Species[sN].Population[Best].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			//		NeuronLevel.Species[sN].BestIndex = Best;
			//	}

			//	for (int sN = NeuronLevel.NoSpecies - 1; sN < NeuronLevel.NoSpecies; sN++)
			//	{
			//		for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
			//			NeuronLevel.Species[sN].Population[Best].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			//		NeuronLevel.Species[sN].BestIndex = Best;
			//	}

			//	//network.ChoromesToNeurons(OneLevel.Individual, 2);
			//	network.SaveLearnedData(layersize, file); //Save learned data for test

			//	//End Blind ########################################################################################################
			//}

			//################################################################################################################################

			out1 << "---------------------------------------" << endl;

			if (TotalEval > 1000)
			{
				//run the validation test
				//test the neural network with validation dataset

				Valid = network.TestTrainingData(layersize, file, validsize, validfile, input, output, out2);
				ValidRMSEArray.push_back(Valid);
				vcout << hidden << " nl " << "Valid: " << Valid << "     Error: " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;


				/*if (Valid < MinimumError)
				{
				vcout << "Break Training" << endl;
				vcout << "NeuronLevel.TotalEval " << NeuronLevel.TotalEval << endl;
				Sucess = true;
				//break;
				}*/
			}

			/*	if (Sucess == true)
			{
			vcout << "Break" << endl;
			break;
			}*/
			//	break;
		}
		//ValidRMSEArray.empty();

		out2 << "Train" << endl;
		//test the neural network with training data
		Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
		setRMSETrain(Train);
		TrainNMSE = network.NMSError();

		out2 << "Test" << endl;
		//test the neural network and the learnt data with the testing data set
		Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
		setRMSETest(Test);
		TestNMSE = network.NMSError();

		//clock stops here - execution ends here
		clock_t stop = clock();
		//cout<<"clock stop  "<<stop<<endl;

		//gives time in seconds
		double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
		setTIME(elapsed);

		out2 << endl;
		out1 << endl;
		vcout << endl;

		//ouput training accuracy for each evaluation
		out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
		//output training and test accuracy for file in which we have the actual vs predicted comparison
		out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
		//summary of one experimental run...RMSE and NMSE
		out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << endl;
	}
}
void    CombinedEvolution::ProcedureMemory(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	ofstream out;
	out.open("Rnuuu.txt");

	//for( int row = 0; row< 50 ; row++)
	//	    testing_result.push_back(0);//initialize with 0s

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, Samples);//setup neural network

	vector<vector<double> > memorySnapShots;


	/*memorySnapShots.resize(RandParent, vector<double>(NumVariable, 0));*/


	bp = true;

	if (bp) {
		gene = network.BackPropogation(Samples, 0.1, layersize, file, true); //use backpropogation
	}
	else {
		// ############################################################ BCICN Code #######################################################//

		Sucess = false;
		Cycles = 0;

		// >>>>>>>>>>>>>>>>> Initialize Neuron Level Island <<<<<<<<<<<< //

		for (int n = 0; n < hidden; n++)
			NeuronLevel.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

		for (int n = 0; n < output; n++)
			NeuronLevel.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

		NeuronLevel.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).
		NeuronLevel.InitializeSpecies(CCPOPSIZE); //Generate random populations for each species.
		NeuronLevel.EvaluateSpecies(network, Samples, 1); //Get fitness of populations in species

		cout << "Hidden: " << h << " Total Sub-pops: " << NeuronLevel.NoSpecies << endl;

		for (int s = 0; s < NeuronLevel.NoSpecies; s++)
			NeuronLevel.NotConverged[s] = true;

		cout << " Evaluated Neuronlevel ----------->" << endl;


		TotalEval = 0;
		NeuronLevel.TotalEval = 0;


		int NeuronTempEval = 0;

		double BestWL, BestLL, BestNetL;

		int total_epoch = 0;

		int count = 0;
		Layer ErrorArray;
		Layer ValidRMSEArray;

		//while (NeuronLevel.TotalEval < 200) //Run CC for 1000 iterations
		//{

		//	NeuronLevel.EvolveSubPopulations(1, 1, network, Samples, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
		//	NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
		//	NeuronLevel.Join();//join the species together
		//	network.ChoromesToNeurons(NeuronLevel.Individual, 1);//encode into neural network the best cc individual
		//	network.SaveLearnedData(layersize, file);//save the neural network weights
		//	Error = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].Population[NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex].Fitness;

		//	ErrorArray.push_back(Error);
		//	//TotalEval += NeuronLevel.TotalEval;

		//	ecout << Error << "\n";

		//	out1 << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with
		//	cout << TotalEval << "/" << maxgen << " H: " << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with



		//	count++;

		//}

		//while the maximum number of function evaluations is not reached
		while (TotalEval <= maxgen)
		{
			for (int i = 0; i < 10; i++)
			{

				NeuronLevel.EvolveSubPopulations(1, 1, network, Samples, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
				NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
				NeuronLevel.Join();//join the species together
				network.ChoromesToNeurons(NeuronLevel.Individual, 1);//encode into neural network the best cc individual
				network.SaveLearnedData(layersize, file);//save the neural network weights
				Error = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].Population[NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex].Fitness;

				ErrorArray.push_back(Error);
				TotalEval += NeuronLevel.TotalEval;

				//ecout << Error << "\n";

				out1 << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with
				cout << TotalEval << "/" << maxgen << " H: " << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with



				count++;



			}

			//########### Create snapshot ###########//
			cout << "Saving snapshot" << endl;
			vector<double> snapShot(NeuronLevel.TotalSize);
			network.SaveLearnedDataMemory(layersize, file, &snapShot);
			memorySnapShots.push_back(snapShot);
			//#######################################//


			//monitor test and validation with time
			/*double validerror = 0;
			double testerror = 0;
			validerror = network.TestTrainingData(layersize, file, validsize, validfile, input, output, out2);
			testerror = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
			ecout << validerror << "," << testerror << "\n";*/





			////best cc individual already encoded into the network. Just run BP using existing network weights
			//gene = network.BackPropogation(Samples, 0.1, layersize, file, true); //use backpropogation
			//TotalEval += NeuronLevel.TotalEval + gene;

			//double LSError = network.SumSquaredError(Samples, layersize);
			//bool guided = false;

			////Pass to CC

			////Blind Guidance - Replace Best Only ################################################################################
			//cout << "Transferring\n";

			//NeuronLevel.Individual = network.Neurons_to_chromes(1); //Set neuron level individual from current network individual

			//int index = 0;

			//for (int row = 0; row < NeuronLevel.NoSpecies; row++)
			//{
			//	for (int col = 0; col < NeuronLevel.SpeciesSize[row]; col++)
			//	{
			//		NeuronLevel.TableSp[row].SingleSp[col] = NeuronLevel.Individual[index];
			//		index++;
			//	}
			//}

			//---------------------------------------------------------------------------------------------------
			//transfer to best only
			/*int Best = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex;


			for (int sN = 0; sN < NeuronLevel.NoSpecies - 1; sN++)
			{
			for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
			NeuronLevel.Species[sN].Population[Best].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			NeuronLevel.Species[sN].BestIndex = Best;
			}

			for (int sN = NeuronLevel.NoSpecies - 1; sN < NeuronLevel.NoSpecies; sN++)
			{
			for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
			NeuronLevel.Species[sN].Population[Best].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			NeuronLevel.Species[sN].BestIndex = Best;
			}*/

			//---------------------------------------------------------------------------------------------------
			//transfer to worst only
			/*int Worst = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].WorstIndex;


			for (int sN = 0; sN < NeuronLevel.NoSpecies - 1; sN++)
			{
			for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
			NeuronLevel.Species[sN].Population[Worst].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			NeuronLevel.Species[sN].BestIndex = Worst;
			}

			for (int sN = NeuronLevel.NoSpecies - 1; sN < NeuronLevel.NoSpecies; sN++)
			{
			for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
			NeuronLevel.Species[sN].Population[Worst].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			NeuronLevel.Species[sN].BestIndex = Worst;
			}
			*/

			out1 << "---------------------------------------" << endl;

			if (TotalEval > 1000)
			{
				//run the validation test
				//test the neural network with validation dataset

				Valid = network.TestTrainingData(layersize, file, validsize, validfile, input, output, out2);
				ValidRMSEArray.push_back(Valid);
				vcout << hidden << " nl " << "Valid: " << Valid << "     Error: " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;


				/*if (Valid < MinimumError)
				{
				vcout << "Break Training" << endl;
				vcout << "NeuronLevel.TotalEval " << NeuronLevel.TotalEval << endl;
				Sucess = true;
				//break;
				}*/
			}

			/*	if (Sucess == true)
			{
			vcout << "Break" << endl;
			break;
			}*/
			//	break;
		}

		//########### Go through all snapshots refine each one ###########//
		//Run LS on all best snapshots
		//We will not use exhaustive approach whereby we run each for 200 epochs etc
		//We have a base epoch count e.g every 50 epochs we will measure the gradient and 
		//and keep track of the highest gradient value. If a newly calculated gradient is less than previous at the 
		//selected epoch we break. This methd favors steep descent into optima and not gradual (slow) descent

		double currentBestGradient = 0;
		int currentBestGradientIndex = 0;
		double currentBestError = 99;
		int currentBestErrorIndex = 0;

		bcout << "Begin refinement of best individuals: " << memorySnapShots.size() << " snapshots in total" << endl;
		for (int i = 0; i < memorySnapShots.size(); i++)
		{
			/*	cout << "Refining Snapshot " << (i + 1) << endl;*/
			network.LoadSavedDataMemory(layersize, &memorySnapShots[i]);
			gene = network.BackPropogationMemory(Samples, 1, layersize, file, true, &currentBestGradient, &currentBestGradientIndex, i, &currentBestError, &currentBestErrorIndex, bcout); //use backpropogation
																																														   /*cout << "Saving Snapshot " << (i + 1) << endl;*/
			network.SaveLearnedDataMemory(layersize, file, &memorySnapShots[i]); //update memory snapshot


		}

		network.LoadSavedDataMemory(layersize, &memorySnapShots[currentBestErrorIndex]); //load from memory to network
		network.SaveLearnedData(layersize, file);//save the neural network weights for testing


												 ////Blind Guidance - Replace Best Only ################################################################################
		cout << "Transferring\n";

		NeuronLevel.Individual = network.Neurons_to_chromes(1); //Set neuron level individual from current network individual

		int index = 0;

		for (int row = 0; row < NeuronLevel.NoSpecies; row++)
		{
			for (int col = 0; col < NeuronLevel.SpeciesSize[row]; col++)
			{
				NeuronLevel.TableSp[row].SingleSp[col] = NeuronLevel.Individual[index];
				index++;
			}
		}

		//---------------------------------------------------------------------------------------------------
		//transfer to best only
		int Best = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex;


		for (int sN = 0; sN < NeuronLevel.NoSpecies - 1; sN++)
		{
			for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
				NeuronLevel.Species[sN].Population[Best].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			NeuronLevel.Species[sN].BestIndex = Best;
		}

		for (int sN = NeuronLevel.NoSpecies - 1; sN < NeuronLevel.NoSpecies; sN++)
		{
			for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
				NeuronLevel.Species[sN].Population[Best].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
			NeuronLevel.Species[sN].BestIndex = Best;
		}

		//Run GS again one last time on the best solution

		while (TotalEval <= maxgen)
		{
			for (int i = 0; i < 10; i++)
			{

				NeuronLevel.EvolveSubPopulations(1, 1, network, Samples, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
				NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
				NeuronLevel.Join();//join the species together
				network.ChoromesToNeurons(NeuronLevel.Individual, 1);//encode into neural network the best cc individual
				network.SaveLearnedData(layersize, file);//save the neural network weights
				Error = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].Population[NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex].Fitness;

				ErrorArray.push_back(Error);

			}

		}



		//#######################################//

		//ValidRMSEArray.empty();

		out2 << "Train" << endl;
		//test the neural network with training data
		Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
		setRMSETrain(Train);
		TrainNMSE = network.NMSError();

		out2 << "Test" << endl;
		//test the neural network and the learnt data with the testing data set
		Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
		setRMSETest(Test);
		TestNMSE = network.NMSError();

		bcout << "Best Index: " << currentBestGradientIndex << " Train Fitness RMSE: " << Train << " Test Fitness RMSE: " << Test << endl;

		//clock stops here - execution ends here
		clock_t stop = clock();
		//cout<<"clock stop  "<<stop<<endl;

		//gives time in seconds
		double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
		setTIME(elapsed);

		out2 << endl;
		out1 << endl;
		vcout << endl;

		//ouput training accuracy for each evaluation
		out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
		//output training and test accuracy for file in which we have the actual vs predicted comparison
		out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
		//summary of one experimental run...RMSE and NMSE
		out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << endl;
	}
}

void    CombinedEvolution::ProcedureCC(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	ofstream out;
	out.open("Rnuuu.txt");

	//for( int row = 0; row< 50 ; row++)
	//	    testing_result.push_back(0);//initialize with 0s

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, Samples);//setup neural network

	vector<vector<double> > memorySnapShots;


	if (bp) {
		gene = network.BackPropogation(Samples, 0.1, layersize, file, true); //use backpropogation
	}
	else {
		// ############################################################ BCICN Code #######################################################//

		Sucess = false;
		Cycles = 0;

		// >>>>>>>>>>>>>>>>> Initialize Neuron Level Island <<<<<<<<<<<< //

		for (int n = 0; n < hidden; n++)
			NeuronLevel.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

		for (int n = 0; n < output; n++)
			NeuronLevel.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

		NeuronLevel.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).
		NeuronLevel.InitializeSpecies(CCPOPSIZE); //Generate random populations for each species.
		NeuronLevel.EvaluateSpecies(network, Samples, 1); //Get fitness of populations in species

		cout << "Hidden: " << h << " Total Sub-pops: " << NeuronLevel.NoSpecies << endl;

		for (int s = 0; s < NeuronLevel.NoSpecies; s++)
			NeuronLevel.NotConverged[s] = true;

		cout << " Evaluated Neuronlevel ----------->" << endl;


		TotalEval = 0;
		NeuronLevel.TotalEval = 0;


		int NeuronTempEval = 0;

		double BestWL, BestLL, BestNetL;

		int total_epoch = 0;

		int count = 0;
		Layer ErrorArray;
		Layer ValidRMSEArray;

		//while the maximum number of function evaluations is not reached
		while (TotalEval <= maxgen)
		{
			for (int i = 0; i < 10; i++)
			{
				NeuronLevel.EvolveSubPopulations(1, 1, network, Samples, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
				NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
				NeuronLevel.Join();//join the species together
				network.ChoromesToNeurons(NeuronLevel.Individual, 1);//encode into neural network the best cc individual
				network.SaveLearnedData(layersize, file);//save the neural network weights
				Error = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].Population[NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex].Fitness;

				ErrorArray.push_back(Error);
				TotalEval += NeuronLevel.TotalEval;

				out1 << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with
				cout << TotalEval << "/" << maxgen << " H: " << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with

				count++;
			}


			/*if (TotalEval > 1000)
			{

			Valid = network.TestTrainingData(layersize, file, validsize, validfile, input, output, out2);
			ValidRMSEArray.push_back(Valid);
			vcout << hidden << " nl " << "Valid: " << Valid << "     Error: " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;
			}*/


		}



		out2 << "Train" << endl;
		//test the neural network with training data
		Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
		setRMSETrain(Train);
		TrainNMSE = network.NMSError();

		out2 << "Test" << endl;
		//test the neural network and the learnt data with the testing data set
		Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
		setRMSETest(Test);
		TestNMSE = network.NMSError();


		//clock stops here - execution ends here
		clock_t stop = clock();
		//cout<<"clock stop  "<<stop<<endl;

		//gives time in seconds
		double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
		setTIME(elapsed);

		out2 << endl;
		out1 << endl;
		vcout << endl;

		//ouput training accuracy for each evaluation
		out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
		//output training and test accuracy for file in which we have the actual vs predicted comparison
		out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
		//summary of one experimental run...RMSE and NMSE
		out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << endl;
	}
}

void    CombinedEvolution::SaveBCNEData(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout, OneDVector_Double w)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, Samples);//setup neural network

	network.ChoromesToNeurons(w, 1);//encode into neural network bcne weights
	network.SaveLearnedData(layersize, file);//save the neural network weights

	out2 << "Train" << endl;
	//test the neural network with training data
	Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
	setRMSETrain(Train);
	TrainNMSE = network.NMSError();

	out2 << "Test" << endl;
	//test the neural network and the learnt data with the testing data set
	Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
	setRMSETest(Test);
	TestNMSE = network.NMSError();

	cout << "\nTest RMSE: " << Test;


	//clock stops here - execution ends here
	clock_t stop = clock();
	//cout<<"clock stop  "<<stop<<endl;

	//gives time in seconds
	double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
	setTIME(elapsed);

	out2 << endl;
	out1 << endl;
	vcout << endl;

	//ouput training accuracy for each evaluation
	out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
	//output training and test accuracy for file in which we have the actual vs predicted comparison
	out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
	//summary of one experimental run...RMSE and NMSE
	out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << endl;
}



/*
This implementation sits at G3PCX. Whenever a child is generated it is vetted using random walk. The fitter parent
is used as the prior solution and the child the proposed solution.
*/

void    CombinedEvolution::ProcedureBayesianCNE1GenerateOnly(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	ofstream out;
	out.open("Rnuuu.txt");

	//for( int row = 0; row< 50 ; row++)
	//	    testing_result.push_back(0);//initialize with 0s

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, Samples);//setup neural network


	if (bp) {
		gene = network.BackPropogation(Samples, 0.1, layersize, file, true); //use backpropogation
	}
	else {

		Sucess = false;
		Cycles = 0;

		// >>>>>>>>>>>>>>>>> Initialize Neuron Level Island <<<<<<<<<<<< //

		for (int n = 0; n < hidden; n++)
			NeuronLevel.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

		for (int n = 0; n < output; n++)
			NeuronLevel.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

		NeuronLevel.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).
		NeuronLevel.InitializeSpecies(CCPOPSIZE); //Generate random populations for each species.
		NeuronLevel.EvaluateSpeciesBayesian(network, Samples, 1); //Get fitness of populations in species

		cout << "Hidden: " << h << " Total Sub-pops: " << NeuronLevel.NoSpecies << endl;

		for (int s = 0; s < NeuronLevel.NoSpecies; s++)
			NeuronLevel.NotConverged[s] = true;

		cout << " Evaluated Neuronlevel ----------->" << endl;


		TotalEval = 0;
		NeuronLevel.TotalEval = 0;


		int NeuronTempEval = 0;

		double BestWL, BestLL, BestNetL;

		int total_epoch = 0;

		int count = 0;
		Layer ErrorArray;
		Layer ValidRMSEArray;

		Error = 99;

		//while the maximum number of function evaluations is not reached
		while (TotalEval <= maxgen)
			/*while (Error >= 0.01)*/
		{
			for (int i = 0; i < 10; i++)
			{
				NeuronLevel.EvolveSubPopulationsBCNEGenerateOnly(1, 1, network, Samples, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
				NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
				NeuronLevel.Join();//join the species together
				network.ChoromesToNeurons(NeuronLevel.Individual, 1);//encode into neural network the best cc individual
				network.SaveLearnedData(layersize, file);//save the neural network weights
				Error = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].Population[NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex].Fitness;

				ErrorArray.push_back(Error);
				TotalEval += NeuronLevel.TotalEval;

				out1 << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with
				cout << "\n" << TotalEval << "/" << maxgen << " H: " << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with

				count++;
			}


			/*if (TotalEval > 1000)
			{

			Valid = network.TestTrainingData(layersize, file, validsize, validfile, input, output, out2);
			ValidRMSEArray.push_back(Valid);
			vcout << hidden << " nl " << "Valid: " << Valid << "     Error: " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;
			}*/


		}



		out2 << "Train" << endl;
		//test the neural network with training data
		Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
		setRMSETrain(Train);
		TrainNMSE = network.NMSError();

		out2 << "Test" << endl;
		//test the neural network and the learnt data with the testing data set
		Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
		setRMSETest(Test);
		TestNMSE = network.NMSError();

		cout << "\nTest RMSE: " << Test;


		//clock stops here - execution ends here
		clock_t stop = clock();
		//cout<<"clock stop  "<<stop<<endl;

		//gives time in seconds
		double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
		setTIME(elapsed);

		out2 << endl;
		out1 << endl;
		vcout << endl;

		//ouput training accuracy for each evaluation
		out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
		//output training and test accuracy for file in which we have the actual vs predicted comparison
		out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
		//summary of one experimental run...RMSE and NMSE
		out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << endl;
	}
}

/*
This implementation sits at G3PCX. Whenever a child is generated it is vetted using random walk. The fitter parent
is used as the prior solution and the child the proposed solution.
*/

void    CombinedEvolution::ProcedureBayesianCNE1Step(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	ofstream out;
	out.open("Rnuuu.txt");

	//for( int row = 0; row< 50 ; row++)
	//	    testing_result.push_back(0);//initialize with 0s

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, Samples);//setup neural network


	if (bp) {
		gene = network.BackPropogation(Samples, 0.1, layersize, file, true); //use backpropogation
	}
	else {

		Sucess = false;
		Cycles = 0;

		// >>>>>>>>>>>>>>>>> Initialize Neuron Level Island <<<<<<<<<<<< //

		for (int n = 0; n < hidden; n++)
			NeuronLevel.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

		for (int n = 0; n < output; n++)
			NeuronLevel.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

		NeuronLevel.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).
		NeuronLevel.InitializeSpecies(CCPOPSIZE); //Generate random populations for each species.
		NeuronLevel.EvaluateSpecies(network, Samples, 1); //Get fitness of populations in species

		cout << "Hidden: " << h << " Total Sub-pops: " << NeuronLevel.NoSpecies << endl;

		for (int s = 0; s < NeuronLevel.NoSpecies; s++)
			NeuronLevel.NotConverged[s] = true;

		cout << " Evaluated Neuronlevel ----------->" << endl;


		TotalEval = 0;
		NeuronLevel.TotalEval = 0;


		int NeuronTempEval = 0;

		double BestWL, BestLL, BestNetL;

		int total_epoch = 0;

		int count = 0;
		Layer ErrorArray;
		Layer ValidRMSEArray;

		Error = 99;

		//while the maximum number of function evaluations is not reached
		while (TotalEval <= maxgen)
			/*while (Error >= 0.01)*/
		{
			for (int i = 0; i < 10; i++)
			{
				NeuronLevel.EvolveSubPopulationsBCNESteps(1, 1, network, Samples, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
				NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
				NeuronLevel.Join();//join the species together
				network.ChoromesToNeurons(NeuronLevel.Individual, 1);//encode into neural network the best cc individual
				network.SaveLearnedData(layersize, file);//save the neural network weights
				Error = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].Population[NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex].Fitness;

				ErrorArray.push_back(Error);
				TotalEval += NeuronLevel.TotalEval;

				out1 << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with
				cout << "\n" << TotalEval << "/" << maxgen << " H: " << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with

				count++;
			}


			/*if (TotalEval > 1000)
			{

			Valid = network.TestTrainingData(layersize, file, validsize, validfile, input, output, out2);
			ValidRMSEArray.push_back(Valid);
			vcout << hidden << " nl " << "Valid: " << Valid << "     Error: " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;
			}*/


		}



		out2 << "Train" << endl;
		//test the neural network with training data
		Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
		setRMSETrain(Train);
		TrainNMSE = network.NMSError();

		out2 << "Test" << endl;
		//test the neural network and the learnt data with the testing data set
		Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
		setRMSETest(Test);
		TestNMSE = network.NMSError();

		cout << "\nTest RMSE: " << Test;


		//clock stops here - execution ends here
		clock_t stop = clock();
		//cout<<"clock stop  "<<stop<<endl;

		//gives time in seconds
		double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
		setTIME(elapsed);

		out2 << endl;
		out1 << endl;
		vcout << endl;

		//ouput training accuracy for each evaluation
		out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
		//output training and test accuracy for file in which we have the actual vs predicted comparison
		out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
		//summary of one experimental run...RMSE and NMSE
		out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << endl;
	}
}

void    CombinedEvolution::GSSpikesCooperativeNeuroEvolution(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	ofstream out;
	out.open("Rnuuu.txt");

	//for( int row = 0; row< 50 ; row++)
	//	    testing_result.push_back(0);//initialize with 0s

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	BPCoevolution.TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	int epochs = 300;

	//TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	//Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer


	Sucess = false;
	Cycles = 0;

	// >>>>>>>>>>>>>>>>> Initialize Neuron Level Island <<<<<<<<<<<< //

	for (int n = 0; n < hidden; n++)
		BPCoevolution.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

	for (int n = 0; n < output; n++)
		BPCoevolution.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

	BPCoevolution.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).

	BPCoevolution.InitializeSpecies(epochs); //Generate random populations for each species.

											 /* --------------------------------------------------------------------------------- */
	VectorManipulation vectormanipulate;

	TrainingExamples SamplesTrain(trainfile, trainsize, input + output, input, output); //get training data
	TrainingExamples SamplesTest(testfile, testsize, input + output, input, output); //get test data

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, SamplesTrain);//setup neural network

	int w_size = weightsize1 + weightsize2 + biasize;

	double currentbest = 99;
	double repeatcount = 0;
	bool prematureconverge = false;

	while (repeatcount < 10) //if same rmse hasnt improved in 20 cycles
	{
		TwoDVector_Double savedsamples = network.BackPropogationCC(SamplesTrain, 0.1, layersize, file, true, epochs, w_size); //use backpropogation

		cout << "Samples Size" << savedsamples.size() << " " << epochs;

		if (savedsamples.size() != epochs) {
			cout << "\nBP Ended - Quitting Run";
			prematureconverge = true;
			break;
		}
		//vectormanipulate.Print2DVector(savedsamples);

		BPCoevolution.UpdateSpeciesBaye(epochs, savedsamples);

		BPCoevolution.EvaluateSpecies(network, SamplesTrain, 1); //Get fitness of populations in species


																 //1 cycle evolution
		for (int i = 0; i < 20; i++)
		{
			cout << "\nCC Cycle: " << i;
			BPCoevolution.EvolveSubPopulations(1, 1, network, SamplesTrain, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
			BPCoevolution.GetBestTable(BPCoevolution.NoSpecies - 1);
			BPCoevolution.Join();//join the species together
			network.ChoromesToNeurons(BPCoevolution.Individual, 1);//encode into neural network the best cc individual
			network.SaveLearnedData(layersize, file);//save the neural network weights
		}




		/*	BPCoevolution.GetBestTable(BPCoevolution.NoSpecies - 1);
		BPCoevolution.Join();*/

		TotalEval += BPCoevolution.TotalEval;

		//Test train data
		//network.ChoromesToNeurons(BPCoevolution.Individual, 1);//encode into neural network the best cc individual
		//network.SaveLearnedData(layersize, file);//save the neural network weights

		//test the neural network with training data
		Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);

		cout << "\nPop Best RMSE: " << Train;


		if (Train > currentbest || Train == currentbest)
		{
			repeatcount++;

			cout << "\nRepeat Count: " << repeatcount;
		}
		else
		{
			currentbest = Train;
			repeatcount = 0;

		}


	}

	/*system("pause");*/
	//Update network weights. If premature convergence, use current weights in the network as the best.
	if (prematureconverge == false) {
		network.ChoromesToNeurons(BPCoevolution.Individual, 1);//encode into neural network the best cc individual
		network.SaveLearnedData(layersize, file);//save the neural network weights
	}


	out2 << "Train" << endl;
	//test the neural network with training data
	Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
	setRMSETrain(Train);
	TrainNMSE = network.NMSError();

	out2 << "Test" << endl;
	//test the neural network and the learnt data with the testing data set
	Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
	setRMSETest(Test);
	TestNMSE = network.NMSError();

	cout << "\nTest RMSE: " << Test;


	//clock stops here - execution ends here
	clock_t stop = clock();
	//cout<<"clock stop  "<<stop<<endl;

	//gives time in seconds
	double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
	setTIME(elapsed);

	out2 << endl;
	out1 << endl;
	vcout << endl;

	//ouput training accuracy for each evaluation
	out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
	//output training and test accuracy for file in which we have the actual vs predicted comparison
	out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
	//summary of one experimental run...RMSE and NMSE
	out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << endl;
}

void    CombinedEvolution::LocalCooperativeNeuroEvolution2(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	ofstream out;
	out.open("Rnuuu.txt");

	//for( int row = 0; row< 50 ; row++)
	//	    testing_result.push_back(0);//initialize with 0s

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	BPCoevolution.TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	int epochs = 150;

	//TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	//Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer


	Sucess = false;
	Cycles = 0;

	// >>>>>>>>>>>>>>>>> Initialize Neuron Level Island <<<<<<<<<<<< //

	for (int n = 0; n < hidden; n++)
		BPCoevolution.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

	for (int n = 0; n < output; n++)
		BPCoevolution.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

	BPCoevolution.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).

	BPCoevolution.InitializeSpecies(epochs * 2); //Generate random populations 150 * 2

											 /* --------------------------------------------------------------------------------- */
	VectorManipulation vectormanipulate;

	TrainingExamples SamplesTrain(trainfile, trainsize, input + output, input, output); //get training data
	TrainingExamples SamplesTest(testfile, testsize, input + output, input, output); //get test data

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, SamplesTrain);//setup neural network

	int w_size = weightsize1 + weightsize2 + biasize;

	double currentbest = 99;
	double repeatcount = 0;
	bool prematureconverge = false;

	while (repeatcount < 10) //if same rmse hasnt improved in 20 cycles
	{
		TwoDVector_Double savedsamples = network.BackPropogationCC(SamplesTrain, 0.1, layersize, file, true, epochs, w_size); //use backpropogation

		cout << "\nSamples Size" << savedsamples.size() << " " << epochs;

		if (savedsamples.size() != epochs) {
			cout << "\nBP Ended - Quitting Run";
			prematureconverge = true;
			break;
		}
		//vectormanipulate.Print2DVector(savedsamples);

		//BPCoevolution.UpdateSpeciesBaye(epochs, savedsamples);

		BPCoevolution.PrintBestIndexes(BPCoevolution.NoSpecies);

		BPCoevolution.UpdateSpeciesPartial(epochs, savedsamples);

		BPCoevolution.EvaluateSpecies(network, SamplesTrain, 1); //Get fitness of populations in species

		BPCoevolution.PrintBestIndexes(BPCoevolution.NoSpecies);
		//1 cycle evolution
		for (int i = 0; i < 20; i++)
		{
			cout << "\nCC Cycle: " << i;
			BPCoevolution.EvolveSubPopulations(1, 1, network, SamplesTrain, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
			BPCoevolution.GetBestTable(BPCoevolution.NoSpecies - 1);
			BPCoevolution.Join();//join the species together
			network.ChoromesToNeurons(BPCoevolution.Individual, 1);//encode into neural network the best cc individual
			network.SaveLearnedData(layersize, file);//save the neural network weights
		}

		//BPCoevolution.EvaluateSpecies(network, SamplesTrain, 1); //Get fitness of populations in species
		BPCoevolution.PrintBestIndexes(BPCoevolution.NoSpecies);

		BPCoevolution.SwapLocalBestToCCRegion(epochs);

		BPCoevolution.PrintBestIndexes(BPCoevolution.NoSpecies);

		/*	BPCoevolution.GetBestTable(BPCoevolution.NoSpecies - 1);
		BPCoevolution.Join();*/

		TotalEval += BPCoevolution.TotalEval;

		//Test train data
		//network.ChoromesToNeurons(BPCoevolution.Individual, 1);//encode into neural network the best cc individual
		//network.SaveLearnedData(layersize, file);//save the neural network weights

		//test the neural network with training data
		Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);

		cout << "\nPop Best RMSE: " << Train;


		if (Train > currentbest || Train == currentbest)
		{
			repeatcount++;

			cout << "\nRepeat Count: " << repeatcount;
		}
		else
		{
			currentbest = Train;
			repeatcount = 0;

		}


	}

	/*system("pause");*/
	//Update network weights. If premature convergence, use current weights in the network as the best.
	if (prematureconverge == false) {
		network.ChoromesToNeurons(BPCoevolution.Individual, 1);//encode into neural network the best cc individual
		network.SaveLearnedData(layersize, file);//save the neural network weights
	}


	out2 << "Train" << endl;
	//test the neural network with training data
	Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
	setRMSETrain(Train);
	TrainNMSE = network.NMSError();

	out2 << "Test" << endl;
	//test the neural network and the learnt data with the testing data set
	Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
	setRMSETest(Test);
	TestNMSE = network.NMSError();

	cout << "\nTest RMSE: " << Test;


	//clock stops here - execution ends here
	clock_t stop = clock();
	//cout<<"clock stop  "<<stop<<endl;

	//gives time in seconds
	double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
	setTIME(elapsed);

	out2 << endl;
	out1 << endl;
	vcout << endl;

	//ouput training accuracy for each evaluation
	out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
	//output training and test accuracy for file in which we have the actual vs predicted comparison
	out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
	//summary of one experimental run...RMSE and NMSE
	out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << endl;
}
/*
This implementation sits at G3PCX. Whenever a child is generated it is vetted using random walk. The fitter parent
is used as the prior solution and the child the proposed solution.
*/


void    CombinedEvolution::LocalCooperativeNeuroEvolution(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	ofstream out;
	out.open("Rnuuu.txt");

	//for( int row = 0; row< 50 ; row++)
	//	    testing_result.push_back(0);//initialize with 0s

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	BPCoevolution.TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	int epochs = 300;

	//TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	//Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer


	Sucess = false;
	Cycles = 0;

	// >>>>>>>>>>>>>>>>> Initialize Neuron Level Island <<<<<<<<<<<< //

	for (int n = 0; n < hidden; n++)
		BPCoevolution.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

	for (int n = 0; n < output; n++)
		BPCoevolution.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

	BPCoevolution.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).

	BPCoevolution.InitializeSpecies(epochs); //Generate random populations for each species.

	/* --------------------------------------------------------------------------------- */
	VectorManipulation vectormanipulate;

	TrainingExamples SamplesTrain(trainfile, trainsize, input + output, input, output); //get training data
	TrainingExamples SamplesTest(testfile, testsize, input + output, input, output); //get test data

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, SamplesTrain);//setup neural network

	int w_size = weightsize1 + weightsize2 + biasize;

	double currentbest = 99;
	double repeatcount = 0;
	bool prematureconverge = false;

	while (repeatcount < 10) //if same rmse hasnt improved in 20 cycles
	{
		TwoDVector_Double savedsamples = network.BackPropogationCC(SamplesTrain, 0.1, layersize, file, true, epochs, w_size); //use backpropogation

		cout << "Samples Size" << savedsamples.size() << " " << epochs;

		if (savedsamples.size() != epochs) {
			cout << "\nBP Ended - Quitting Run";
			prematureconverge = true;
			break;
		}
		//vectormanipulate.Print2DVector(savedsamples);

		BPCoevolution.UpdateSpeciesBaye(epochs, savedsamples);

		BPCoevolution.EvaluateSpecies(network, SamplesTrain, 1); //Get fitness of populations in species


		//1 cycle evolution
		for (int i = 0; i < 20; i++)
		{
			cout << "\nCC Cycle: " << i;
			BPCoevolution.EvolveSubPopulations(1, 1, network, SamplesTrain, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
			BPCoevolution.GetBestTable(BPCoevolution.NoSpecies - 1);
			BPCoevolution.Join();//join the species together
			network.ChoromesToNeurons(BPCoevolution.Individual, 1);//encode into neural network the best cc individual
			network.SaveLearnedData(layersize, file);//save the neural network weights
		}




		/*	BPCoevolution.GetBestTable(BPCoevolution.NoSpecies - 1);
			BPCoevolution.Join();*/

		TotalEval += BPCoevolution.TotalEval;

		//Test train data
		//network.ChoromesToNeurons(BPCoevolution.Individual, 1);//encode into neural network the best cc individual
		//network.SaveLearnedData(layersize, file);//save the neural network weights

												 //test the neural network with training data
		Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);

		cout << "\nPop Best RMSE: " << Train;


		if (Train > currentbest || Train == currentbest)
		{
			repeatcount++;

			cout << "\nRepeat Count: " << repeatcount;
		}
		else
		{
			currentbest = Train;
			repeatcount = 0;

		}


	}

	/*system("pause");*/
	//Update network weights. If premature convergence, use current weights in the network as the best.
	if (prematureconverge == false) {
		network.ChoromesToNeurons(BPCoevolution.Individual, 1);//encode into neural network the best cc individual
		network.SaveLearnedData(layersize, file);//save the neural network weights
	}


	out2 << "Train" << endl;
	//test the neural network with training data
	Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
	setRMSETrain(Train);
	TrainNMSE = network.NMSError();

	out2 << "Test" << endl;
	//test the neural network and the learnt data with the testing data set
	Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
	setRMSETest(Test);
	TestNMSE = network.NMSError();

	cout << "\nTest RMSE: " << Test;


	//clock stops here - execution ends here
	clock_t stop = clock();
	//cout<<"clock stop  "<<stop<<endl;

	//gives time in seconds
	double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
	setTIME(elapsed);

	out2 << endl;
	out1 << endl;
	vcout << endl;

	//ouput training accuracy for each evaluation
	out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
	//output training and test accuracy for file in which we have the actual vs predicted comparison
	out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
	//summary of one experimental run...RMSE and NMSE
	out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << endl;
}


void    CombinedEvolution::BayesianCooperativeNeuroEvolution(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	ofstream out;
	out.open("Rnuuu.txt");

	//for( int row = 0; row< 50 ; row++)
	//	    testing_result.push_back(0);//initialize with 0s

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	BayesianCoevolution.TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	int samples = 10;

	TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer


	Sucess = false;
	Cycles = 0;

	// >>>>>>>>>>>>>>>>> Initialize Neuron Level Island <<<<<<<<<<<< //

	for (int n = 0; n < hidden; n++)
		BayesianCoevolution.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

	for (int n = 0; n < output; n++)
		BayesianCoevolution.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

	BayesianCoevolution.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).

	BayesianCoevolution.InitializeSpecies(samples); //Generate random populations for each species.

	/* --------------------------------------------------------------------------------- */

	MCMC bayesian;
	VectorManipulation vectormanipulate;

	TrainingExamples SamplesTrain(trainfile, trainsize, input + output, input, output); //get training data
	TrainingExamples SamplesTest(testfile, testsize, input + output, input, output); //get test data

	OneDVector_Double y_train;
	OneDVector_Double y_test;

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, SamplesTrain);//setup neural network

	int w_size = weightsize1 + weightsize2 + biasize;
	/*TwoDVector_Double pos_w = vectormanipulate.Generate2DVector_Double(samples, w_size, 1);
	OneDVector_Double pos_tau = vectormanipulate.Generate1DVector_Double(samples, 1);

	TwoDVector_Double fxtrain_samples = vectormanipulate.Generate2DVector_Double(samples, trainsize, 1);
	TwoDVector_Double fxtest_samples = vectormanipulate.Generate2DVector_Double(samples, testsize, 1);

	OneDVector_Double rmse_train = vectormanipulate.Generate1DVector_Double(samples, 0);
	OneDVector_Double rmse_test = vectormanipulate.Generate1DVector_Double(samples, 0);*/

	OneDVector_Double w = bayesian.GenerateRandomWeights(w_size);
	OneDVector_Double w_proposal = bayesian.GenerateRandomWeights(w_size);

	vectormanipulate.Print1DVector(w);

	double step_w = 0.02;
	double step_eta = 0.01;



	OneDVector_Double pred_train = network.Evaluate_Proposal(SamplesTrain, w, layersize);
	OneDVector_Double pred_test = network.Evaluate_Proposal(SamplesTest, w, layersize);



	double eta = bayesian.GetEta(pred_train, SamplesTrain.GetOutputValues());
	double tau_pro = exp(eta);

	int sigma_squared = 25;
	int nu_1 = 0;
	int nu_2 = 0;


	double prior_likelihood = bayesian.Prior_Likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, h);

	LikelihoodReturnObject likelihood = bayesian.Likelihood_Func(network, SamplesTrain, w, tau_pro, layersize);
	cout << "Begin sampling using mcmc random walk";

	TwoDVector_Double savedsamples = vectormanipulate.Generate2DVector_Double(samples, w_size, 0);

	double currentbest = 99;
	double repeatcount = 0;


	while (repeatcount < 100) //100 Loops
	{


		int naccept = 0;
		//Create new w proposal
		while (naccept < (samples))
		{

			for (int y = 0; y < w_proposal.size(); y++)
			{
				w_proposal[y] = w[y] + rand_normal(0, step_w);
			}


			/*vectormanipulate.Print1DVector(w_proposal);*/

			double eta_pro = eta + rand_normal(0, step_eta);
			double tau_pro = exp(eta_pro);

			LikelihoodReturnObject likelihood_proposal = bayesian.Likelihood_Func(network, SamplesTrain, w_proposal, tau_pro, layersize);

			double prior_prop = bayesian.Prior_Likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, h);

			double diff_likelihood = likelihood_proposal.likelihood - likelihood.likelihood;
			double diff_priorliklihood = prior_prop - prior_likelihood;

			double  mh_prob = 1;

			if (exp(diff_likelihood + diff_priorliklihood) < 1) {
				mh_prob = exp(diff_likelihood + diff_priorliklihood);
			}

			/*	RandomNumber rn;*/
			double u = bayesian.randfrom(0, 1);

			if (u < mh_prob)
			{
				/*cout << "\n Size: " << savedsamples.size() << " naccept: " << naccept;*/
				savedsamples[naccept] = w;

				naccept += 1;
				likelihood = likelihood_proposal;

				cout << "\nRMSE: " << likelihood_proposal.rmse;

				prior_likelihood = prior_prop;
				w = w_proposal;
				eta = eta_pro;


			}

		}

		//vectormanipulate.Print2DVector(savedsamples);

		/* ---------------------------------------------------------------------------------------- */
		BayesianCoevolution.UpdateSpeciesBaye(samples, savedsamples);

		BayesianCoevolution.EvaluateSpecies(network, Samples, 1); //Get fitness of populations in species

		BayesianCoevolution.GetBestTable(BayesianCoevolution.NoSpecies - 1);
		BayesianCoevolution.Join();

		TotalEval += BayesianCoevolution.TotalEval;

		w = BayesianCoevolution.Individual;


		//Test train data
		network.ChoromesToNeurons(w, 1);//encode into neural network the best cc individual
		network.SaveLearnedData(layersize, file);//save the neural network weights

		//test the neural network with training data
		Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);

		cout << "\nPop Best RMSE: " << Train;


		/*if (Train > currentbest || Train == currentbest)
		{
			repeatcount++;
		}
		else
		{
			currentbest = Train;
			repeatcount = 0;

		}*/

		repeatcount++;

	}

	network.ChoromesToNeurons(w, 1);//encode into neural network the best cc individual
	network.SaveLearnedData(layersize, file);//save the neural network weights



	/*cout << "Hidden: " << h << " Total Sub-pops: " << BayesianCoevolution.NoSpecies << endl;

	for (int s = 0; s < BayesianCoevolution.NoSpecies; s++)
		BayesianCoevolution.NotConverged[s] = true;


	TotalEval = 0;
	BayesianCoevolution.TotalEval = 0;*/


	//

	//int count = 0;
	//Layer ErrorArray;
	//Layer ValidRMSEArray;

	//Error = 99;

	////while the maximum number of function evaluations is not reached
	//while (TotalEval <= maxgen)
	//	/*while (Error >= 0.01)*/
	//{

	//	

	//		//for (int i = 0; i < 10; i++)
	//		//{
	//		//	BayesianCoevolution.EvolveSubPopulationsBCNESteps(1, 1, network, Samples, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
	//		//	BayesianCoevolution.GetBestTable(BayesianCoevolution.NoSpecies - 1);
	//		//	BayesianCoevolution.Join();//join the species together
	//		//	network.ChoromesToNeurons(BayesianCoevolution.Individual, 1);//encode into neural network the best cc individual
	//		//	network.SaveLearnedData(layersize, file);//save the neural network weights
	//		//	Error = BayesianCoevolution.Species[BayesianCoevolution.NoSpecies - 1].Population[BayesianCoevolution.Species[BayesianCoevolution.NoSpecies - 1].BestIndex].Fitness;

	//		//	ErrorArray.push_back(Error);
	//		//	TotalEval += BayesianCoevolution.TotalEval;

	//		//	out1 << hidden << " nl " << Train << "    " << Error << "    " << BayesianCoevolution.TotalEval << "    " << count << endl;//shows how the RMSE is going down with
	//		//	cout << "\n" << TotalEval << "/" << maxgen << " H: " << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with

	//		//	count++;
	//		//}


	//		/*if (TotalEval > 1000)
	//		{

	//		Valid = network.TestTrainingData(layersize, file, validsize, validfile, input, output, out2);
	//		ValidRMSEArray.push_back(Valid);
	//		vcout << hidden << " nl " << "Valid: " << Valid << "     Error: " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;
	//		}*/


	//}



	out2 << "Train" << endl;
	//test the neural network with training data
	Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
	setRMSETrain(Train);
	TrainNMSE = network.NMSError();

	out2 << "Test" << endl;
	//test the neural network and the learnt data with the testing data set
	Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
	setRMSETest(Test);
	TestNMSE = network.NMSError();

	cout << "\nTest RMSE: " << Test;


	//clock stops here - execution ends here
	clock_t stop = clock();
	//cout<<"clock stop  "<<stop<<endl;

	//gives time in seconds
	double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
	setTIME(elapsed);

	out2 << endl;
	out1 << endl;
	vcout << endl;

	//ouput training accuracy for each evaluation
	out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
	//output training and test accuracy for file in which we have the actual vs predicted comparison
	out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
	//summary of one experimental run...RMSE and NMSE
	out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << endl;

}

void    CombinedEvolution::CCMCMC(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout, ofstream &trainout, ofstream &testout, ofstream &weightsout, ofstream &rmsetrainout, ofstream &rmsetestout, ofstream &acceptout)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	ofstream out;
	out.open("Rnuuu.txt");

	//for( int row = 0; row< 50 ; row++)
	//	    testing_result.push_back(0);//initialize with 0s

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	BayesianCoevolution.TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	int samples = 300;

	TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer


	Sucess = false;
	Cycles = 0;


	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, Samples);//setup neural network

	// >>>>>>>>>>>>>>>>> Initialize Neuron Level Island <<<<<<<<<<<< //

	for (int n = 0; n < hidden; n++)
		BayesianCoevolution.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

	for (int n = 0; n < output; n++)
		BayesianCoevolution.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

	BayesianCoevolution.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).

	BayesianCoevolution.InitializeSpecies(samples); //Generate random populations for each species.
	BayesianCoevolution.EvaluateSpecies(network, Samples, 1); //Get fitness of populations in species


	BayesianCoevolution.EvolveBCNE(file, trainout, testout, weightsout, rmsetrainout, rmsetestout, acceptout, 1, hidden, network, Samples, mutation, 0, out1, out2, 1);//evolve sub-populations in round-robin fashion

	BayesianCoevolution.EvaluateSpecies(network, Samples, 1); //Get fitness of populations in species

	BayesianCoevolution.GetBestTable(BayesianCoevolution.NoSpecies - 1);
	BayesianCoevolution.Join();

	TotalEval += BayesianCoevolution.TotalEval;

	//Test train data
	network.ChoromesToNeurons(BayesianCoevolution.Individual, 1);//encode into neural network the best cc individual
	network.SaveLearnedData(layersize, file);//save the neural network weights

											 //test the neural network with training data
	Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);

	cout << "\nPop Best RMSE: " << Train;



	out2 << "Train" << endl;
	//test the neural network with training data
	Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
	setRMSETrain(Train);
	TrainNMSE = network.NMSError();

	out2 << "Test" << endl;
	//test the neural network and the learnt data with the testing data set
	Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
	setRMSETest(Test);
	TestNMSE = network.NMSError();

	cout << "\nTest RMSE: " << Test;


	//clock stops here - execution ends here
	clock_t stop = clock();
	//cout<<"clock stop  "<<stop<<endl;

	//gives time in seconds
	double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
	setTIME(elapsed);

	out2 << endl;
	out1 << endl;
	vcout << endl;

	//ouput training accuracy for each evaluation
	out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
	//output training and test accuracy for file in which we have the actual vs predicted comparison
	out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
	//summary of one experimental run...RMSE and NMSE
	out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << endl;

}

/*
This implementation sits at G3PCX. Whenever a child is generated it is vetted using random walk. The fitter parent
is used as the prior solution and the child the proposed solution.
*/

void    CombinedEvolution::ProcedureBayesianCNE(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	ofstream out;
	out.open("Rnuuu.txt");

	//for( int row = 0; row< 50 ; row++)
	//	    testing_result.push_back(0);//initialize with 0s

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, Samples);//setup neural network


	if (bp) {
		gene = network.BackPropogation(Samples, 0.1, layersize, file, true); //use backpropogation
	}
	else {

		Sucess = false;
		Cycles = 0;

		// >>>>>>>>>>>>>>>>> Initialize Neuron Level Island <<<<<<<<<<<< //

		for (int n = 0; n < hidden; n++)
			NeuronLevel.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

		for (int n = 0; n < output; n++)
			NeuronLevel.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

		NeuronLevel.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).
		NeuronLevel.InitializeSpecies(CCPOPSIZE); //Generate random populations for each species.
		NeuronLevel.EvaluateSpecies(network, Samples, 1); //Get fitness of populations in species

		cout << "Hidden: " << h << " Total Sub-pops: " << NeuronLevel.NoSpecies << endl;

		for (int s = 0; s < NeuronLevel.NoSpecies; s++)
			NeuronLevel.NotConverged[s] = true;

		cout << " Evaluated Neuronlevel ----------->" << endl;


		TotalEval = 0;
		NeuronLevel.TotalEval = 0;


		int NeuronTempEval = 0;

		double BestWL, BestLL, BestNetL;

		int total_epoch = 0;

		int count = 0;
		Layer ErrorArray;
		Layer ValidRMSEArray;

		Error = 99;

		//while the maximum number of function evaluations is not reached
		while (TotalEval <= maxgen)
			/*while (Error >= 0.01)*/
		{
			for (int i = 0; i < 10; i++)
			{
				NeuronLevel.EvolveSubPopulationsBCNE(1, 1, network, Samples, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
				NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
				NeuronLevel.Join();//join the species together
				network.ChoromesToNeurons(NeuronLevel.Individual, 1);//encode into neural network the best cc individual
				network.SaveLearnedData(layersize, file);//save the neural network weights
				Error = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].Population[NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex].Fitness;

				ErrorArray.push_back(Error);
				TotalEval += NeuronLevel.TotalEval;

				out1 << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with
				cout << "\n" << TotalEval << "/" << maxgen << " H: " << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with

				count++;
			}


			/*if (TotalEval > 1000)
			{

			Valid = network.TestTrainingData(layersize, file, validsize, validfile, input, output, out2);
			ValidRMSEArray.push_back(Valid);
			vcout << hidden << " nl " << "Valid: " << Valid << "     Error: " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;
			}*/


		}



		out2 << "Train" << endl;
		//test the neural network with training data
		Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
		setRMSETrain(Train);
		TrainNMSE = network.NMSError();

		out2 << "Test" << endl;
		//test the neural network and the learnt data with the testing data set
		Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
		setRMSETest(Test);
		TestNMSE = network.NMSError();

		cout << "\nTest RMSE: " << Test;


		//clock stops here - execution ends here
		clock_t stop = clock();
		//cout<<"clock stop  "<<stop<<endl;

		//gives time in seconds
		double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
		setTIME(elapsed);

		out2 << endl;
		out1 << endl;
		vcout << endl;

		//ouput training accuracy for each evaluation
		out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
		//output training and test accuracy for file in which we have the actual vs predicted comparison
		out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
		//summary of one experimental run...RMSE and NMSE
		out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << endl;
	}
}
void    CombinedEvolution::ProcedureMCNE(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout, ofstream &trainout, ofstream &testout, ofstream &weightsout)
{

	//############## CC Initialization ##############//

	clock_t start = clock();

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	//TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	//Samples.printData();//print the data
	//double error;
	TrainingExamples SamplesTrain(trainfile, trainsize, input + output, input, output); //get training data
	TrainingExamples SamplesTest(testfile, testsize, input + output, input, output); //get test data

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, SamplesTrain);//setup neural network
	Layer in = network.Neurons_to_chromes(1);
	Sucess = false;
	Cycles = 0;

	for (int n = 0; n < hidden; n++)
		NeuronLevel.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

	for (int n = 0; n < output; n++)
		NeuronLevel.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

	NeuronLevel.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).
	NeuronLevel.InitializeSpecies(CCPOPSIZE); //Generate random populations for each species.
	NeuronLevel.EvaluateSpecies(network, SamplesTrain, 1); //Get fitness of populations in species

	for (int s = 0; s < NeuronLevel.NoSpecies; s++)
		NeuronLevel.NotConverged[s] = true;

	cout << " Evaluated GS Populations" << endl;

	TotalEval = 0;
	NeuronLevel.TotalEval = 0;

	Layer ErrorArray;
	Layer ValidRMSEArray;

	Error = 99;

	MCMC bayesian;

	NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
	NeuronLevel.Join();//join the species together

					   //############## MCMC Initialization ##############//
	int samples = 100;
	VectorManipulation vectormanipulate;


	/*OneDVector_Double y_train;
	OneDVector_Double y_test;*/

	//NeuralNetwork network(layersize); //initialize network
	//network.CreateNetwork(layersize, SamplesTrain);//setup neural network

	int w_size = weightsize1 + weightsize2 + biasize;
	//TwoDVector_Double pos_w = vectormanipulate.Generate2DVector_Double(samples, w_size, 1);
	//OneDVector_Double pos_tau = vectormanipulate.Generate1DVector_Double(samples, 1);

	//TwoDVector_Double fxtrain_samples = vectormanipulate.Generate2DVector_Double(samples, trainsize, 1);
	//TwoDVector_Double fxtest_samples = vectormanipulate.Generate2DVector_Double(samples, testsize, 1);

	//OneDVector_Double rmse_train = vectormanipulate.Generate1DVector_Double(samples, 0);
	//OneDVector_Double rmse_test = vectormanipulate.Generate1DVector_Double(samples, 0);

	//OneDVector_Double w = NeuronLevel.Individual; //Set initial to CC best
	//OneDVector_Double w_proposal = bayesian.GenerateRandomWeights(w_size);

	/*BayesianCoevolution.Population = NeuronLevel.Population;*/

	/*vectormanipulate.Print1DVector(w);*/

	/*double step_w = 0.02;
	double step_eta = 0.01;

	OneDVector_Double pred_train = network.Evaluate_Proposal(SamplesTrain, w, layersize);
	OneDVector_Double pred_test = network.Evaluate_Proposal(SamplesTest, w, layersize);

	double eta = bayesian.GetEta(pred_train, SamplesTrain.GetOutputValues());
	double tau_pro = exp(eta);

	int sigma_squared = 25;
	int nu_1 = 0;
	int nu_2 = 0;
*/

//double prior_likelihood = bayesian.Prior_Likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, h);

//LikelihoodReturnObject likelihood = bayesian.Likelihood_Func(network, SamplesTrain, w, tau_pro, layersize);


	double currentbest = 99;
	double repeatcount = 0;

	int naccept = 0;
	int totalrounds = 0;

	while (totalrounds < 100) //if same rmse hasnt improved in 20 cycles

	{
		totalrounds++;
		//get proposed solution
		cout << "\n------------------------------------" << endl;
		cout << "\nProposing solution" << endl;

		//GS
		cout << "\nGS";
		for (int i = 0; i < 5; i++)
		{
			NeuronLevel.EvolveSubPopulations(1, 1, network, SamplesTrain, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
			NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
			NeuronLevel.Join();//join the species together
			network.ChoromesToNeurons(NeuronLevel.Individual, 1);//encode into neural network the best cc individual

			network.SaveLearnedData(layersize, file);//save the neural network weights
			Error = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].Population[NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex].Fitness;

			TotalEval += NeuronLevel.TotalEval;

		}

		//LS
		//NeuronLevel.EvaluateSpecies(network, SamplesTrain, 1);
		cout << "\nGS Error: " << network.ForwardFitnessPass(NeuronLevel.Individual, SamplesTrain, 1);

		//Set initial to current cc best
		//OneDVector_Double initial_individual = NeuronLevel.Individual;
		//cout << "\nInitial LS Error: " << network.ForwardFitnessPass(initial_individual, Samples, 1);

		//cc best already in network, apply bp

		//apply ls
		cout << "\nLS" << endl;
		gene = network.BackPropogation(SamplesTrain, 0.1, layersize, file, true); //use backpropogation

		//w_proposal = network.Neurons_to_chromes(1);

		/*vectormanipulate.Print1DVector(w_proposal);*/

		//double eta_pro = eta + rand_normal(0, step_eta);
		//double tau_pro = exp(eta_pro);

		//LikelihoodReturnObject likelihood_proposal = bayesian.Likelihood_Func(network, SamplesTrain, w_proposal, tau_pro, layersize);

		//double prior_prop = bayesian.Prior_Likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, h);

		//double diff_likelihood = likelihood_proposal.likelihood - likelihood.likelihood;
		//double diff_priorliklihood = prior_prop - prior_likelihood;

		//double  mh_prob = 1;

		//if (exp(diff_likelihood + diff_priorliklihood) < 1) {
		//	mh_prob = exp(diff_likelihood + diff_priorliklihood);
		//}

		///*	RandomNumber rn;*/
		//double u = bayesian.randfrom(0, 1);

		//if (u < mh_prob)
		//{
			//cout << "\nAccepted Solution";
			//naccept += 1;
			//likelihood = likelihood_proposal;

			//cout << "\nRMSE: " << likelihood_proposal.rmse;

			//prior_likelihood = prior_prop;
			////w = w_proposal;
			//eta = eta_pro;

			//########## Replace worst in CC pop ##########//
		NeuronLevel.Individual = network.Neurons_to_chromes(1); //Set neuron level individual from current network individual
		int Worst = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].WorstIndex;
		for (int sN = 0; sN < NeuronLevel.NoSpecies; sN++)
		{
			OneDVector_Double sampledchild = NeuronLevel.SplitIndividualGetBackOffspring(NeuronLevel.Individual, sN);
			for (int o = 0; o < NeuronLevel.Species[sN].NumVariable; o++)
				NeuronLevel.Species[sN].Population[Worst].Chrome[o] = sampledchild[o];
		}

		/*BayesianCoevolution.Population = NeuronLevel.Population;*/

		//if (likelihood_proposal.rmse > currentbest)
		//{
		//	repeatcount++;
		//	cout << "\nRepeat Count: " << repeatcount;
		//}
		//else
		//{
		//	currentbest = likelihood_proposal.rmse;
		//	repeatcount = 0;

		//}

	//}
	//else
	//{
	//	cout << "\nRejected Solution";
	//	//Proposal was rejected
	//	//Reset GS solution

	//	NeuronLevel.Population = BayesianCoevolution.Population;
	//}

		cout << "\nAccepted: " << naccept << " Total: " << totalrounds;
	}

	ChartData writedata;


	out2 << "Train" << endl;
	//test the neural network with training data
	Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
	setRMSETrain(Train);
	TrainNMSE = network.NMSError();

	writedata.WriteAccuracyData(network.GetOutput(), trainout, trainsize);


	out2 << "Test" << endl;
	//test the neural network and the learnt data with the testing data set
	Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
	setRMSETest(Test);
	TestNMSE = network.NMSError();

	cout << "\nTest RMSE: " << Test;

	writedata.WriteAccuracyData(network.GetOutput(), testout, testsize);


	//clock stops here - execution ends here
	clock_t stop = clock();
	//cout<<"clock stop  "<<stop<<endl;

	//gives time in seconds
	double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
	setTIME(elapsed);

	out2 << endl;
	out1 << endl;
	vcout << endl;

	//ouput training accuracy for each evaluation
	out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
	//output training and test accuracy for file in which we have the actual vs predicted comparison
	out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
	//summary of one experimental run...RMSE and NMSE
	out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << " Accepted: " << naccept << " Total: " << totalrounds << endl;
	writedata.WriteAccuracyData(in, weightsout, in.size());

}

void    CombinedEvolution::ProcedureBayesianMemetic_MCNEINMCMC(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout, ofstream &trainout, ofstream &testout, ofstream &weightsout)
{

	//############## CC Initialization ##############//

	clock_t start = clock();

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	//TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	//Samples.printData();//print the data
	//double error;
	TrainingExamples SamplesTrain(trainfile, trainsize, input + output, input, output); //get training data
	TrainingExamples SamplesTest(testfile, testsize, input + output, input, output); //get test data

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, SamplesTrain);//setup neural network
	Layer in = network.Neurons_to_chromes(1);
	Sucess = false;
	Cycles = 0;

	for (int n = 0; n < hidden; n++)
		NeuronLevel.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

	for (int n = 0; n < output; n++)
		NeuronLevel.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

	NeuronLevel.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).
	NeuronLevel.InitializeSpecies(CCPOPSIZE); //Generate random populations for each species.
	NeuronLevel.EvaluateSpecies(network, SamplesTrain, 1); //Get fitness of populations in species

	for (int s = 0; s < NeuronLevel.NoSpecies; s++)
		NeuronLevel.NotConverged[s] = true;

	cout << " Evaluated GS Populations" << endl;

	TotalEval = 0;
	NeuronLevel.TotalEval = 0;

	Layer ErrorArray;
	Layer ValidRMSEArray;

	Error = 99;

	MCMC bayesian;

	NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
	NeuronLevel.Join();//join the species together

	//############## MCMC Initialization ##############//
	int samples = 100;
	VectorManipulation vectormanipulate;


	OneDVector_Double y_train;
	OneDVector_Double y_test;

	//NeuralNetwork network(layersize); //initialize network
	//network.CreateNetwork(layersize, SamplesTrain);//setup neural network

	int w_size = weightsize1 + weightsize2 + biasize;
	TwoDVector_Double pos_w = vectormanipulate.Generate2DVector_Double(samples, w_size, 1);
	OneDVector_Double pos_tau = vectormanipulate.Generate1DVector_Double(samples, 1);

	TwoDVector_Double fxtrain_samples = vectormanipulate.Generate2DVector_Double(samples, trainsize, 1);
	TwoDVector_Double fxtest_samples = vectormanipulate.Generate2DVector_Double(samples, testsize, 1);

	OneDVector_Double rmse_train = vectormanipulate.Generate1DVector_Double(samples, 0);
	OneDVector_Double rmse_test = vectormanipulate.Generate1DVector_Double(samples, 0);

	OneDVector_Double w = NeuronLevel.Individual; //Set initial to CC best
	OneDVector_Double w_proposal = bayesian.GenerateRandomWeights(w_size);

	BayesianCoevolution.Population = NeuronLevel.Population;

	vectormanipulate.Print1DVector(w);

	double step_w = 0.02;
	double step_eta = 0.01;

	OneDVector_Double pred_train = network.Evaluate_Proposal(SamplesTrain, w, layersize);
	OneDVector_Double pred_test = network.Evaluate_Proposal(SamplesTest, w, layersize);

	double eta = bayesian.GetEta(pred_train, SamplesTrain.GetOutputValues());
	double tau_pro = exp(eta);

	int sigma_squared = 25;
	int nu_1 = 0;
	int nu_2 = 0;


	double prior_likelihood = bayesian.Prior_Likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, h);

	LikelihoodReturnObject likelihood = bayesian.Likelihood_Func(network, SamplesTrain, w, tau_pro, layersize);


	double currentbest = 99;
	double repeatcount = 0;

	int naccept = 0;
	int totalrounds = 0;

	while (repeatcount < 2) //if same rmse hasnt improved in 20 cycles

	{
		totalrounds++;
		//get proposed solution
		cout << "\n------------------------------------" << endl;
		cout << "\nProposing solution" << endl;

		//GS
		cout << "\nGS";
		for (int i = 0; i < 5; i++)
		{
			NeuronLevel.EvolveSubPopulations(1, 1, network, SamplesTrain, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
			NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
			NeuronLevel.Join();//join the species together
			network.ChoromesToNeurons(NeuronLevel.Individual, 1);//encode into neural network the best cc individual

			network.SaveLearnedData(layersize, file);//save the neural network weights
			Error = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].Population[NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex].Fitness;

			TotalEval += NeuronLevel.TotalEval;

		}

		//LS
		//NeuronLevel.EvaluateSpecies(network, SamplesTrain, 1);
		cout << "\nGS Error: " << network.ForwardFitnessPass(NeuronLevel.Individual, SamplesTrain, 1);

		//Set initial to current cc best
		//OneDVector_Double initial_individual = NeuronLevel.Individual;
		//cout << "\nInitial LS Error: " << network.ForwardFitnessPass(initial_individual, Samples, 1);

		//cc best already in network, apply bp

		//apply ls
		cout << "\nLS" << endl;
		gene = network.BackPropogation(SamplesTrain, 0.1, layersize, file, true); //use backpropogation

		w_proposal = network.Neurons_to_chromes(1);

		/*vectormanipulate.Print1DVector(w_proposal);*/

		double eta_pro = eta + rand_normal(0, step_eta);
		double tau_pro = exp(eta_pro);

		LikelihoodReturnObject likelihood_proposal = bayesian.Likelihood_Func(network, SamplesTrain, w_proposal, tau_pro, layersize);

		double prior_prop = bayesian.Prior_Likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, h);

		double diff_likelihood = likelihood_proposal.likelihood - likelihood.likelihood;
		double diff_priorliklihood = prior_prop - prior_likelihood;

		double  mh_prob = 1;

		if (exp(diff_likelihood + diff_priorliklihood) < 1) {
			mh_prob = exp(diff_likelihood + diff_priorliklihood);
		}

		/*	RandomNumber rn;*/
		double u = bayesian.randfrom(0, 1);

		if (u < mh_prob)
		{
			cout << "\nAccepted Solution";
			naccept += 1;
			likelihood = likelihood_proposal;

			cout << "\nRMSE: " << likelihood_proposal.rmse;

			prior_likelihood = prior_prop;
			//w = w_proposal;
			eta = eta_pro;

			//########## Replace worst in CC pop ##########//
			NeuronLevel.Individual = network.Neurons_to_chromes(1); //Set neuron level individual from current network individual
			int Worst = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].WorstIndex;
			for (int sN = 0; sN < NeuronLevel.NoSpecies; sN++)
			{
				OneDVector_Double sampledchild = NeuronLevel.SplitIndividualGetBackOffspring(NeuronLevel.Individual, sN);
				for (int o = 0; o < NeuronLevel.Species[sN].NumVariable; o++)
					NeuronLevel.Species[sN].Population[Worst].Chrome[o] = sampledchild[o];
			}

			BayesianCoevolution.Population = NeuronLevel.Population;
		}
		else
		{
			cout << "\nRejected Solution";
			//Proposal was rejected
			//Reset GS solution

			NeuronLevel.Population = BayesianCoevolution.Population;
		}


		if (likelihood_proposal.rmse > currentbest)
		{
			repeatcount++;
			cout << "\nRepeat Count: " << repeatcount;
		}
		else
		{
			currentbest = likelihood_proposal.rmse;
			repeatcount = 0;

		}

		cout << "\nAccepted: " << naccept << " Total: " << totalrounds;
	}

	ChartData writedata;


	out2 << "Train" << endl;
	//test the neural network with training data
	Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
	setRMSETrain(Train);
	TrainNMSE = network.NMSError();

	writedata.WriteAccuracyData(network.GetOutput(), trainout, trainsize);


	out2 << "Test" << endl;
	//test the neural network and the learnt data with the testing data set
	Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
	setRMSETest(Test);
	TestNMSE = network.NMSError();

	cout << "\nTest RMSE: " << Test;

	writedata.WriteAccuracyData(network.GetOutput(), testout, testsize);


	//clock stops here - execution ends here
	clock_t stop = clock();
	//cout<<"clock stop  "<<stop<<endl;

	//gives time in seconds
	double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
	setTIME(elapsed);

	out2 << endl;
	out1 << endl;
	vcout << endl;

	//ouput training accuracy for each evaluation
	out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
	//output training and test accuracy for file in which we have the actual vs predicted comparison
	out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
	//summary of one experimental run...RMSE and NMSE
	out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << " Accepted: " << naccept << " Total: " << totalrounds << endl;
	writedata.WriteAccuracyData(in, weightsout, in.size());

}

void    CombinedEvolution::ProcedureBayesianMemetic(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout, ofstream &trainout, ofstream &testout, ofstream &weightsout)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	ofstream out;
	out.open("Rnuuu.txt");

	//for( int row = 0; row< 50 ; row++)
	//	    testing_result.push_back(0);//initialize with 0s

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, Samples);//setup neural network

	Layer in = network.Neurons_to_chromes(1);


	if (bp) {
		gene = network.BackPropogation(Samples, 0.1, layersize, file, true); //use backpropogation
	}
	else {

		Sucess = false;
		Cycles = 0;

		// >>>>>>>>>>>>>>>>> Initialize Neuron Level Island <<<<<<<<<<<< //

		for (int n = 0; n < hidden; n++)
			NeuronLevel.SpeciesSize.push_back(input + 1); //Set size of species between input and hidden layer (total input neurons + 1 bias)

		for (int n = 0; n < output; n++)
			NeuronLevel.SpeciesSize.push_back(hidden + 1); //Set size of species between hidden and output layer (total hidden neurons + 1 bias)

		NeuronLevel.NoSpecies = hidden + output; //Set size of all species in neuronlevel(total hidden neurons + total output neurons).
		NeuronLevel.InitializeSpecies(CCPOPSIZE); //Generate random populations for each species.
		NeuronLevel.EvaluateSpecies(network, Samples, 1); //Get fitness of populations in species

		cout << "Hidden: " << h << " Total Sub-pops: " << NeuronLevel.NoSpecies << endl;

		for (int s = 0; s < NeuronLevel.NoSpecies; s++)
			NeuronLevel.NotConverged[s] = true;

		cout << " Evaluated Neuronlevel ----------->" << endl;


		TotalEval = 0;
		NeuronLevel.TotalEval = 0;


		int NeuronTempEval = 0;

		double BestWL, BestLL, BestNetL;

		int total_epoch = 0;

		int count = 0;
		Layer ErrorArray;
		Layer ValidRMSEArray;

		Error = 99;

		MCMC bayesian;


		NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
		NeuronLevel.Join();//join the species together


		double currentbest = 99;
		double repeatcount = 0;

		int totalrounds = 0;
		int acceptedrounds = 0;

		while (acceptedrounds < 150) //if same rmse hasnt improved in 20 cycles
		{
			totalrounds++;

			bool applyls = false;

			//Global Search Time
			cout << "\nGS Time" << endl;
			for (int i = 0; i < 10; i++)
			{
				NeuronLevel.EvolveSubPopulations(1, 1, network, Samples, mutation, 0, out2, 1);//evolve sub-populations in round-robin fashion
				NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
				NeuronLevel.Join();//join the species together
				network.ChoromesToNeurons(NeuronLevel.Individual, 1);//encode into neural network the best cc individual
				//need to do one forward pass to get the correct sumquaredd error


				network.SaveLearnedData(layersize, file);//save the neural network weights
				Error = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].Population[NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex].Fitness;


				ErrorArray.push_back(Error);
				TotalEval += NeuronLevel.TotalEval;

				out1 << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with
				cout << "\n" << TotalEval << "/" << maxgen << " H: " << hidden << " nl " << Train << "    " << Error << "    " << NeuronLevel.TotalEval << "    " << count << endl;//shows how the RMSE is going down with

				count++;
			}

			NeuronLevel.EvaluateSpecies(network, Samples, 1);

			cout << "\nGS Error: " << network.ForwardFitnessPass(NeuronLevel.Individual, Samples, 1);


			//Set initial to current cc best
			OneDVector_Double initial_individual = NeuronLevel.Individual;
			cout << "\nInitial LS Error: " << network.ForwardFitnessPass(initial_individual, Samples, 1);


			//cc best already in network, apply bp

			//apply ls
			gene = network.BackPropogation(Samples, 0.1, layersize, file, true); //use backpropogation

			bool guided = false;
			cout << "\nProposed LS Error: " << network.ForwardFitnessPass(network.Neurons_to_chromes(1), Samples, 1);


			applyls = bayesian.ExecuteVetting(initial_individual, network.Neurons_to_chromes(1), h);

			if (applyls)
			{
				acceptedrounds++;
				cout << "\nAccepted for GS" << endl;
				cout << "\nTransferring";

				NeuronLevel.Individual = network.Neurons_to_chromes(1); //Set neuron level individual from current network individual

			/*	int index = 0;

				for (int row = 0; row < NeuronLevel.NoSpecies; row++)
				{
					for (int col = 0; col < NeuronLevel.SpeciesSize[row]; col++)
					{
						NeuronLevel.TableSp[row].SingleSp[col] = NeuronLevel.Individual[index];
						index++;
					}
				}
*/
//---------------------------------------------------------------------------------------------------
//transfer to best only
/*int Best = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].BestIndex;


for (int sN = 0; sN < NeuronLevel.NoSpecies - 1; sN++)
{
for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
NeuronLevel.Species[sN].Population[Best].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
NeuronLevel.Species[sN].BestIndex = Best;
}

for (int sN = NeuronLevel.NoSpecies - 1; sN < NeuronLevel.NoSpecies; sN++)
{
for (int s = 0; s < NeuronLevel.SpeciesSize[sN]; s++)
NeuronLevel.Species[sN].Population[Best].Chrome[s] = NeuronLevel.TableSp[sN].SingleSp[s];
NeuronLevel.Species[sN].BestIndex = Best;
}*/

//---------------------------------------------------------------------------------------------------
//transfer to worst only
				int Worst = NeuronLevel.Species[NeuronLevel.NoSpecies - 1].WorstIndex;


				for (int sN = 0; sN < NeuronLevel.NoSpecies; sN++)
				{
					OneDVector_Double sampledchild = NeuronLevel.SplitIndividualGetBackOffspring(NeuronLevel.Individual, sN);

					for (int o = 0; o < NeuronLevel.Species[sN].NumVariable; o++)
						NeuronLevel.Species[sN].Population[Worst].Chrome[o] = sampledchild[o];
				}



				cout << "\nDone Transferring";

			}
			else {

				cout << "\nRejected for GS" << endl;
			}



			NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
			NeuronLevel.Join();

			Train = network.ForwardFitnessPass(NeuronLevel.Individual, Samples, 1);

			/*	if (Train > currentbest || Train == currentbest)
				{
					repeatcount++;
					cout << "Repeat Count: " << repeatcount;
				}
				else
				{
					currentbest = Train;
					repeatcount = 0;

				}*/

			cout << "\n Accepted: " << acceptedrounds << "/" << totalrounds;

		}

		ChartData writedata;


		out2 << "Train" << endl;
		//test the neural network with training data
		Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
		setRMSETrain(Train);
		TrainNMSE = network.NMSError();

		writedata.WriteAccuracyData(network.GetOutput(), trainout, trainsize);


		out2 << "Test" << endl;
		//test the neural network and the learnt data with the testing data set
		Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
		setRMSETest(Test);
		TestNMSE = network.NMSError();

		cout << "\nTest RMSE: " << Test;

		writedata.WriteAccuracyData(network.GetOutput(), testout, testsize);


		//clock stops here - execution ends here
		clock_t stop = clock();
		//cout<<"clock stop  "<<stop<<endl;

		//gives time in seconds
		double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
		setTIME(elapsed);

		out2 << endl;
		out1 << endl;
		vcout << endl;

		//ouput training accuracy for each evaluation
		out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
		//output training and test accuracy for file in which we have the actual vs predicted comparison
		out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
		//summary of one experimental run...RMSE and NMSE
		out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << " Accepted: " << acceptedrounds << " Total: " << totalrounds << endl;
		writedata.WriteAccuracyData(in, weightsout, in.size());
	}
}

/*
This implementation sits at G3PCX. Whenever a child is generated it is vetted using random walk. The fitter parent
is used as the prior solution and the child the proposed solution.
*/

void    CombinedEvolution::ProcedureSGD(string file, bool bp, double h, ofstream &out1, ofstream &out2, ofstream &out3, double mutation, double depth, ofstream &vcout, ofstream &ecout, ofstream &bcout, ofstream &trainout, ofstream &testout, ofstream &weightsout)
{

	//measures the execution time - execution begins here
	clock_t start = clock();
	//cout<<"clock start  "<<start<<endl;

	int hidden = static_cast<int>(h);
	int weightsize1 = (input*hidden);//number of weights between hidden and input layer
	int weightsize2 = (hidden*output);//number of weights between hidden and output layer
	int biasize = hidden + output; //bias for hidden and output layer

	ofstream out;
	out.open("Rnuuu.txt");

	//for( int row = 0; row< 50 ; row++)
	//	    testing_result.push_back(0);//initialize with 0s

	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;
	TotalEval = 0;

	double H = 0;
	int gene = 1;
	int item = 2;

	TrainingExamples Samples(trainfile, trainsize, input + output, input, output); //get training data
	Samples.printData();//print the data
	double error;

	Sizes layersize;
	layersize.push_back(input);//set size(neurons) of input layer
	layersize.push_back(hidden);//set size(neurons) of hidden layer--only one hidden layer
	layersize.push_back(output); //set size(neurons) of output layer

	NeuralNetwork network(layersize); //initialize network
	network.CreateNetwork(layersize, Samples);//setup neural network
	Layer individual = network.Neurons_to_chromes(1);

	//gene = network.BackPropogation(Samples, 0.1, layersize, file, true); //use backpropogation
	gene = network.BackPropogationExhausted(Samples, 0.1, layersize, file, true); //use backpropogation
	ChartData writedata;

	out2 << "Train" << endl;
	//test the neural network with training data
	Train = network.TestTrainingData(layersize, file, trainsize, trainfile, input, output, out2);
	setRMSETrain(Train);
	TrainNMSE = network.NMSError();

	writedata.WriteAccuracyData(network.GetOutput(), trainout, trainsize);


	out2 << "Test" << endl;
	//test the neural network and the learnt data with the testing data set
	Test = network.TestTrainingData(layersize, file, testsize, testfile, input, output, out2);
	setRMSETest(Test);
	TestNMSE = network.NMSError();

	cout << "\nTest RMSE: " << Test;
	writedata.WriteAccuracyData(network.GetOutput(), testout, testsize);

	//clock stops here - execution ends here
	clock_t stop = clock();
	//cout<<"clock stop  "<<stop<<endl;

	//gives time in seconds
	double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
	setTIME(elapsed);

	out2 << endl;
	out1 << endl;
	vcout << endl;

	//ouput training accuracy for each evaluation
	out1 << " ------------------------------ " << h << "  " << TotalEval << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;
	//output training and test accuracy for file in which we have the actual vs predicted comparison
	out2 << " ------------------------------ " << h << "  " << TotalEval << "  " << Train << "  " << Test << endl;
	//summary of one experimental run...RMSE and NMSE
	out3 << h << "\t" << TotalEval << "\tRMSE:\t" << Train << "\t" << Test << "\tNMSE:\t" << TrainNMSE << "\t" << TestNMSE << "\t" << elapsed << endl;

	writedata.WriteAccuracyData(individual, weightsout, individual.size());

}






int main(void)
{
	ofstream lcout("Log\\ExecutionLog.txt");
	ofstream rcout("Log\\ResultSummary.txt");

	vector<string> datasetNames;
	string sourcefolder = "PythonDatasets\\";
	//datasetNames.push_back(sourcefolder + "Sunspot");
	datasetNames.push_back(sourcefolder + "Mackey");
	datasetNames.push_back(sourcefolder + "ACFinance");
	datasetNames.push_back(sourcefolder + "Lazer");
	//datasetNames.push_back(sourcefolder + "Lorenz");
	/*datasetNames.push_back(sourcefolder + "Sunspot");*/
	/*datasetNames.push_back(sourcefolder + "Henon");*/
	/*datasetNames.push_back(sourcefolder + "Rossler");*/



	/*datasetNames.push_back("HenonPython");*/
	/*datasetNames.push_back("ACFinancePython");*/
	/*datasetNames.push_back("SunspotPython");*/
	/*datasetNames.push_back("TWIExchange");
	datasetNames.push_back("SPSEVolTrades");
	datasetNames.push_back("MelbourneMaxTemp");
	datasetNames.push_back("GlobalTempChange");
	datasetNames.push_back("HeartRate");
	datasetNames.push_back("NadiWeather2016");*/
	/*datasetNames.push_back("Laser");
	datasetNames.push_back("Sunspot");
	datasetNames.push_back("Mackey");
	datasetNames.push_back("Lorenz");*/

	string datasetSourceFolder = "Datasets";
	string datasetSourceResultFolder = "Results";

	lcout << "No. of Datasets: " << datasetNames.size() << endl;
	lcout << "Datasets: ";
	for (int u = 0; u < datasetNames.size(); u++)
	{
		lcout << datasetNames[u] << " ";
	}

	lcout << "\n\n";

	ostringstream oss;


	for (int datasetEnumerator = 0; datasetEnumerator < datasetNames.size(); datasetEnumerator++)
	{
		lcout << "Running " << datasetNames[datasetEnumerator] << " Experiment" << endl;

		try
		{


			string temp = datasetSourceFolder + "\\" + datasetNames[datasetEnumerator] + "\\" + "\\SETTINGS.txt";
			ifstream in(temp.c_str());
			if (!in) {
				cout << endl << "Failed to open settings" << endl;//error reading from file
				return 0;
			}
			string str;
			while (in >> str) {

				if (str == "Dataset:") {
					in >> str;
					cout << "Dataset: " << str << endl;
				}

				if (str == "TrainSize:") {
					in >> str;
					trainsize = atoi(str.c_str());
					cout << "TrainSize: " << trainsize << endl;
				}
				if (str == "TestSize:") {
					in >> str;
					testsize = atoi(str.c_str());
					cout << "TestSize: " << testsize << endl;
				}
				if (str == "ValidationSize:") {
					in >> str;
					validsize = atoi(str.c_str());
					cout << "ValidationSize: " << validsize << endl;
				}
				if (str == "Input:") {
					in >> str;
					input = atoi(str.c_str());
					cout << "Input: " << input << endl;
				}
				if (str == "MinimumError:") {
					in >> str;
					MinimumError = atof(str.c_str());
					cout << "MinimumError: " << MinimumError << endl;
				}
				if (str == "HiddenNeuronsMax:") {
					in >> str;
					MaxNumNeurons = atoi(str.c_str());
					cout << "MaxNumNeurons: " << MaxNumNeurons << endl;
				}
				if (str == "HiddenIncrementBy:") {
					in >> str;
					HiddenIncrementBy = atoi(str.c_str());
					cout << "HiddenIncrementBy: " << HiddenIncrementBy << endl;
				}
				if (str == "HiddenNeuronsStart:") {
					in >> str;
					HiddenNeuronsStart = atoi(str.c_str());
					cout << "HiddenNeuronsStart: " << HiddenNeuronsStart << endl;
				}







			}


			in.close();

			/*system("pause");*/


			int VSize = 90;
			string file = "";

			string thisDatasetResultsLocation = datasetSourceFolder + "\\" + datasetNames[datasetEnumerator] + "\\" + datasetSourceResultFolder + "\\";
			string thisDatasetResultsLocationLearnt = thisDatasetResultsLocation + "Learnt.txt";

			//string thisDatasetLocation = datasetSourceFolder + "\\" + datasetNames[datasetEnumerator] + "\\" + "train.txt";
			trainfile = datasetSourceFolder + "\\" + datasetNames[datasetEnumerator] + "\\" + "train.txt";
			testfile = datasetSourceFolder + "\\" + datasetNames[datasetEnumerator] + "\\" + "test.txt";
			validfile = datasetSourceFolder + "\\" + datasetNames[datasetEnumerator] + "\\" + "validation.txt";
			file = thisDatasetResultsLocationLearnt;

			temp = thisDatasetResultsLocation + "ValidationCheck.txt";
			ofstream vcout(temp.c_str());
			temp = thisDatasetResultsLocation + "Error.txt";
			ofstream ecout(temp.c_str());
			temp = thisDatasetResultsLocation + "BestSnapshot.txt";
			ofstream bcout(temp.c_str());

			//create output files
			ofstream out1;
			temp = thisDatasetResultsLocation + "Oneout1.txt";
			out1.open(temp.c_str());
			ofstream out2;
			temp = thisDatasetResultsLocation + "Oneout2.txt";
			out2.open(temp.c_str());//actual vs predicted values for each run
			ofstream out3;
			temp = thisDatasetResultsLocation + "Oneout3.txt";
			out3.open(temp.c_str());//summary of each experimental run
			ofstream out4;
			temp = thisDatasetResultsLocation + "Oneout4.txt";
			out4.open(temp.c_str());//summary of each hidden neuron x number of experimental runs

			ofstream trainout;
			temp = thisDatasetResultsLocation + "\\ChartData\\trainout.txt";
			trainout.open(temp.c_str());

			ofstream testout;
			temp = thisDatasetResultsLocation + "\\ChartData\\testout.txt";
			testout.open(temp.c_str());

			ofstream weightsout;
			temp = thisDatasetResultsLocation + "\\ChartData\\weightsout.txt";
			weightsout.open(temp.c_str());

			ofstream rmsetrainout;
			temp = thisDatasetResultsLocation + "\\ChartData\\rmsetrainout.txt";
			rmsetrainout.open(temp.c_str());

			ofstream rmsetestout;
			temp = thisDatasetResultsLocation + "\\ChartData\\rmsetestout.txt";
			rmsetestout.open(temp.c_str());

			ofstream acceptout;
			temp = thisDatasetResultsLocation + "\\ChartData\\acceptout.txt";
			acceptout.open(temp.c_str());

			TrainingExamples Train(trainfile, trainsize, input + output, input, output);
			TrainingExamples Test(testfile, testsize, input + output, input, output);

			ChartData data;
			data.WriteAccuracyData(Train.GetOutputValues(), trainout, trainsize);
			data.WriteAccuracyData(Test.GetOutputValues(), testout, testsize);

			for (double hidden = HiddenNeuronsStart; hidden <= MaxNumNeurons; hidden += HiddenIncrementBy) {//13
																											//for(double onelevelstop=0.01;onelevelstop>=0.01;onelevelstop-=0.002){
				double onelevelstop = 0.01;
				//variable declaration
				Sizes EvalAverage;
				Layer ErrorAverage;//error everage
				Layer CycleAverage;
				Layer NMSETrainAve;//store NMSE training average
				Layer NMSETestAve;//store NMSE test average
				Layer RMSETest;
				Layer RMSETrain;
				int MeanEval = 0;
				double MeanError = 0;//initialize all to zeros
				double MeanCycle = 0;
				double NMSETrainMean = 0;
				double NMSETestMean = 0;
				double EvalSum = 0;
				double NMSETrainSum = 0;
				double NMSETestSum = 0;
				double ErrorSum = 0;
				double TrainErrorSum = 0;
				double CycleSum = 0;
				double maxrun = 1;//number of experimental runs
				int success = 0;
				double test_average = 0;
				double test_best_rmse = 0;
				double test_sum = 0;
				double test_best_nmse = 0;
				double train_average = 0;
				double train_best_rmse = 0;
				double train_sum = 0;
				double BestRMSE = 10;
				double BestNMSE = 10;

				for (int run = 1; run <= maxrun; run++) {
					oss.str("");
					oss.clear();
					oss << "Experiment: SGD Dataset " << (datasetEnumerator + 1) << "/" << (datasetNames.size()) << " - " << datasetNames[datasetEnumerator] << " Run " << run << "/" << maxrun;



					SetConsoleTitle(oss.str().c_str()); //Update title

					ecout << "\n\n";

					CombinedEvolution Combined;
#ifdef UseMCNE

					Combined.ProcedureMCNE(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout, trainout, testout, weightsout);//run the main CC algorithm

#endif

#ifdef UseBayesianMemetic_MCNEINMCMC

					Combined.ProcedureBayesianMemetic_MCNEINMCMC(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout, trainout, testout, weightsout);//run the main CC algorithm

#endif

#ifdef UseBayesianMemetic

					Combined.ProcedureBayesianMemetic(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout, trainout, testout, weightsout);//run the main CC algorithm

#endif

#ifdef UseLocalCooperativeNeuroEvolutionPartial

					Combined.LocalCooperativeNeuroEvolution2(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout);//run the main CC algorithm

#endif
#ifdef UseLocalCooperativeNeuroEvolution

					Combined.LocalCooperativeNeuroEvolution(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout);//run the main CC algorithm

#endif

#ifdef UseCCMCMC

					Combined.CCMCMC(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout,trainout, testout, weightsout, rmsetrainout, rmsetestout, acceptout);//run the main CC algorithm

#endif
#ifdef UseBayesianCooperativeNeuroEvolution

					Combined.BayesianCooperativeNeuroEvolution(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout);//run the main CC algorithm

#endif



#ifdef UseSGD

					Combined.ProcedureSGD(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout, trainout, testout, weightsout);//run the main CC algorithm

#endif

#ifdef UseBCNE1Step

					Combined.ProcedureBayesianCNE1Step(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout);//run the main CC algorithm

#endif

#ifdef UseBCNEGenerateOnly

					Combined.ProcedureBayesianCNE1GenerateOnly(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout);//run the main CC algorithm

#endif




#ifdef UseBCNE

					Combined.ProcedureBayesianCNE(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout);//run the main CC algorithm

#endif
#ifdef UseCC

					Combined.ProcedureCC(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout);//run the main CC algorithm

#endif

#ifdef UseMCMC

					MCMC bayesian;
					/*OneDVector_Double w = bayesian.Execute(50, hidden);*/
					OneDVector_Double w = bayesian.ExecuteTillExhausted(50, hidden);

					Combined.SaveBCNEData(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout, w);//run the main CC algorithm

#endif





#ifdef Memory
					Combined.ProcedureMemory(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout);//run the main CC algorithm
#endif

#ifdef LSMaster
					Combined.ProcedureLSMaster(file, false, hidden, out1, out2, out3, 0, onelevelstop, vcout, ecout, bcout);//run the main CC algorithm
#endif

																									   //Combined.ProcedureLSMaster(file, false, hidden, out1, out2, out3, 0, onelevelstop);//run the main CC algorithm

					if (Combined.GetSucess()) {
						success++;
					}

					NMSETrainAve.push_back(Combined.NMSETrain());
					NMSETrainMean += Combined.NMSETrain();

					NMSETestAve.push_back(Combined.NMSETest());
					NMSETestMean += Combined.NMSETest();

					RMSETrain.push_back(Combined.getRMSETrain());
					train_average += Combined.getRMSETrain();

					RMSETest.push_back(Combined.getRMSETest());
					test_average += Combined.getRMSETest();

					EvalAverage.push_back(Combined.GetEval());
					MeanEval += Combined.GetEval();

					//cout<<"hejfs"<<endl;
					if (Combined.GetError() < BestRMSE)
						BestRMSE = Combined.GetError();
				}//run

				NMSETestMean /= NMSETestAve.size();
				NMSETrainMean /= NMSETrainAve.size();
				MeanEval = MeanEval / static_cast<int>(EvalAverage.size());
				//calculating average test rmse
				for (int y = 0; y < RMSETest.size(); y++) {
					test_sum = test_sum + RMSETest[y];
				}
				test_average = test_average / RMSETest.size();////////////

															  //calculating best test rmse
				test_best_rmse = RMSETest[0];
				for (int i = 1; i < RMSETest.size(); i++) {
					if (test_best_rmse > RMSETest[i]) {
						test_best_rmse = RMSETest[i];
					}
				}

				//calculating average train rmse
				for (int n = 0; n < RMSETrain.size(); n++) {
					train_sum = train_sum + RMSETrain[n];
				}
				train_average = train_average / RMSETrain.size();

				//calculating best train rmse
				train_best_rmse = RMSETrain[0];
				for (int x = 1; x < RMSETrain.size(); x++) {
					if (train_best_rmse > RMSETrain[x]) {
						train_best_rmse = RMSETrain[x];
					}
				}

				//calculating best test nmse
				test_best_nmse = NMSETestAve[0];
				for (int j = 1; j < NMSETestAve.size(); j++) {
					if (test_best_nmse > NMSETestAve[j]) {
						test_best_nmse = NMSETestAve[j];
					}
				}

				//CI for training nmse
				for (int a = 0; a < NMSETrainAve.size(); a++)
					NMSETrainSum += (NMSETrainAve[a] - NMSETrainMean)*(NMSETrainAve[a] - NMSETrainMean);

				NMSETrainSum /= NMSETrainAve.size();
				NMSETrainSum = sqrt(NMSETrainSum);
				NMSETrainSum = 1.96 * (NMSETrainSum / sqrt(NMSETrainAve.size()));

				//CI for test nmse
				for (int a = 0; a < NMSETestAve.size(); a++)
					NMSETestSum += (NMSETestAve[a] - NMSETestMean)*(NMSETestAve[a] - NMSETestMean);

				NMSETestSum /= NMSETestAve.size();
				NMSETestSum = sqrt(NMSETestSum);
				NMSETestSum = 1.96 * (NMSETestSum / sqrt(NMSETestAve.size()));

				//-------------------------------------------------

				for (int a = 0; a < EvalAverage.size(); a++)
					EvalSum += (EvalAverage[a] - MeanEval)*(EvalAverage[a] - MeanEval);

				EvalSum = EvalSum / static_cast<int>(EvalAverage.size());
				EvalSum = sqrt(EvalSum);

				EvalSum = 1.96*(EvalSum / sqrt(EvalAverage.size()));
				/////////////////////////////////////////////////////////////////////////
				//CI for test rmse
				for (int a = 0; a < RMSETest.size(); a++)
					ErrorSum += (RMSETest[a] - test_average)*(RMSETest[a] - test_average);

				ErrorSum = ErrorSum / RMSETest.size();
				ErrorSum = sqrt(ErrorSum);
				ErrorSum = 1.96*(ErrorSum / sqrt(RMSETest.size()));
				//////////////////////////////////////////////////////////////////////////
				//CI for train rmse
				for (int b = 0; b < RMSETrain.size(); b++)
					TrainErrorSum += (RMSETrain[b] - train_average)*(RMSETrain[b] - train_average);

				TrainErrorSum = TrainErrorSum / RMSETrain.size();
				TrainErrorSum = sqrt(TrainErrorSum);
				TrainErrorSum = 1.96*(TrainErrorSum / sqrt(RMSETrain.size()));

				////////////////////////////////////////////////////////////////////////////

				rcout << "Dataset: " << datasetNames[datasetEnumerator] << " H: " << hidden << "	Mean Eval: " << MeanEval << " Eval Error: " << EvalSum << "	\nRMSE Train Average: " << train_average << "	RMSE Error Sum: " << TrainErrorSum << "	[RMSE Test Average: " << test_average << " - Benchmark " << MinimumError << "]	RMSE Test Error Sum: " << ErrorSum << " Best RMSE: " << BestRMSE << "   \nNMSE Train Average: " << NMSETrainMean << "  NMSE Train Error Sum: " << NMSETrainSum << " NMSE Test Average: " << NMSETestMean << " NMSE Test Error Sum: " << NMSETestSum << " Best NMSE: " << test_best_nmse << endl;
				out4 << "H: " << hidden << "	Mean Eval: " << MeanEval << " Eval Error: " << EvalSum << "	\nRMSE Train Average: " << train_average << "	RMSE Error Sum: " << TrainErrorSum << "	RMSE Test Average: " << test_average << "	RMSE Test Error Sum: " << ErrorSum << " Best RMSE: " << BestRMSE << "   \nNMSE Train Average: " << NMSETrainMean << "  NMSE Train Error Sum: " << NMSETrainSum << " NMSE Test Average: " << NMSETestMean << " NMSE Test Error Sum: " << NMSETestSum << " Best NMSE: " << test_best_nmse << endl;

				EvalAverage.empty();
				ErrorAverage.empty();
				CycleAverage.empty();
				RMSETrain.empty();
				RMSETest.empty();
				//}//onelevelstop
			}//hidden

			out3 << "\\hline" << endl;

			//close output stream
			out1.close();
			out2.close();
			out3.close();
			out4.close();

			lcout << "Completed " << datasetNames[datasetEnumerator] << " Experiment" << endl;
		}
		catch (exception& e)
		{
			lcout << "Error " << e.what() << endl;
			cout << e.what() << '\n';
		}
	}

	lcout.close();
	rcout.close();



	system("pause");

	return 0;
};


