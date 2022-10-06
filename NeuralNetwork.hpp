#include <Eigen/Core>
#include <vector>
#include <iostream>

using namespace Eigen;
using namespace std;

// typedefs for ease of coding
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColumnVector;

class NeuralNetork
{
    public:
    NeuralNetwork(vector<int> topology, float learningRate = 0.005);

    void propagateForward(RowVector& input);
    void propagateBackward(RowVector& output);
    void calculateErrors(RowVector& output);
    void updateWeights();
    void trainNN(vecotR<RowVector*> data);
    vector<RowVector*> neuronLayers;

}