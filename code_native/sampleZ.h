// Several functions to sample the weight indicators Z_{i,j} from the
// posterior of the Dirichlet process neural network
//
// @author Wolfgang Roth

#ifndef __SAMPLEZ_H__
#define __SAMPLEZ_H__

int sampleZ_NealAlgorithm8(int* layout, int task, int sharing,
    int activation, int num_layers, int** ZW, int** Zb, double*** W_ptr,
    int*** num_weights_ptr, int *num_unique_weights, double* alpha, double beta,
    double* gamma, double* x, double* t, int N_total, int N_batch, int approx_N,
    int approx_method, int m, int verbosity);

void feedforward(double** W, double** b, double* t, double** a,
    double** z, int* layout, int num_layers, int N, int activation, int task);

double computeErrors(double* t, double* a, int D_o, int N, int task);

double computeErrorsNeuron(double* t, double* a, int D_o, int N, int task,
    int neuron_index);

double computeErrorsNeuronSoftmax(double* t, double* a, int N,
    double* logsumexp, int output_index);

void computeIndividualErrors(double* t, double* a, int D_o, int N, int task,
    double* errors);

void feedforwardIndividualErrors(double** W, double** b, double* t,
    double** a, double** z, int* layout, int num_layers, int N, int activation,
    int task, double* errors);

void feedforwardLayer(double* W, double* b, double* x, double* a, int D_x,
    int D_a, int N);

void feedforwardNeuron(double* W, double* x, double* a, int D_x, int D_a, int N,
    int neuron_index);

void feedforwardNeuronNegative(double* W, double* x, double* a, int D_x,
    int D_a, int N, int neuron_index);

void updateConnection(double* W, double* x, double* a, int D_x, int N,
    int neuron_index_in, int neuron_index_out);

void updateConnectionNegative(double* W, double* x, double* a, int D_x, int N,
    int neuron_index_in, int neuron_index_out);

void updateBias(double* b, double* a, int N, int neuron_index_out);

void updateBiasNegative(double* b, double* a, int N, int neuron_index_out);

void activationFunction(double* a, double* z, int D_a, int N, int activation);

void activationFunctionSingle(double* a, double* z, int D_a, int N,
    int activation, int neuron_index);

// Computes the softmax activation function only for those neurons that
// correspond to the target class in order to save some computation.
void activationFunctionSoftmax(double* a, double* z, int D_a, int N, double* t);

void initActivationFunctionSoftmax(double* a, int D_a, int N,
    double* logsumexp, int output_index);

void activationFunctionSoftmaxLastLayer(double* a, double* z, int D_a, int N,
    double* t, double* max_values, int* max_indices, double* exp_values,
    double* sum_values, int output_index);

void computeApproximation(double** W, double** b, double* t, double** a,
    double** z, int* layout, int num_layers, int N, int activation, int task,
    int approx_neuron, double* approx_likelihoods, int approx_N,
    double* approx_x);

void computePchipCoefficients(double* pchip_delta1, double* pchip_delta2,
    double* pchip_b, double* pchip_c, double* pchip_d, double* lookup,
    double* x_vals, int lookup_N, int lookup_L);

void initReluIdxLookup(int* approx_idx_relu_lookup, int approx_method,
    int lookup_L, double* x_vals);

#endif
