// Common definitions for the Dirichlet process neural network model
//
// @author Wolfgang Roth

#ifndef __DEFINITIONS_H__
#define __DEFINITIONS_H__

#define TASK_REGRESSION                1
#define TASK_BINARY_CLASSIFICATION     2
#define TASK_MULTICLASS_CLASSIFICATION 3

#define SHARING_LAYERWISE 1
#define SHARING_GLOBAL    2

#define ACTIVATION_NONE    0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH    2
#define ACTIVATION_RELU    3
#define ACTIVATION_SOFTMAX 4

#define INTERPOLATE_NONE    0
#define INTERPOLATE_NEAREST 1
#define INTERPOLATE_LINEAR  2
#define INTERPOLATE_PCHIP   3

#define VERBOSITY_NONE   0  // Only displays implementation errors and fatal errors
#define VERBOSITY_LAYER  1
#define VERBOSITY_NEURON 2
#define VERBOSITY_EDGE   3
#define VERBOSITY_ALL    4

#endif
