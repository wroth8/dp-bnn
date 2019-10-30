// Several useful functions
//
// @author Wolfgang Roth

#ifndef __UTILS_H__
#define __UTILS_H__

// printf wrapper which accepts a verbosity level. If verbosity is less than
// verbosity_needed, the function will not display anything. It uses either the
// usual printf or the mexPrintf function.
void _printf_verbosity(int verbosity, int verbosity_needed, const char* format, ...);

// printf wrapper which uses either the usual printf or the mexPrintf
// function.
void _printf(const char* str, ...);

// realloc wrapper which uses either the usual calloc or the mxRealloc function
void* _realloc(void* ptr, int size);

// Returns the current timestamp in milliseconds.
unsigned long long get_time_in_ms();

// Returns the next largest power of two of the given argument
int get_next_power2(int num);

// Returns the position of the leftmost set bit (v should not be zero)
int log2int(unsigned int v);

void my_tanh_init();

double my_tanh(double x);

void my_sigmoid_init();

double my_sigmoid(double x);

void my_log01_init();

double my_log01(double x);

void softplus_approx_init();

double softplus_approx(double x);

double softplus(double x);

// Spline approximation of the softplus function for non-positive x.
// Note: It is not checked whether x is really non-positive.
double softplus_negative_approx(double x);

// Frees the model parameters if the corresponding pointers are not NULL and
// sets the points to NULL afterwards.
void free_model(int num_layers, int sharing, int** layout, int*** ZW, int*** Zb,
    double*** W, int*** num_weights, int** num_unique_weights, double** alpha,
    double** gamma);

// Loads a model that was stored in Matlab
int load_model(char* file, int** layout, int* task, int* sharing,
    int* activation, int* num_layers, int*** ZW, int*** Zb, double*** W,
    int*** num_weights, int** num_unique_weights, double** alpha, double* beta,
    double** gamma);

// Loads data that was stored in Matlab
int load_data(const char* file, double** x, double** t, int* D_x, int* D_t,
    int* N);

#endif
