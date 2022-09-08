#include <iostream>
#include <math.h>
#include <random>

typedef double(*activation_function)(double);

template <typename InputToken, typename OutputToken, activation_function activ_func>
class DNN {
private:
    // Sizes, needed for iteration bounds
    int i_layer_sz;
    int h_layer_sz;
    int o_layer_sz;
    int n_hidden_layers;

    // Layers
    double* input_layer;
    double* output_layer;
    double** hidden_layers;

    // Weights
    double*** weights;

    // Utility/Temporary
    double* delta_output_layer;
    double** delta_hidden_layers;
    double*** delta_weights;

    // Constants
    double A, B;

public:
    // Virtual Functions
    virtual void process_input_token (InputToken input_token) = 0;

    virtual OutputToken extract_output_token () = 0;

public:
    DNN (int i, int h, int o, int n, int s) : i_layer_sz(i), h_layer_sz(h), o_layer_sz(o), n_hidden_layers(n) {
        input_layer = (double*)malloc(i_layer_sz * sizeof(double));
        output_layer = (double*)malloc(o_layer_sz * sizeof(double));
        delta_output_layer = (double*)malloc(o_layer_sz * sizeof(double));

        hidden_layers = (double**)malloc(n_hidden_layers * sizeof(double*));
        delta_hidden_layers = (double**)malloc(n_hidden_layers * sizeof(double*));
        for (int j = 0; j < n_hidden_layers; j++) {
            hidden_layers[j] = (double*)malloc(h_layer_sz * sizeof(double));
            delta_hidden_layers[j] = (double*)malloc(h_layer_sz * sizeof(double));
        }

        weights = (double***)malloc((n_hidden_layers + 1) * sizeof(double**));
        delta_weights = (double***)malloc((n_hidden_layers + 1) * sizeof(double**));

        weights[0] = (double**)malloc(i_layer_sz * sizeof(double*));
        delta_weights[0] = (double**)malloc(i_layer_sz * sizeof(double*));
        for (int j = 0; j < i_layer_sz; j++) {
            weights[0][j] = (double*)malloc(h_layer_sz * sizeof(double));
            delta_weights[0][j] = (double*)malloc(h_layer_sz * sizeof(double));
        }

        for (int j = 1; j < n_hidden_layers; j++) {
            weights[j] = (double**)malloc(h_layer_sz * sizeof(double*));
            delta_weights[j] = (double**)malloc(h_layer_sz * sizeof(double*));
            for (int k = 0; k < h_layer_sz; k++) {
                weights[j][k] = (double*)malloc(h_layer_sz * sizeof(double));
                delta_weights[j][k] = (double*)malloc(h_layer_sz * sizeof(double));
            }
        }

        weights[n_hidden_layers] = (double**)malloc(h_layer_sz * sizeof(double*));
        delta_weights[n_hidden_layers] = (double**)malloc(h_layer_sz * sizeof(double*));
        for (int j = 0; j < h_layer_sz; j++) {
            weights[n_hidden_layers][j] = (double*)malloc(o_layer_sz * sizeof(double));
            delta_weights[n_hidden_layers][j] = (double*)malloc(o_layer_sz * sizeof(double));
        }

        A = 0.1;
        B = 0.1;
        randomize_weights(s);
    }

    DNN (const DNN& other) {
        i_layer_sz = other.i_layer_sz;
        h_layer_sz = other.h_layer_sz;
        o_layer_sz = other.o_layer_sz;
        n_hidden_layers = other.n_hidden_layers;

        memcpy(&input_layer, &other.input_layer, sizeof(other.input_layer));
        memcpy(&output_layer, &other.output_layer, sizeof(output_layer));
        memcpy(&hidden_layers, &other.hidden_layers, sizeof(hidden_layers));
        memcpy(&weights, &other.weights, sizeof(weights));
        memcpy(&delta_output_layer, &other.delta_output_layer, sizeof(delta_output_layer));
        memcpy(&delta_hidden_layers, &other.delta_hidden_layers, sizeof(delta_hidden_layers));
    }

    ~DNN () {
        free(input_layer);
        free(output_layer);
        free(hidden_layers);

        // Weights
        free(weights);

        // Utility/Temporary
        free(delta_output_layer);
        free(delta_hidden_layers);
        free(delta_weights);
    }

    void randomize_weights (int seed) {
        srand(seed);

        for (int i = 0; i < i_layer_sz; i++) {
            for (int j = 0; j < h_layer_sz; j++) {
                weights[0][i][j] = rand() / (double)(RAND_MAX);
                delta_weights[0][i][j] = 0.0;
            }
        }

        for (int i = 1; i < n_hidden_layers; i++) {
            for (int j = 0; j < h_layer_sz; j++) {
                for (int k = 0; k < h_layer_sz; k++) {
                    weights[i][j][k] = rand() / (double)(RAND_MAX);
                    delta_weights[i][j][k] = 0.0;
                }
            }
        }

        for (int j = 0; j < h_layer_sz; j++) {
            for (int k = 0; k < o_layer_sz; k++) {
                weights[n_hidden_layers][j][k] = rand() / (double)(RAND_MAX);
                delta_weights[n_hidden_layers][j][k] = 0.0;
            }
        }

        for (int i = 0; i < n_hidden_layers; i++) {
            for (int j = 0; j < h_layer_sz; j++) {
                delta_hidden_layers[i][j] = rand() / (double)(RAND_MAX);
            }
        }

        for (int i = 0; i < o_layer_sz; i++) {
            delta_output_layer[i] = 0.0;
        }
    }

    void use_net_with_current_input () {
        do_feed_forward();
    }

    void train_with_current_input (int correct) {
        do_feed_forward();

        do_back_propogation(correct);

        do_stochastic_gradient_descent();
    }

private:
    void do_feed_forward () {
        double sum = 0.0;
        // Feed Forward
        for (int i = 0; i < n_hidden_layers + 1; i++) {
            sum = 0.0;
            if (i == 0) {
                for (int j = 0; j < h_layer_sz; j++) {
                    sum = 0.0;
                    for (int k = 0; k < i_layer_sz; k++) {
                        sum += input_layer[k] * weights[0][k][j];
                    }
                    hidden_layers[0][j] = activ_func(sum);
                }
            }
            else if (i == n_hidden_layers) {
                for (int j = 0; j < o_layer_sz; j++) {
                    sum = 0.0;
                    for (int k = 0; k < h_layer_sz; k++) {
                        sum += hidden_layers[i - 1][k] * weights[i][k][j];
                    }
                    output_layer[j] = activ_func(sum);
                }
            }
            else {
                for (int j = 0; j < h_layer_sz; j++) {
                    sum = 0.0;
                    for (int k = 0; k < h_layer_sz; k++) {
                        sum += hidden_layers[i - 1][k] * weights[i][j][k];
                    }
                    hidden_layers[i][j] = activ_func(sum);
                }
            }
        }
    }

    void do_back_propogation (int correct) {
        double errorTemp = 0.0, errorSum = 0.0;
        
        for (int i = 0; i < o_layer_sz; i++) {
            if (i == correct) {
                errorTemp = 1 - output_layer[i];
            }
            else {
                errorTemp = -output_layer[i];
            }
            delta_output_layer[i] = -errorTemp * activ_func(output_layer[i]) * (1 - activ_func(output_layer[i]));
            errorSum += errorTemp * errorTemp;
        }

        for (int i = n_hidden_layers; i > 0; i--) {
            if (i == n_hidden_layers) {
                for (int j = 0; j < h_layer_sz; j++) {
                    errorTemp = 0.0;
                    for (int k = 0; k < o_layer_sz; k++) {
                        errorTemp += delta_output_layer[k] * weights[i][j][k];
                    }
                    delta_hidden_layers[i - 1][j] = errorTemp * 
                        (1.0 + hidden_layers[i - 1][j]) * (1.0 - hidden_layers[i - 1][j]);
                }
            }
            else {
                for (int j = 0; j < h_layer_sz; j++) {
                    errorTemp = 0.0;
                    for (int k = 0; k < h_layer_sz; k++) {
                        errorTemp += delta_hidden_layers[i][k] * weights[i][j][k];
                    }
                    delta_hidden_layers[i - 1][j] = errorTemp * 
                        (1.0 + hidden_layers[i - 1][j]) * (1.0 - hidden_layers[i - 1][j]);
                }
            }
        }
    }

    void do_stochastic_gradient_descent () {
        for (int i = n_hidden_layers; i > 0; i--) {
            if (i == n_hidden_layers) {
                for (int j = 0; j < h_layer_sz; j++) {
                    for (int k = 0; k < o_layer_sz; k++) {
                        delta_weights[i][j][k] = 
                            A * delta_weights[i][j][k] + 
                            B * delta_output_layer[k] * hidden_layers[i - 1][j]; // index problem?
                        weights[i][j][k] -= delta_weights[i][j][k]; // order of brackets?
                    }
                }
            }
            else {
                for (int j = 0; j < h_layer_sz; j++) {
                    for (int k = 0; k < h_layer_sz; k++) {
                        delta_weights[i][j][k] = 
                            A * delta_weights[i][j][k] + 
                            B * delta_hidden_layers[i - 1][k] * hidden_layers[i - 1][j]; // index problem?
                        weights[i][j][k] -= delta_weights[i][j][k]; // order of brackets?
                    }
                }
            }
        }
    }
};