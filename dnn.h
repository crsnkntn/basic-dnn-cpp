#include <iostream>
#include <fstream>
#include <math.h>
#include <random>
/*
    The activation function must be of type double.
*/
typedef double(*activation_function)(double);

template <typename InputToken, typename OutputToken, activation_function activ_func>
class DNN {
protected:
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
    double errorSum = 0.0;

    // Constants
    const double A = 0.1, B = 0.1;


public:
    /* 
        Define how the input layer of the dnn is filled given an Input Token.
    */
    void process_input_token (InputToken input_token);

    /* 
        Define how to construct an Output Token using the output layer.
    */
    OutputToken extract_output_token ();

    /* 
        This is used before back propogation. Fill the output_delta_layer according to how close the output layer is to the correct Output Token.
    */
    void fill_output_delta (OutputToken correct);

public:
    DNN () = default;

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

        randomize_weights(s);
    }

    DNN (std::ifstream& fin) {
        // Input the Size of the DNN
        int i = 0;
        fin >> i;
        i_layer_sz = i;
        fin >> i;
        o_layer_sz = i;
        fin >> i;
        h_layer_sz = i;
        fin >> i;
        n_hidden_layers = i;

        // Allocate the memory for the dnn
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

        // Fill the allocated memory with the correct values
        double d = 0.0;
        for (int i = 0; i < i_layer_sz; i++) {
            for (int j = 0; j < h_layer_sz; j++) {
                fin >> weights[0][i][j];
                delta_weights[0][i][j] = 0.0;
            }
        }

        for (int i = 1; i < n_hidden_layers; i++) {
            for (int j = 0; j < h_layer_sz; j++) {
                for (int k = 0; k < h_layer_sz; k++) {
                    fin >> weights[i][j][k];
                    delta_weights[i][j][k] = 0.0;
                }
            }
        }

        for (int j = 0; j < h_layer_sz; j++) {
            for (int k = 0; k < o_layer_sz; k++) {
                fin >> weights[n_hidden_layers][j][k];
                delta_weights[n_hidden_layers][j][k] = 0.0;
            }
        }

        for (int i = 0; i < n_hidden_layers; i++) {
            for (int j = 0; j < h_layer_sz; j++) {
                fin >> delta_hidden_layers[i][j];
            }
        }

        for (int i = 0; i < o_layer_sz; i++) {
            delta_output_layer[i] = 0.0;
        }

        fin.close();
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

    void train_with_current_input (OutputToken correct) {
        do_feed_forward();

        fill_output_delta(correct);

        do_back_propogation();

        do_stochastic_gradient_descent();
    }

    void save_current_dnn (std::string file_name) {
        file_name = "trained_dnns/" + file_name + ".dnn";
        std::ofstream fout(file_name);

        fout << i_layer_sz << '\n';
        fout << o_layer_sz << '\n';
        fout << h_layer_sz << '\n';
        fout << n_hidden_layers << '\n';

        for (int i = 0; i < i_layer_sz; i++) {
            for (int j = 0; j < h_layer_sz; j++) {
                fout << weights[0][i][j] >> ' ';
            }
        }
        fout << '\n';

        for (int i = 1; i < n_hidden_layers; i++) {
            for (int j = 0; j < h_layer_sz; j++) {
                for (int k = 0; k < h_layer_sz; k++) {
                    fout << weights[i][j][k] >> ' ';
                }
            }
        }
        fout << '\n';

        for (int j = 0; j < h_layer_sz; j++) {
            for (int k = 0; k < o_layer_sz; k++) {
                fout << weights[n_hidden_layers][j][k] >> ' ';
            }
        }
        fout << '\n';

        for (int i = 0; i < n_hidden_layers; i++) {
            for (int j = 0; j < h_layer_sz; j++) {
                fout << delta_hidden_layers[i][j] >> ' ';
            }
        }
        fout << '\n';

        fout.close();
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

    void do_back_propogation () {
        double errorTemp = 0.0;
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
                            B * delta_output_layer[k] * hidden_layers[i - 1][j];
                        weights[i][j][k] -= delta_weights[i][j][k];
                    }
                }
            }
            else {
                for (int j = 0; j < h_layer_sz; j++) {
                    for (int k = 0; k < h_layer_sz; k++) {
                        delta_weights[i][j][k] = 
                            A * delta_weights[i][j][k] + 
                            B * delta_hidden_layers[i - 1][k] * hidden_layers[i - 1][j];
                        weights[i][j][k] -= delta_weights[i][j][k];
                    }
                }
            }
        }
    }
};