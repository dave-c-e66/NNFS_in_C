#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

# define W_ROWS 3
# define W_COLS 2
# define I_ROWS 300
# define I_COLS 2
# define CSV_FILE_ROWS 300

struct matrix{
    int rows;
    int columns;
    double **array;

};

// Function prototypes
double dot_product(double inputs[], double weights[], int len);
void multiply_matrices(struct matrix *result_matrix, struct matrix *inputs, struct matrix *weights);
void calculate_layer_output(struct matrix *result_matrix, struct matrix *inputs, struct matrix *weights, struct matrix *bias);
void print_matrix(struct matrix *mat);
void allocate_matrix(struct matrix *m, int rows, int columns);
void free_matrix(struct matrix *m);
void initialize_weights(struct matrix *weights, int rows, int columns);
void relu_activation(struct matrix *m_array, int rows, int columns);
void softmax_activation(struct matrix *m_array, int rows, int columns);


double dot_product(double inputs[], double weights[], int len) {
    double product = 0.0;
    for (int i = 0; i < len; i++) {
        product += inputs[i] * weights[i];
    }
    return product;
}

void print_matrix(struct matrix *mat) {
    //for (int i = 0; i < mat->rows; i++) {
    int iterations;
    iterations = mat->rows;
    if (iterations > 10){
        iterations = 10;
    }
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < mat->columns; j++) {
            printf("%.8f ", mat->array[i][j]);
        }
        printf("\n");
    }
}

void multiply_matrices(struct matrix *result_matrix, struct matrix *inputs, struct matrix *weights) {
    if (inputs->columns != weights->columns) {
        fprintf(stderr, "Error: Columns of inputs (%d) must match columns of weights (%d)\n", inputs->columns, weights->columns);
        return;
    }

    for (int i = 0; i < inputs->rows; i++) {
        for (int j = 0; j < weights->rows; j++) {
            result_matrix->array[i][j] = dot_product(inputs->array[i], weights->array[j], inputs->columns);
        }
    }
}

void calculate_layer_output(struct matrix *result_matrix, struct matrix *inputs, struct matrix *weights, struct matrix *bias) {
    if (inputs->columns != weights->columns) {
        fprintf(stderr, "Error: Columns of inputs (%d) must match columns of weights (%d)\n", inputs->columns, weights->columns);
        return;
    }

    for (int i = 0; i < inputs->rows; i++) {
        for (int j = 0; j < weights->rows; j++) {
            result_matrix->array[i][j] = dot_product(inputs->array[i], weights->array[j], inputs->columns) + bias->array[0][j];
        }
    }
}

void allocate_matrix(struct matrix *m, int rows, int columns) {
    m->rows = rows;
    m->columns = columns;
    m->array = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        m->array[i] = (double *)calloc(columns, sizeof(double)); // Use calloc for initializing with zeros
    }
}

void free_matrix(struct matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        free(m->array[i]);
    }
    free(m->array);
}

void initialize_weights(struct matrix *weights, int rows, int columns) {
    // Set the seed for random number generation
    srand(time(NULL));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            // Generate a random number sampled from a Gaussian distribution
            // Using the Box-Muller transform to approximate the Gaussian distribution
            double u1 = ((double) rand() / (double) RAND_MAX);
            double u2 = ((double) rand() / (double) RAND_MAX);
            double rand_std_normal = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            weights->array[i][j] = rand_std_normal * .01; // Mean is 0 by default
        }
    }
}

void relu_activation(struct matrix *m_array, int rows, int columns){
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (m_array->array[i][j] < 0){
                m_array->array[i][j] = 0;
            }
        }
    }
}

void softmax_activation(struct matrix *m_array, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        double sum_exp = 0.0;
        for (int j = 0; j < columns; j++) {
            m_array->array[i][j] = exp(m_array->array[i][j]);
            sum_exp += m_array->array[i][j];
        }
        for (int j = 0; j < columns; j++) {
            m_array->array[i][j] /= sum_exp;
        }
    }
}

void loadCSV(const char *filename, double x_array[CSV_FILE_ROWS][2], int y_array[CSV_FILE_ROWS]) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    for (int i = 0; i < CSV_FILE_ROWS; i++) {
        double x1, x2;
        int y;
        int result = fscanf(file, "%lf,%lf,%d", &x1, &x2, &y);

        if (result != 3) {
            fprintf(stderr, "Error reading row %d: expected 3 values, got %d\n", i, result);
            break;
        }
        x_array[i][0] = x1;
        x_array[i][1] = x2;
        y_array[i] = y;
    }

    fclose(file);
}

float calc_cat_cross_entrop_loss(struct matrix *output, struct matrix *y_array, int batch_size){
    float loss = .999999;
    float total_loss = 0.0;
    //printf(" y_array->rows %d batch_size %d y_array_columns %d output->columns %d \n", y_array->rows, batch_size, y_array->columns, output->columns);
    for (int i = 0; i < batch_size; i++) {
        // Clip values to prevent division by 0
        for (int j = 0; j < output->columns; j++) {
            if (output->array[i][j] < 1e-7) {
                output->array[i][j] = 1e-7;
            } else if (output->array[i][j] > 1 - 1e-7) {
                output->array[i][j] = 1 - 1e-7;
            }
        }

        float correct_confidence = 0.0;
        if (y_array->columns == batch_size && y_array->rows == 1) {
            // If labels are not one-hot encoded (categorical labels)
            correct_confidence = output->array[i][(int)y_array->array[0][i]];
        } else if (y_array->columns == batch_size && y_array->rows == output->columns) {
            // If labels are one-hot encoded
            for (int j = 0; j < output->columns; j++) {
                correct_confidence += output->array[i][j] * y_array->array[j][i];
            }
        }
        // Calculate the negative log likelihood
        float negative_log_likelihood = -log(correct_confidence);
    
        // Accumulate the loss
        total_loss += negative_log_likelihood;
    }

    // Calculate the mean loss
    loss = total_loss / batch_size;
    
    return loss;
}

int main() {
    double x_array[CSV_FILE_ROWS][2];
    int y_array[CSV_FILE_ROWS];

    loadCSV("spiral_100_data.csv", x_array, y_array);
    /*double m_inputs[I_ROWS][I_COLS] = {
        {1.0, 2.0, 3.0, 2.5},
        {2.0, 5.0, -1.0, 2.0},
        {-1.5, 2.7, 3.3, -0.8}
    };
    double m_weights[W_ROWS][W_COLS] = {
        {0.2, 0.8, -0.5, 1.0},
        {0.5, -0.91, 0.26, -0.5},
        {-0.26, -0.27, 0.17, 0.87}
    };
    double m_bias[W_ROWS] = {2.0, 3.0, 0.5};*/

    struct matrix input_m, weight_m, result_m, bias_m, layer2_m, weight_l2_m, bias_l2_m, label_m;

    allocate_matrix(&input_m, I_ROWS, I_COLS);
    allocate_matrix(&weight_m, W_ROWS, W_COLS);
    allocate_matrix(&result_m, I_ROWS, W_ROWS);
    allocate_matrix(&bias_m, 1, W_ROWS);
    allocate_matrix(&layer2_m, I_ROWS, W_ROWS);
    allocate_matrix(&weight_l2_m, W_ROWS, W_ROWS);
    allocate_matrix(&bias_l2_m, 1, W_ROWS);
    allocate_matrix(&label_m, 1, CSV_FILE_ROWS);
    // Copy data into matrices
    /*for (int i = 0; i < input_m.rows; i++) {
        for (int j = 0; j < input_m.columns; j++) {
            input_m.array[i][j] = m_inputs[i][j];//m_inputs[i][j];
        }
    }*/
    for (int i = 0; i < I_ROWS; i++) {
        for (int j = 0; j < I_COLS; j++) {
            input_m.array[i][j] = x_array[i][j];//m_inputs[i][j];
        }
    }
    /*for (int i = 0; i < weight_m.rows; i++) {
        for (int j = 0; j < weight_m.columns; j++) {
            weight_m.array[i][j] = m_weights[i][j];
        }
    }*/
    initialize_weights(&weight_m, weight_m.rows, weight_m.columns);
    initialize_weights(&weight_l2_m, weight_l2_m.rows, weight_l2_m.columns);

    for (int i = 0; i < bias_m.columns; i++) {
        bias_m.array[0][i] = 0; //m_bias[i];
    }
    for (int i = 0; i < bias_l2_m.columns; i++) {
        bias_l2_m.array[0][i] = 0; //m_bias[i];
    }

    printf("Input matrix:\n");
    //print_matrix(&input_m);
    
    print_matrix(&input_m);
    printf("Weight matrix:\n");
    print_matrix(&weight_m);

    calculate_layer_output(&result_m, &input_m, &weight_m, &bias_m);
    printf("Result matrix after applying weights and bias:\n");
    //result_m.array[0][0] = -1.25;
    print_matrix(&result_m);
    relu_activation(&result_m, I_ROWS, W_ROWS);
    printf("\n After RELU \n");
    print_matrix(&result_m);
    calculate_layer_output(&layer2_m, &result_m, &weight_l2_m, &bias_l2_m);
    printf(" results after layer 2 forward pass \n");
    print_matrix(&layer2_m);
    softmax_activation(&layer2_m, I_ROWS, W_ROWS);
    printf("\n after softmax\n");
    print_matrix(&layer2_m);

    int batch_size = CSV_FILE_ROWS;
    float loss;
    loss = calc_cat_cross_entrop_loss(&layer2_m, &label_m, batch_size);
    printf(" Loss is %f \n", loss);
    // calculate accuracy
    int predictions[CSV_FILE_ROWS];
    int correct = 0;
    for (int i=0; i < CSV_FILE_ROWS; i++){
        int index_predicted;
        float hi_prediction = 0.0;
        for (int j=0; j < W_ROWS; j++){
            if (layer2_m.array[i][j] > hi_prediction){
                hi_prediction = layer2_m.array[i][j];
                index_predicted = j;
            }
        }
        predictions[i] = index_predicted;
        // in y values are one-hot encoded they will need to be converted, need to add code for that.
        if (y_array[i] == index_predicted){
            correct += 1;
        }
        //printf("prediction %d  ground truth %d correct total %d \n", index_predicted, y_array[i], correct);
    }
    printf("correct is %d ", correct);
    float accuracy = (float)correct / CSV_FILE_ROWS;
    printf("Accuracy is : %f \n", accuracy );    
    // Free allocated memory
    free_matrix(&input_m);
    free_matrix(&weight_m);
    free_matrix(&result_m);
    free_matrix(&bias_m);
    //printf("Row %d: x_array = [%lf, %lf], y_array = %d\n", 1, x_array[1][0], x_array[1][1], y_array[1]);
    return 0;
}


