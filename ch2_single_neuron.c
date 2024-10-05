#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

# define W_ROWS 3
# define W_COLS 4
# define I_ROWS 3
# define I_COLS 4

struct matrix{
    int rows;
    int columns;
    float **array;

};

// Function prototypes
float dot_product(float inputs[], float weights[], int len);
void multiply_matrices(struct matrix *result_matrix, struct matrix *inputs, struct matrix *weights);
void calculate_layer_output(struct matrix *result_matrix, struct matrix *inputs, struct matrix *weights, struct matrix *bias);
void print_matrix(struct matrix *mat);
void allocate_matrix(struct matrix *m, int rows, int columns);
void free_matrix(struct matrix *m);
void initialize_weights(struct matrix *weights, int rows, int columns);


float dot_product(float inputs[], float weights[], int len) {
    float product = 0.0;
    for (int i = 0; i < len; i++) {
        product += inputs[i] * weights[i];
    }
    return product;
}

void print_matrix(struct matrix *mat) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->columns; j++) {
            printf("%.4f ", mat->array[i][j]);
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
    m->array = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        m->array[i] = (float *)calloc(columns, sizeof(float)); // Use calloc for initializing with zeros
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
            float u1 = ((float) rand() / (float) RAND_MAX);
            float u2 = ((float) rand() / (float) RAND_MAX);
            float rand_std_normal = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            weights->array[i][j] = rand_std_normal * .01; // Mean is 0 by default
        }
    }
}


int main() {
    float m_inputs[I_ROWS][I_COLS] = {
        {1.0, 2.0, 3.0, 2.5},
        {2.0, 5.0, -1.0, 2.0},
        {-1.5, 2.7, 3.3, -0.8}
    };
    float m_weights[W_ROWS][W_COLS] = {
        {0.2, 0.8, -0.5, 1.0},
        {0.5, -0.91, 0.26, -0.5},
        {-0.26, -0.27, 0.17, 0.87}
    };
    float m_bias[W_ROWS] = {2.0, 3.0, 0.5};

    struct matrix input_m, weight_m, result_m, bias_m;

    allocate_matrix(&input_m, I_ROWS, I_COLS);
    allocate_matrix(&weight_m, W_ROWS, W_COLS);
    allocate_matrix(&result_m, I_ROWS, W_ROWS);
    allocate_matrix(&bias_m, 1, W_ROWS);

    // Copy data into matrices
    for (int i = 0; i < input_m.rows; i++) {
        for (int j = 0; j < input_m.columns; j++) {
            input_m.array[i][j] = m_inputs[i][j];
        }
    }
    /*for (int i = 0; i < weight_m.rows; i++) {
        for (int j = 0; j < weight_m.columns; j++) {
            weight_m.array[i][j] = m_weights[i][j];
        }
    }*/
    initialize_weights(&weight_m, weight_m.rows, weight_m.columns);

    for (int i = 0; i < bias_m.columns; i++) {
        bias_m.array[0][i] = m_bias[i];
    }

    printf("Input matrix:\n");
    print_matrix(&input_m);
    printf("Weight matrix:\n");
    print_matrix(&weight_m);

    calculate_layer_output(&result_m, &input_m, &weight_m, &bias_m);
    printf("Result matrix after applying weights and bias:\n");
    print_matrix(&result_m);

    // Free allocated memory
    free_matrix(&input_m);
    free_matrix(&weight_m);
    free_matrix(&result_m);
    free_matrix(&bias_m);

    return 0;
}
/*    float random_floats[5];
    srand(time(NULL)); // Seed the random number generator

    for (int i = 0; i < 5; i++) {
        random_floats[i] = (float)rand() / RAND_MAX; // Generate a random float between 0 and 1
    }

    // Print the array to verify
    for (int i = 0; i < 5; i++) {
        printf("%f\n", random_floats[i]);
    }*/