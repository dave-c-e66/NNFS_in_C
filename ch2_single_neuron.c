#include <stdio.h>
#include <stdlib.h>

# define W_ROWS 3
# define W_COLS 4
# define I_ROWS 3
# define I_COLS 4

struct matrix{
    int rows;
    int columns;
    float **array;

};

//float dot_matrix(struct matrix weights, struct matrix);
float dot(float inputs[], float weights[], int len);
void matrix_p(struct matrix result_matrix, struct matrix inputs, struct matrix weights);
void print_matrix(struct matrix mat);


float dot(float inputs[], float weights[], int len){
    printf("Inputs: %.2f  %.2f  Weights: %.2f  %.2f \n", inputs[0], inputs[1], weights[0], weights[1]);
    float product = 0.0;
    for(int i = 0; i < len; i++ ){
        product += inputs[i] * weights[i];
    }
    printf(" Product %.2f \n", product);
    return product;
}

void print_matrix(struct matrix mat){
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.columns; j++) {
            printf("%.2f ", mat.array[i][j]);
        }
        printf("\n");
    }

}

void matrix_p(struct matrix result_m, struct matrix inputs_m, struct matrix weights_m){
    if (inputs_m.columns == weights_m.columns){
    for (int i = 0; i < inputs_m.rows; i++) {
        for (int j = 0; j < weights_m.rows; j++) {
            //printf("\n inputs %.2f ", inputs_m.array[i][j]);
            //printf(" weights %.2f ", weights_m.array[i][j]);
            result_m.array[i][j] = dot(inputs_m.array[i], weights_m.array[j], I_COLS);
        }
    }
    }
    else{
        printf("ERROR Number of columns in input %d must match number of columns (not transposed) in weights %d \n", inputs_m.columns, weights_m.columns);
    }
}

void calc_layer_output(struct matrix result_m, struct matrix inputs_m, struct matrix weights_m, struct matrix bias_m){
    if (inputs_m.columns == weights_m.columns){
    for (int i = 0; i < inputs_m.rows; i++) {
        for (int j = 0; j < weights_m.rows; j++) {
            //printf("\n inputs %.2f ", inputs_m.array[i][j]);
            //printf(" weights %.2f ", weights_m.array[i][j]);
            result_m.array[i][j] = dot(inputs_m.array[i], weights_m.array[j], I_COLS) + bias_m.array[0][j];
        }
    }
    }
    else{
        printf("ERROR Number of columns in input %d must match number of columns (not transposed) in weights %d \n", inputs_m.columns, weights_m.columns);
    }
}

void calc_layer_outp_old(struct matrix result_m, struct matrix bias_m){
    for (int i = 0; i < result_m.rows; i++) {
        for (int j = 0; j < result_m.columns; j++) {
            result_m.array[i][j] = result_m.array[i][j] + bias_m.array[0][j];
        }
    }
}


void allocate_matrix_mem(struct matrix *m, int rows, int columns) {
    m->rows = rows;
    m->columns = columns;
    m->array = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        m->array[i] = (float *)malloc(columns * sizeof(float));
    }
}

void free_matrix_mem(struct matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        free(m->array[i]);
    }
    free(m->array);
}

void copy_matrix(struct matrix *m, float *original){

}

int main(){
    float inputs[] = {1.0, 2.0, 3.0, 2.5};
    float weights[] = {0.2, 0.8, -0.5, 1.0};
    float bias = 2.0;
    int len = sizeof(weights) / sizeof(weights[0]);
    float result;
    result = dot(inputs, weights, len);
    printf("%f", result);
    float m_inputs[3][4] = {
    {1.0, 2.0, 3.0, 2.5},
    {2.0, 5.0, -1.0, 2.0},
    {-1.5, 2.7, 3.3, -0.8}};
    float m_weights[3][4] = {
    {0.2, 0.8, -0.5, 1.0},
    {0.5, -0.91, 0.26, -0.5},
    {-0.26, -0.27, 0.17, 0.87}};
    float m_bias[3] = {2.0, 3.0, 0.5};
    float m_inputs2[2][3] = {
        {-3, -1, 6},
        {5, 7, -2}
    };
    float m_weights2[4][3] = {
        {0, 3, 1},
        {-3, 4, 7},
        {2, 5, 8},
        {-1, 6, 9}
    };

    struct matrix input_m;
    struct matrix weight_m;
    struct matrix product_m;
    struct matrix input_m2;
    struct matrix weight_m2;
    struct matrix product_m2;
    struct matrix bias_m;
    // do a check that I_COLS == W_COLS
    input_m.rows = I_ROWS;
    input_m.columns = I_COLS;
    weight_m.rows = W_ROWS;
    weight_m.columns = W_COLS;
    product_m.rows = I_ROWS;
    product_m.columns = W_ROWS;
    product_m2.rows = 2;
    product_m2.columns = 4;
    bias_m.rows = 1;
    bias_m.columns = W_ROWS;
    // Dynamically allocate memory for the 2D array
    /*input_m.array = (float **)malloc(input_m.rows * sizeof(float *));
    for (int i = 0; i < input_m.rows; i++) {
        input_m.array[i] = (float *)malloc(input_m.columns * sizeof(float));
    }*/
    allocate_matrix_mem(&input_m, I_ROWS, I_COLS);

    for (int i = 0; i < input_m.rows; i++) {
        for (int j = 0; j < input_m.columns; j++) {
            input_m.array[i][j] = m_inputs[i][j];
        }
    }
    /*weight_m.array = (float **)malloc(weight_m.rows * sizeof(float *));
    for (int i = 0; i < weight_m.rows; i++) {
        weight_m.array[i] = (float *)malloc(weight_m.columns * sizeof(float));
    }*/
    allocate_matrix_mem(&weight_m, W_ROWS, W_COLS);
    for (int i = 0; i < weight_m.rows; i++) {
        for (int j = 0; j < weight_m.columns; j++) {
            weight_m.array[i][j] = m_weights[i][j];
        }
    }
    /*product_m.array = (float **)malloc(product_m.rows * sizeof(float *));
    for (int i = 0; i < product_m.rows; i++) {
        product_m.array[i] = (float *)malloc(product_m.columns * sizeof(float));
    }*/
    allocate_matrix_mem(&product_m, I_ROWS, W_ROWS);
    allocate_matrix_mem(&bias_m, 1, W_ROWS);
    for (int i = 0; i < bias_m.columns; i++) {
        bias_m.array[0][i] = m_bias[i];   
    }   
    ////////////// Work on this part 
    /*
    bias_m.array = malloc(bias_m.columns * sizeof(float));
    for (int i = 0; i < bias_m.columns; i++) {
        bias_m.array[0][i] = m_bias[i];   
    }*/
    /////////////////////////////////////////
    printf("+++++++++++++ START +++++++++++++");
    printf("The input array is:\n");
    print_matrix(input_m);
    printf("The weight array is:\n");
    print_matrix(weight_m);
    //matrix_p(product_m, input_m, weight_m);
    calc_layer_output(product_m, input_m, weight_m, bias_m);
    printf("The product array is:\n");
    print_matrix(product_m);
    printf("bias is \n");
    print_matrix(bias_m);
    allocate_matrix_mem(&product_m2, I_ROWS, W_ROWS);
    
    printf("The final array is:\n");
    print_matrix(product_m);

    allocate_matrix_mem(&input_m2, 2, 3);

    for (int i = 0; i < input_m2.rows; i++) {
        for (int j = 0; j < input_m2.columns; j++) {
            input_m2.array[i][j] = m_inputs2[i][j];
        }
    }

    allocate_matrix_mem(&weight_m2, 4, 3);
    for (int i = 0; i < weight_m2.rows; i++) {
        for (int j = 0; j < weight_m2.columns; j++) {
            weight_m2.array[i][j] = m_weights2[i][j];
        }
    }
    
    
    //matrix_p(product_m2, input_m2, weight_m2);
    //printf("The product array is:\n");
    //print_matrix(product_m2);
    free_matrix_mem(&input_m2);
    free_matrix_mem(&weight_m2);
    free_matrix_mem(&product_m2);    

        // Free allocated memory
    free_matrix_mem(&input_m);
    free_matrix_mem(&weight_m);
    free_matrix_mem(&product_m);
    free_matrix_mem(&bias_m);
    return 0;


}
/*
int main() {
    struct matrix_array matrix;

    // Get the number of rows and columns from the user
    printf("Enter the number of rows: ");
    scanf("%d", &matrix.rows);
    printf("Enter the number of columns: ");
    scanf("%d", &matrix.columns);

    // Dynamically allocate memory for the 2D array
    matrix.array = (float **)malloc(matrix.rows * sizeof(float *));
    for (int i = 0; i < matrix.rows; i++) {
        matrix.array[i] = (float *)malloc(matrix.columns * sizeof(float));
    }

    // Fill the array with some values (for demonstration)
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.columns; j++) {
            matrix.array[i][j] = (float)(i * matrix.columns + j);
        }
    }

    // Print the array
    printf("The array is:\n");
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.columns; j++) {
            printf("%.2f ", matrix.array[i][j]);
        }
        printf("\n");
    }

    // Free the allocated memory
    for (int i = 0; i < matrix.rows; i++) {
        free(matrix.array[i]);
    }
    free(matrix.array);

    return 0;
}*/