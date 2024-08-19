#include <stdio.h>

float dot(float weights[], float inputs[], int len);




float dot(float weights[], float inputs[], int len){
    
    float product;
    for(int i = 0; i < len; i++ ){
        product += weights[i] * inputs[i];
    }
    return product;
}






int main(){
    float inputs[] = {1.0, 2.0, 3.0, 2.5};
    float weights[] = {0.2, 0.8, -0.5, 1.0};
    float bias = 2.0;
    int len = sizeof(weights) / sizeof(weights[0]);
    float result;
    result = dot(inputs, weights, len);
    printf("%f", result);
    return 0;


}