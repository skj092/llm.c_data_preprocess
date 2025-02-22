#include <stdio.h>
#include <stdlib.h>

#define MAX_SAMPLES 120000  // Adjust this based on your dataset
#define MAX_LENGTH 512      // Same as used in Python

int main() {
    // Open the binary files
    FILE *inputFile = fopen("ag_news_train_input_ids.bin", "rb");
    FILE *maskFile = fopen("ag_news_train_attention_masks.bin", "rb");
    FILE *labelFile = fopen("ag_news_train_labels.bin", "rb");

    if (!inputFile || !maskFile || !labelFile) {
        printf("Error opening files\n");
        return 1;
    }

    // Allocate memory
    int *input_ids = (int *)malloc(MAX_SAMPLES * MAX_LENGTH * sizeof(int));
    int *attention_masks = (int *)malloc(MAX_SAMPLES * MAX_LENGTH * sizeof(int));
    int64_t *labels = (int64_t *)malloc(MAX_SAMPLES * sizeof(int64_t));

    // Read data
    fread(input_ids, sizeof(int), MAX_SAMPLES * MAX_LENGTH, inputFile);
    fread(attention_masks, sizeof(int), MAX_SAMPLES * MAX_LENGTH, maskFile);
    fread(labels, sizeof(int64_t), MAX_SAMPLES, labelFile);

    // Print first sample for verification
    printf("First input_ids: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", input_ids[i]);
    }
    printf("\nFirst label: %lld\n", labels[0]);

    // Close files
    fclose(inputFile);
    fclose(maskFile);
    fclose(labelFile);

    // Free memory
    free(input_ids);
    free(attention_masks);
    free(labels);

    return 0;
}

