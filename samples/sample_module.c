#include <stdio.h>

static int evaluate_module(int values[], int size) {
    int score = 0;
    for (int i = 0; i < size; ++i) {
        if (values[i] > 0) {
            score += values[i];
        } else if (values[i] == 0) {
            score += 1;
        } else {
            score -= values[i];
        }
    }

    if (score > 20) {
        score += 3;
    } else if (score > 10) {
        score += 1;
    } else {
        score -= 2;
    }

    return score;
}

int main(void) {
    int data[] = {4, -2, 7, 0, 1, -1, 9, 2};
    int result = evaluate_module(data, 8);
    printf("Module risk probe value: %d\n", result);
    return 0;
}
