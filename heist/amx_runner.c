#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    printf("[*] AMX runner active. Sleeping 1s...\n");
    sleep(1);

    int n = 512;
    printf("[*] Allocating matrices for n=%d...\n", n);
    float *a = malloc(n * n * sizeof(float));
    float *b = malloc(n * n * sizeof(float));
    float *c = malloc(n * n * sizeof(float));

    for (int i = 0; i < n * n; i++) {
        a[i] = 1.0f;
        b[i] = 1.0f;
    }

    printf("[*] Running sgemm(%d, %d, %d)...\n", n, n, n);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, a, n, b, n, 0.0f, c, n);
    printf("[*] Done. c[0] = %f\n", c[0]);

    free(a);
    free(b);
    free(c);
    return 0;
}
