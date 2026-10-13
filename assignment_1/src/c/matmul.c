/*
 * matmul.c - Basic O(n^3) dense matrix multiplication (Assignment 1, C version)
 *
 * Responsibilities are split into small functions (SOLID-lite):
 *   1. load_matrices()   -> read A and B from the shared binary file
 *   2. multiply()        -> the O(n^3) triple-loop kernel (the ONLY timed part)
 *   3. checksum()        -> a simple sum of C, used to cross-check languages
 *   4. main()            -> orchestration + the timer boundary
 *
 * Input file format (written by common/data_generator.py):
 *   int64 n | n*n float64 (A, row-major) | n*n float64 (B, row-major)
 *   All values little-endian.
 *
 * Memory layout note (to document in the report):
 *   Matrices are stored as ONE contiguous block of n*n doubles in row-major
 *   order, element (i, j) at index i*n + j. This is the natural C layout.
 *   Java (double[][]) and Python (list of lists) may store rows separately,
 *   which is an unavoidable language-specific difference.
 *
 * Build (flags will be fixed and justified in the README):
 *   gcc -O2 -std=c11 -Wall -Wextra -o matmul matmul.c
 * Run:
 *   ./matmul data/matrices_n100.bin
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Data loading                                                        */
/* ------------------------------------------------------------------ */

/*
 * Reads n, A and B from `path`.
 * On success returns 0 and sets *n_out, *a_out, *b_out (caller must free).
 * On failure returns -1 and prints a message to stderr.
 *
 * Assumption: the machine is little-endian (true for x86-64 and ARM64,
 * i.e. practically every laptop and lab machine). Because of that we can
 * fread() the bytes directly into int64_t / double without conversion.
 */
static int load_matrices(const char *path, int64_t *n_out,
                         double **a_out, double **b_out)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: cannot open input file '%s'\n", path);
        return -1;
    }

    int64_t n = 0;
    if (fread(&n, sizeof(int64_t), 1, file) != 1 || n <= 0) {
        fprintf(stderr, "Error: invalid header in '%s'\n", path);
        fclose(file);
        return -1;
    }

    size_t count = (size_t)n * (size_t)n;   /* number of elements per matrix */
    double *a = malloc(count * sizeof(double));
    double *b = malloc(count * sizeof(double));
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Error: out of memory for n=%lld\n", (long long)n);
        free(a);
        free(b);
        fclose(file);
        return -1;
    }

    /* A and B are stored one after the other, so two block reads suffice. */
    if (fread(a, sizeof(double), count, file) != count ||
        fread(b, sizeof(double), count, file) != count) {
        fprintf(stderr, "Error: file '%s' is truncated\n", path);
        free(a);
        free(b);
        fclose(file);
        return -1;
    }

    fclose(file);
    *n_out = n;
    *a_out = a;
    *b_out = b;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Multiplication kernel                                               */
/* ------------------------------------------------------------------ */

/*
 * C = A x B using the textbook i-j-k triple loop.
 *
 * Why i-j-k: it is the direct translation of the definition
 *     C[i][j] = sum_k A[i][k] * B[k][j]
 * and the SAME loop order will be used in Python and Java, so all three
 * languages do identical work. (Cache-friendlier orders like i-k-j exist,
 * but changing the order would break the fair comparison.)
 *
 * The running sum is kept in a local variable so the inner loop does not
 * write to memory on every iteration - the same choice is made in the
 * other two implementations.
 */
static void multiply(const double *a, const double *b, double *c, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (int64_t k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Result check                                                        */
/* ------------------------------------------------------------------ */

/*
 * Sum of all elements of C. Printing it lets us quickly confirm that
 * C, Java and Python produced the same result (within a tolerance).
 * It also guarantees the compiler cannot optimise the kernel away,
 * because the result is actually used.
 */
static double checksum(const double *c, int64_t n)
{
    double total = 0.0;
    size_t count = (size_t)n * (size_t)n;
    for (size_t idx = 0; idx < count; idx++) {
        total += c[idx];
    }
    return total;
}

/* ------------------------------------------------------------------ */
/* Orchestration                                                       */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <matrices_file.bin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    /* 1. Load inputs - NOT timed (guidelines: inputs prepared outside). */
    int64_t n = 0;
    double *a = NULL;
    double *b = NULL;
    if (load_matrices(argv[1], &n, &a, &b) != 0) {
        return EXIT_FAILURE;
    }

    /* 2. Allocate the output - NOT timed, so we measure only arithmetic.
     *    calloc also touches/zeroes the pages up front. */
    double *c = calloc((size_t)n * (size_t)n, sizeof(double));
    if (c == NULL) {
        fprintf(stderr, "Error: out of memory for result matrix\n");
        free(a);
        free(b);
        return EXIT_FAILURE;
    }

    /* ============ TIMED REGION START ============
     * Next iteration: read a monotonic clock here
     * (clock_gettime(CLOCK_MONOTONIC, ...)) and repeat the call
     * for the required number of repetitions. */

    multiply(a, b, c, n);

    /* ============ TIMED REGION END ==============
     * Next iteration: read the clock again and record the elapsed time. */

    /* 3. Report - NOT timed. */
    printf("n=%lld checksum=%.10e\n", (long long)n, checksum(c, n));

    free(a);
    free(b);
    free(c);
    return EXIT_SUCCESS;
}
