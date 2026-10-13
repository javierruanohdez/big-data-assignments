/*
 * matmul.c - Basic O(n^3) dense matrix multiplication (Assignment 1, C version)
 *
 * Responsibilities are split into small functions (SOLID-lite):
 *   1. load_matrices()   -> read A and B from the shared binary file
 *   2. multiply()        -> the O(n^3) triple-loop kernel (the ONLY timed part)
 *   3. now_seconds()     -> monotonic clock used for the timer boundary
 *   4. checksum()        -> a simple sum of C, used to cross-check languages
 *   5. save_matrix()     -> optionally write C to disk for verify_correctness.py
 *   6. main()            -> orchestration + the timer boundary
 *
 * Matrix file format (shared by all languages, see common/data_generator.py):
 *   Input : int64 n | n*n float64 (A, row-major) | n*n float64 (B, row-major)
 *   Output: int64 n | n*n float64 (C, row-major)
 *   All values little-endian.
 *
 * Memory layout note:
 *   Each matrix is ONE contiguous block of n*n doubles in row-major order,
 *   element (i, j) at index i*n + j. Java (flat double[]) and Python (flat
 *   list) use the same indexing; only Python stores boxed float objects.
 *
 * Build (flags will be fixed and justified in the README):
 *   gcc -O2 -std=c11 -Wall -Wextra -o matmul matmul.c
 * Run:
 *   ./matmul data/matrices_n100.bin [--output result_c.bin]
 *
 * Output (one line on stdout, parsed by run_benchmark.py):
 *   n=<n> time_s=<seconds> checksum=<sum of C>
 */

/* Needed so <time.h> exposes clock_gettime() under strict -std=c11. */
#define _POSIX_C_SOURCE 199309L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
 * and the SAME loop order is used in Python and Java, so all three
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
/* Timer                                                               */
/* ------------------------------------------------------------------ */

/*
 * Current time in seconds from CLOCK_MONOTONIC.
 * A monotonic clock never jumps backwards (unlike the wall clock, which
 * NTP can adjust), so the difference of two readings is a valid duration.
 * This is the same kind of clock as Python's perf_counter_ns() and
 * Java's System.nanoTime().
 */
static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ------------------------------------------------------------------ */
/* Result check and output                                             */
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

/* Writes n and C to `path` (same little-endian format as the input). */
static int save_matrix(const char *path, const double *c, int64_t n)
{
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error: cannot open output file '%s'\n", path);
        return -1;
    }
    size_t count = (size_t)n * (size_t)n;
    int ok = fwrite(&n, sizeof(int64_t), 1, file) == 1 &&
             fwrite(c, sizeof(double), count, file) == count;
    fclose(file);
    if (!ok) {
        fprintf(stderr, "Error: failed to write '%s'\n", path);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Orchestration                                                       */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    /* Arguments: <input.bin> [--output <result.bin>] */
    const char *input_path = NULL;
    const char *output_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (input_path == NULL) {
            input_path = argv[i];
        } else {
            input_path = NULL;   /* unexpected extra argument */
            break;
        }
    }
    if (input_path == NULL) {
        fprintf(stderr, "Usage: %s <matrices_file.bin> [--output <result.bin>]\n", argv[0]);
        return EXIT_FAILURE;
    }

    /* 1. Load inputs - NOT timed (guidelines: inputs prepared outside). */
    int64_t n = 0;
    double *a = NULL;
    double *b = NULL;
    if (load_matrices(input_path, &n, &a, &b) != 0) {
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

    /* ============ TIMED REGION START ============ */
    double start = now_seconds();

    multiply(a, b, c, n);

    double end = now_seconds();
    /* ============ TIMED REGION END ============== */

    /* 3. Report and (optionally) save - NOT timed. */
    printf("n=%lld time_s=%.6f checksum=%.10e\n",
           (long long)n, end - start, checksum(c, n));

    int status = EXIT_SUCCESS;
    if (output_path != NULL && save_matrix(output_path, c, n) != 0) {
        status = EXIT_FAILURE;
    }

    free(a);
    free(b);
    free(c);
    return status;
}
