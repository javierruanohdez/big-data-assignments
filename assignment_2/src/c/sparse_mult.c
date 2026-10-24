/*
 * sparse_mult.c - Sparse matrix multiplication in CSR format (Assignment 2, Part B)
 *
 * Pipeline (each phase timed separately with CLOCK_MONOTONIC):
 *   1. load      : read dense A and B from the shared binary file   (not timed)
 *   2. convert   : dense A -> CSR A, dense B -> CSR B                (conversion_time_s)
 *   3. multiply  : C = A x B with A, B and C in CSR                  (time_s)
 *   4. report    : nnz of A, B, C, storage size, times, checksum    (not timed)
 *
 * Why the input is dense: all programs (dense and sparse) read the SAME
 * files from common/data_generator.py, so the CSR kernel and the dense
 * kernels multiply exactly the same logical matrices. The price of getting
 * from that dense input to CSR is the conversion cost, which the
 * guidelines require us to report separately from the multiplication.
 *
 * Responsibilities (SOLID-lite, same structure as dense_mult.c):
 *   load_matrices()          -> read A and B from the shared binary file
 *   dense_to_csr()           -> format conversion (timed: conversion_time_s)
 *   csr_multiply()           -> CSR x CSR kernel (timed: time_s)
 *   csr_product_nnz_bound()  -> size of the output allocation (part of the kernel)
 *   csr_row_to_dense()       -> expand one CSR row (used by checksum/output)
 *   csr_checksum(), save_csr_as_dense(), csr_storage_bytes() -> reporting
 *   main()                   -> orchestration + both timer boundaries
 *
 * Build (same flags as dense_mult.c; fixed in the benchmark config):
 *   gcc -O2 -std=c11 -Wall -Wextra -o sparse_mult sparse_mult.c
 * Run:
 *   ./sparse_mult <matrices.bin> [--output result.bin]
 *
 * Output (one line on stdout, space-separated key=value pairs):
 *   algo=csr n=500 nnz_A=25000 nnz_B=25000 nnz_C=118000 csr_bytes=2178012
 *   conversion_time_s=0.001234567 time_s=0.004567890 checksum=1.2345678901e+02
 *   (printed on a single line). csr_bytes is the storage of the three CSR
 *   matrices A, B and C. Times have 9 decimals (nanosecond resolution).
 *   --output writes C in the dense output format, so verify_correctness.py
 *   can compare it with the dense kernels directly.
 */

/* Needed so <time.h> exposes clock_gettime() under strict -std=c11. */
#define _POSIX_C_SOURCE 199309L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Keep the timed functions out of main(), exactly as in dense_mult.c:
 * a function inlined into main() can be compiled worse because of main()'s
 * register pressure, and we want to measure the algorithm, not that effect.
 */
#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

/* ------------------------------------------------------------------ */
/* The CSR format                                                      */
/* ------------------------------------------------------------------ */

/*
 * Compressed Sparse Row (CSR): only the non-zero entries are stored, row
 * by row, in three arrays.
 *
 *   Dense (4 x 4)        CSR
 *   [ 5 0 0 1 ]          values  = [ 5  1 | 2 | 3  4 | 6 ]
 *   [ 0 0 2 0 ]          col_idx = [ 0  3 | 2 | 0  1 | 3 ]
 *   [ 3 4 0 0 ]          row_ptr = [ 0      2   3      5   6 ]
 *   [ 0 0 0 6 ]
 *
 *   values[p]  : the p-th non-zero, counting row by row (left to right)
 *   col_idx[p] : the column of that non-zero
 *   row_ptr[i] : position in values/col_idx where row i starts. Row i owns
 *                the half-open range [row_ptr[i], row_ptr[i+1]), so
 *                row_ptr has n+1 entries, row_ptr[0] = 0, row_ptr[n] = nnz,
 *                and row i has row_ptr[i+1] - row_ptr[i] non-zeros
 *                (an empty row simply repeats the previous value).
 *
 * Visiting row i therefore looks like:
 *   for (p = row_ptr[i]; p < row_ptr[i+1]; p++)
 *       use column col_idx[p] and value values[p];
 *
 * Index types (memory matters, Peak RSS is a measured metric):
 *   col_idx is int32_t: a column index is < n, and we check n <= INT32_MAX
 *   when loading. This is 4 bytes per non-zero instead of 8 (SciPy and MKL
 *   also use 32-bit indices by default).
 *   row_ptr is int64_t: it counts non-zeros, which can reach n*n.
 *
 * Storage: 12 bytes per non-zero (8 value + 4 column) + 8*(n+1) for row_ptr,
 * versus 8*n*n for a dense matrix. CSR is therefore smaller as soon as
 * fewer than about 2/3 of the entries are non-zero (sparsity > ~33 %).
 */
typedef struct {
    int64_t n;          /* the matrix is n x n */
    int64_t nnz;        /* number of stored (non-zero) entries */
    int64_t *row_ptr;   /* n + 1 entries */
    int32_t *col_idx;   /* nnz entries */
    double *values;     /* nnz entries */
} CsrMatrix;

/* An empty matrix: all pointers NULL, so csr_free() on it is always safe. */
static const CsrMatrix CSR_EMPTY = {0, 0, NULL, NULL, NULL};

static void csr_free(CsrMatrix *m)
{
    free(m->row_ptr);
    free(m->col_idx);
    free(m->values);
    *m = CSR_EMPTY;
}

/* Bytes used by the three arrays of a CSR matrix (logical size, nnz entries). */
static int64_t csr_storage_bytes(const CsrMatrix *m)
{
    return (m->n + 1) * (int64_t)sizeof(int64_t)
         + m->nnz * (int64_t)(sizeof(int32_t) + sizeof(double));
}

/* ------------------------------------------------------------------ */
/* Memory helper                                                       */
/* ------------------------------------------------------------------ */

/*
 * malloc() for an array of `count` elements, with two safety checks:
 *  - count * elem_size must not overflow size_t (otherwise malloc would get
 *    a wrapped, too-small size and later writes would run past the block);
 *  - malloc(0) may legally return NULL, which would look like an error for
 *    an all-zero matrix (nnz = 0). We always ask for at least one element,
 *    so NULL means "out of memory" and nothing else.
 */
static void *allocate_array(size_t count, size_t elem_size)
{
    if (count == 0) {
        count = 1;
    }
    if (count > SIZE_MAX / elem_size) {
        return NULL;
    }
    return malloc(count * elem_size);
}

/* ------------------------------------------------------------------ */
/* Data loading (as in dense_mult.c, plus the size checks CSR needs)   */
/* ------------------------------------------------------------------ */

/*
 * Reads n, A and B (dense, row-major) from `path`.
 * On success returns 0 and sets *n_out, *a_out, *b_out (caller must free).
 * On failure returns -1, prints a message to stderr and leaks nothing.
 * Assumes a little-endian machine (x86-64, ARM64), as in Assignment 1.
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
    if (fread(&n, sizeof(int64_t), 1, file) != 1 || n <= 0 || n > INT32_MAX) {
        fprintf(stderr, "Error: invalid header in '%s' (need 0 < n <= %d)\n",
                path, INT32_MAX);
        fclose(file);
        return -1;
    }

    size_t count = (size_t)n * (size_t)n;   /* fits: n <= INT32_MAX */
    double *a = allocate_array(count, sizeof(double));
    double *b = allocate_array(count, sizeof(double));
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Error: out of memory for n=%lld\n", (long long)n);
        free(a);
        free(b);
        fclose(file);
        return -1;
    }

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
/* Conversion: dense -> CSR (timed as conversion_time_s)               */
/* ------------------------------------------------------------------ */

/*
 * Builds the CSR form of a dense, row-major n x n matrix in two passes.
 *
 *   Pass 1 - count the non-zeros of each row and turn the counts into
 *            row_ptr with a running total (prefix sum):
 *                row_ptr[i+1] = row_ptr[i] + (non-zeros in row i)
 *            Now nnz = row_ptr[n] is known exactly, so col_idx and values
 *            can be allocated at their exact size.
 *   Pass 2 - copy every non-zero and its column into the next free slot p.
 *            p runs 0, 1, ..., nnz-1 in row order, so row i lands exactly in
 *            [row_ptr[i], row_ptr[i+1]). Columns are visited left to right,
 *            so the columns inside each row come out sorted.
 *
 * Why two passes: reading the dense matrix twice costs n*n extra reads.
 * The one-pass alternative must reserve room for n*n entries up front,
 * which wastes the most memory at high sparsity - exactly where CSR should
 * shine.
 *
 * Cost: O(n^2) whatever the sparsity, because every zero still has to be
 * looked at. At high sparsity this can exceed the multiplication itself,
 * which is why the guidelines want it reported separately. The constant
 * factor still varies with sparsity: the `!= 0.0` test is a branch, and the
 * CPU predicts it worst when about half of the entries are zero, so expect
 * the conversion to be slowest around 50 % sparsity.
 *
 * Allocation is part of the conversion and therefore inside its timer.
 * Returns 0 on success, -1 if out of memory (nothing is leaked).
 */
static NOINLINE int dense_to_csr(const double *dense, int64_t n, CsrMatrix *out)
{
    int64_t *row_ptr = allocate_array((size_t)n + 1, sizeof(int64_t));
    if (row_ptr == NULL) {
        return -1;
    }

    /* Pass 1: count non-zeros per row -> row_ptr. */
    row_ptr[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t row_count = 0;
        for (int64_t j = 0; j < n; j++) {
            if (dense[i * n + j] != 0.0) {
                row_count++;
            }
        }
        row_ptr[i + 1] = row_ptr[i] + row_count;
    }
    const int64_t nnz = row_ptr[n];

    int32_t *col_idx = allocate_array((size_t)nnz, sizeof(int32_t));
    double *values = allocate_array((size_t)nnz, sizeof(double));
    if (col_idx == NULL || values == NULL) {
        free(row_ptr);
        free(col_idx);
        free(values);
        return -1;
    }

    /* Pass 2: copy the non-zeros. p = next free slot in col_idx/values. */
    int64_t p = 0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            const double value = dense[i * n + j];
            if (value != 0.0) {
                col_idx[p] = (int32_t)j;    /* safe: j < n <= INT32_MAX */
                values[p] = value;
                p++;
            }
        }
    }
    /* Here p == nnz: pass 2 found exactly the non-zeros pass 1 counted. */

    out->n = n;
    out->nnz = nnz;
    out->row_ptr = row_ptr;
    out->col_idx = col_idx;
    out->values = values;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Multiplication: CSR x CSR -> CSR (timed as time_s)                  */
/* ------------------------------------------------------------------ */

/*
 * Upper bound on nnz(C), used to allocate C once before the kernel runs.
 *
 * Row i of C is built from the rows k of B for which A[i][k] != 0, so it
 * can have at most
 *     min( n , sum over those k of nnz(row k of B) )
 * entries: each row of B contributes at most its own non-zeros, and a row
 * of C has only n columns. Summing over all rows gives the bound.
 *
 * It only reads row lengths (row_ptr), never B's column indices, so it costs
 * O(nnz_A + n). The alternatives are a full "symbolic" pass over the
 * product (as expensive as the multiplication's index work) or growing the
 * arrays with realloc during the kernel. For our uniformly random inputs the
 * bound is close to the real nnz(C), because few products land on the
 * same (i, j) unless the row is nearly full anyway (and then n caps it).
 * At n = 1000 it was within 10 % of nnz(C) for every sparsity we tried.
 */
static int64_t csr_product_nnz_bound(const CsrMatrix *a, const CsrMatrix *b)
{
    int64_t total = 0;
    for (int64_t i = 0; i < a->n; i++) {
        int64_t row_bound = 0;
        for (int64_t p = a->row_ptr[i]; p < a->row_ptr[i + 1]; p++) {
            const int64_t k = a->col_idx[p];
            row_bound += b->row_ptr[k + 1] - b->row_ptr[k];   /* nnz of row k of B */
        }
        total += (row_bound < a->n) ? row_bound : a->n;
    }
    return total;
}

/*
 * C = A x B, all three in CSR.
 * Algorithm: Gustavson's row-by-row method (Gustavson, 1978) with a dense
 * "sparse accumulator" (SPA, Gilbert, Moler & Schreiber, 1992). SciPy's
 * sparse product uses the same approach.
 *
 * The idea is the same "row view" as the i-k-j dense kernel:
 *     row i of C = sum over k with A[i][k] != 0 of  A[i][k] * (row k of B)
 * but it skips every k where A[i][k] == 0 and every j where B[k][j] == 0,
 * because CSR only lists the non-zeros. Rows of A and rows of B are exactly
 * what CSR gives us cheaply: no column access is ever needed.
 *
 * The SPA = three work arrays of length n, reused for every row i:
 *   acc[j]      running value of C[i][j]; all 0.0 between rows
 *   marker[j]   the last row that touched column j. "marker[j] == i" means
 *               "already seen in this row", so we never clear a flag array
 *   touched[t]  the columns touched in the current row, in first-touch order
 *
 * For each row i:
 *   Scatter - for every non-zero A[i][k], walk row k of B and add
 *             A[i][k] * B[k][j] into acc[j]. The first time a column j
 *             shows up in this row, append it to touched[].
 *   Gather  - for each touched column j, append (j, acc[j]) to C and reset
 *             acc[j] to 0.0. Only touched columns are visited, so a row
 *             costs time proportional to its products, not to n.
 *
 * Cost: O(n + nnz_A + products), where products = the number of non-zero
 * products A[i][k] * B[k][j] = sum over non-zeros A[i][k] of nnz(row k of B).
 * For random inputs with density d = 1 - sparsity, products ~ d^2 * n^3:
 *   sparsity 50 % -> 4x fewer multiply-adds than dense,
 *   sparsity 90 % -> 100x fewer,  sparsity 99 % -> 10,000x fewer.
 * The price is paid per product. Before acc[j] can be touched, j must be
 * loaded from col_idx (an indirect access), and the marker test adds a
 * load, a compare and a branch. The compiler cannot vectorise this loop,
 * and in very sparse rows the accesses to acc jump around instead of
 * streaming. (B's columns are sorted, so in fairly dense rows acc is still
 * visited left to right, which keeps the hardware prefetcher useful.)
 * One sparse step therefore costs clearly more than one step of the dense
 * inner loop; the sparsity at which the saved products outweigh this
 * overhead is the crossover we measure.
 *
 * Details worth knowing:
 *   - Fill-in: C is usually much denser than A and B (one non-zero
 *     A[i][k] spreads a whole row of B into row i of C). nnz(C) is reported.
 *   - Exact zeros are dropped in the gather (a touched entry whose terms
 *     cancel to exactly 0.0), so nnz_C is the true number of non-zeros,
 *     the same number one would count in the dense result.
 *   - Same values as the dense kernels: A's columns are sorted (from
 *     dense_to_csr), so acc[j] receives its terms in increasing k, the same
 *     order as in dense_mult.c, minus the terms that are exactly zero.
 *   - The columns inside each row of C are NOT sorted (first-touch order).
 *     CSR does not require sorted columns, and sorting would add
 *     O(r log r) work per row for nothing we need here.
 *
 * Everything the kernel needs is inside the timer: the size bound and the
 * allocation of C and the SPA. A dense kernel knows its output size (n*n)
 * before it starts; a sparse kernel does not, so finding and reserving the
 * output space is part of the real cost of the sparse method.
 *
 * Returns 0 on success, -1 if out of memory (nothing is leaked).
 */
static NOINLINE int csr_multiply(const CsrMatrix *a, const CsrMatrix *b, CsrMatrix *c)
{
    const int64_t n = a->n;
    const int64_t capacity = csr_product_nnz_bound(a, b);

    /* Output arrays (sized by the bound) and the three SPA work arrays. */
    int64_t *row_ptr = allocate_array((size_t)n + 1, sizeof(int64_t));
    int32_t *col_idx = allocate_array((size_t)capacity, sizeof(int32_t));
    double *values = allocate_array((size_t)capacity, sizeof(double));
    double *acc = calloc((size_t)n, sizeof(double));        /* all 0.0 */
    int64_t *marker = allocate_array((size_t)n, sizeof(int64_t));
    int32_t *touched = allocate_array((size_t)n, sizeof(int32_t));

    if (row_ptr == NULL || col_idx == NULL || values == NULL ||
        acc == NULL || marker == NULL || touched == NULL) {
        free(row_ptr);
        free(col_idx);
        free(values);
        free(acc);
        free(marker);
        free(touched);
        return -1;
    }

    for (int64_t j = 0; j < n; j++) {
        marker[j] = -1;                 /* no row has touched any column yet */
    }

    int64_t nnz = 0;                    /* entries written to C so far */
    row_ptr[0] = 0;

    for (int64_t i = 0; i < n; i++) {
        int64_t num_touched = 0;

        /* Scatter: acc += A[i][k] * (row k of B), for every non-zero A[i][k]. */
        for (int64_t p = a->row_ptr[i]; p < a->row_ptr[i + 1]; p++) {
            const int64_t k = a->col_idx[p];
            const double a_ik = a->values[p];

            for (int64_t q = b->row_ptr[k]; q < b->row_ptr[k + 1]; q++) {
                const int32_t j = b->col_idx[q];

                if (marker[j] != i) {           /* first time column j appears in row i */
                    marker[j] = i;
                    touched[num_touched] = j;
                    num_touched++;
                }
                acc[j] += a_ik * b->values[q];
            }
        }

        /* Gather: move the touched entries into C and reset them in acc.
         * Memory safety: row i stores at most num_touched entries, and
         * num_touched <= min(n, products of row i) = this row's share of
         * `capacity`, so nnz never exceeds the allocated size. */
        for (int64_t t = 0; t < num_touched; t++) {
            const int32_t j = touched[t];

            if (acc[j] != 0.0) {                /* drop exact cancellations */
                col_idx[nnz] = j;
                values[nnz] = acc[j];
                nnz++;
            }
            acc[j] = 0.0;                       /* ready for the next row */
        }
        row_ptr[i + 1] = nnz;
    }

    free(acc);
    free(marker);
    free(touched);

    c->n = n;
    c->nnz = nnz;
    c->row_ptr = row_ptr;
    c->col_idx = col_idx;
    c->values = values;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Timer (as in dense_mult.c)                                          */
/* ------------------------------------------------------------------ */

/* Seconds from CLOCK_MONOTONIC: never jumps backwards, so differences are valid durations. */
static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ------------------------------------------------------------------ */
/* Result check and output (not timed)                                 */
/* ------------------------------------------------------------------ */

/* Writes row i of a CSR matrix into `row` (n doubles) as a dense row. */
static void csr_row_to_dense(const CsrMatrix *m, int64_t i, double *row)
{
    memset(row, 0, (size_t)m->n * sizeof(double));
    for (int64_t p = m->row_ptr[i]; p < m->row_ptr[i + 1]; p++) {
        row[m->col_idx[p]] = m->values[p];
    }
}

/*
 * Sum of all n*n elements of C in row-major order, zeros included.
 * Each row is expanded into a dense buffer and added left to right: that is
 * the same sequence of additions as checksum() in dense_mult.c, so for the
 * same input both programs print the SAME checksum. (Summing values[]
 * directly would add in first-touch order and differ in the last digits.)
 * Cost O(n^2), outside the timers.
 */
static double csr_checksum(const CsrMatrix *m, double *row_buffer)
{
    double total = 0.0;
    for (int64_t i = 0; i < m->n; i++) {
        csr_row_to_dense(m, i, row_buffer);
        for (int64_t j = 0; j < m->n; j++) {
            total += row_buffer[j];
        }
    }
    return total;
}

/*
 * Writes C in the dense output format of dense_mult.c (int64 n | n*n float64),
 * one row at a time, so no full dense copy of C is ever allocated.
 */
static int save_csr_as_dense(const char *path, const CsrMatrix *m, double *row_buffer)
{
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error: cannot open output file '%s'\n", path);
        return -1;
    }

    int ok = fwrite(&m->n, sizeof(int64_t), 1, file) == 1;
    for (int64_t i = 0; ok && i < m->n; i++) {
        csr_row_to_dense(m, i, row_buffer);
        ok = fwrite(row_buffer, sizeof(double), (size_t)m->n, file) == (size_t)m->n;
    }

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

static void print_usage(const char *program)
{
    fprintf(stderr, "Usage: %s <matrices_file.bin> [--output <result.bin>]\n", program);
}

int main(int argc, char *argv[])
{
    /* Arguments: <input.bin> [--output <result.bin>] */
    const char *input_path = NULL;
    const char *output_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (argv[i][0] != '-' && input_path == NULL) {
            input_path = argv[i];
        } else {
            fprintf(stderr, "Error: unexpected or incomplete argument '%s'\n", argv[i]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (input_path == NULL) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    /*
     * Every resource starts empty (NULL), and all exits go through the
     * single `cleanup` label below, which frees everything exactly once.
     * free(NULL) and csr_free() on an empty matrix do nothing, so this is
     * safe on every error path. All variables are declared here, before the
     * first `goto`, so no jump ever skips an initialisation.
     */
    int status = EXIT_FAILURE;
    int64_t n = 0;
    double *a_dense = NULL;
    double *b_dense = NULL;
    double *row_buffer = NULL;
    CsrMatrix a_csr = CSR_EMPTY;
    CsrMatrix b_csr = CSR_EMPTY;
    CsrMatrix c_csr = CSR_EMPTY;
    int conversion_status = 0;
    int multiply_status = 0;
    double conversion_start = 0.0, conversion_end = 0.0;
    double multiply_start = 0.0, multiply_end = 0.0;
    int64_t csr_bytes = 0;

    /* 1. Load the dense inputs - NOT timed. */
    if (load_matrices(input_path, &n, &a_dense, &b_dense) != 0) {
        goto cleanup;
    }

    /* ============ TIMED REGION 1: CONVERSION START ============ */
    conversion_start = now_seconds();

    conversion_status = dense_to_csr(a_dense, n, &a_csr);
    if (conversion_status == 0) {
        conversion_status = dense_to_csr(b_dense, n, &b_csr);
    }

    conversion_end = now_seconds();
    /* ============ TIMED REGION 1: CONVERSION END ============== */

    if (conversion_status != 0) {
        fprintf(stderr, "Error: out of memory during dense -> CSR conversion\n");
        goto cleanup;
    }

    /*
     * The multiplication never touches the dense inputs again, so release
     * them before it starts. Note: GNU time's Peak RSS is a high-water mark,
     * so it still includes these buffers from the load phase. csr_bytes
     * reports the size of the CSR data itself.
     */
    free(a_dense);
    free(b_dense);
    a_dense = NULL;
    b_dense = NULL;

    /* ============ TIMED REGION 2: MULTIPLICATION START ============ */
    multiply_start = now_seconds();

    multiply_status = csr_multiply(&a_csr, &b_csr, &c_csr);

    multiply_end = now_seconds();
    /* ============ TIMED REGION 2: MULTIPLICATION END ============== */

    if (multiply_status != 0) {
        fprintf(stderr, "Error: out of memory during CSR multiplication\n");
        goto cleanup;
    }

    /* 4. Report and (optionally) save - NOT timed. */
    row_buffer = allocate_array((size_t)n, sizeof(double));
    if (row_buffer == NULL) {
        fprintf(stderr, "Error: out of memory for the output buffer\n");
        goto cleanup;
    }

    csr_bytes = csr_storage_bytes(&a_csr) + csr_storage_bytes(&b_csr)
              + csr_storage_bytes(&c_csr);

    /* %.9f keeps the nanosecond resolution of the clock: at 99.9 % sparsity
     * the multiplication takes only tens of microseconds, so %.6f (1 us
     * steps) would limit the precision of those runs to about 1-2 %. */
    printf("algo=csr n=%lld nnz_A=%lld nnz_B=%lld nnz_C=%lld csr_bytes=%lld "
           "conversion_time_s=%.9f time_s=%.9f checksum=%.10e\n",
           (long long)n, (long long)a_csr.nnz, (long long)b_csr.nnz,
           (long long)c_csr.nnz, (long long)csr_bytes,
           conversion_end - conversion_start, multiply_end - multiply_start,
           csr_checksum(&c_csr, row_buffer));

    if (output_path != NULL && save_csr_as_dense(output_path, &c_csr, row_buffer) != 0) {
        goto cleanup;
    }
    status = EXIT_SUCCESS;

cleanup:
    free(a_dense);
    free(b_dense);
    free(row_buffer);
    csr_free(&a_csr);
    csr_free(&b_csr);
    csr_free(&c_csr);
    return status;
}
