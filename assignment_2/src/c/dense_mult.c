/*
 * dense_mult.c - Optimised dense matrix multiplication (Assignment 2, Part A)
 *
 * Three sequential kernels that compute C = A x B for square, row-major
 * matrices of 64-bit floats:
 *
 *   baseline  : i-j-k triple loop, identical to Assignment 1 (the reference)
 *   reordered : i-k-j loop order  -> stride-1 access to B and C
 *   tiled     : cache blocking (tiling) on top of the i-k-j order
 *
 * What changes and what does not:
 *   All three kernels do exactly the same n^3 multiply-adds, and every
 *   C[i][j] even receives its n terms in the same order (k = 0, 1, ..., n-1).
 *   The ONLY difference is the order in which memory is visited. So any
 *   speed difference comes from the memory hierarchy (cache lines,
 *   prefetching, cache reuse), not from doing less arithmetic.
 *   A useful side effect: with standard flags (no -ffast-math) the three
 *   results are bit-for-bit identical, so their checksums must match.
 *
 * Responsibilities (SOLID-lite, same structure as Assignment 1):
 *   load_matrices()      -> read A and B from the shared binary file
 *   multiply_baseline()  -> kernel (a)
 *   multiply_reordered() -> kernel (b)
 *   multiply_tiled()     -> kernel (c)
 *   now_seconds()        -> CLOCK_MONOTONIC timer
 *   checksum()           -> sum of C, cross-check between kernels/languages
 *   save_matrix()        -> optional output for verify_correctness.py
 *   parse_arguments()    -> command-line interface
 *   main()               -> orchestration + the timer boundary
 *
 * Matrix file format (shared by all programs, see common/data_generator.py):
 *   Input : int64 n | n*n float64 (A, row-major) | n*n float64 (B, row-major)
 *   Output: int64 n | n*n float64 (C, row-major)
 *   All values little-endian. Sparse inputs use the same format (zeros are
 *   stored as 0.0), so this program reads them unchanged.
 *
 * Memory layout (same as Assignment 1):
 *   Each matrix is ONE contiguous block of n*n doubles in row-major order:
 *   element (i, j) is at index i*n + j. Moving along a row (j+1) is the
 *   next address; moving down a column (i+1) jumps n*8 bytes.
 *
 * Build (the same flags for all three kernels; fixed in the benchmark config):
 *   gcc -O2 -std=c11 -Wall -Wextra -o dense_mult dense_mult.c
 * Run:
 *   ./dense_mult <matrices.bin> --algo baseline|reordered|tiled
 *                [--block-size B] [--output result.bin]
 *
 * Output (one line on stdout, space-separated key=value pairs):
 *   algo=reordered n=500 time_s=0.081234 checksum=1.2345678901e+02
 *   algo=tiled n=500 block_size=64 time_s=0.071234 checksum=1.2345678901e+02
 */

/* Needed so <time.h> exposes clock_gettime() under strict -std=c11. */
#define _POSIX_C_SOURCE 199309L

#include <errno.h>
#include <math.h>      /* NAN (the macro only; no -lm needed) */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Default tile edge for --algo tiled. With 64, one tile is 64*64*8 = 32 KiB,
 * about the size of a typical L1 data cache, and the three tiles in use
 * (96 KiB) fit comfortably in L2. The benchmark sweeps other values.
 */
#define DEFAULT_BLOCK_SIZE 64

/*
 * Keep every kernel a separate function (never inlined into main()).
 *
 * Why: each kernel is called exactly once, so GCC likes to paste it into
 * main(). For the six nested loops of the tiled kernel this left too few
 * free registers, and GCC spilled the B-row pointer to the stack, loading
 * and storing it in EVERY inner iteration. In our tests that made the tiled
 * kernel about 2.5x slower than the very same code as a standalone function.
 * With noinline, every kernel is compiled on its own, under the same
 * conditions, so we compare algorithms and not inlining side effects.
 * The cost is one function call per run: nothing next to n^3 operations.
 * (__attribute__ is a GCC/Clang extension; other compilers simply skip it.)
 */
#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

/* ------------------------------------------------------------------ */
/* Data loading (unchanged from Assignment 1)                          */
/* ------------------------------------------------------------------ */

/*
 * Reads n, A and B from `path`.
 * On success returns 0 and sets *n_out, *a_out, *b_out (caller must free).
 * On failure returns -1 and prints a message to stderr.
 *
 * Assumption: the machine is little-endian (true for x86-64 and ARM64),
 * so we can fread() the bytes directly into int64_t / double.
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
/* Kernel (a): baseline, i-j-k                                         */
/* ------------------------------------------------------------------ */

/*
 * The Assignment 1 kernel, copied without any change so that it is a true
 * baseline: C[i][j] = sum_k A[i][k] * B[k][j], one output element at a time.
 * (Only NOINLINE was added, as for every kernel; see its definition.)
 *
 * Memory access in the innermost loop (k changes):
 *   A[i][k] -> a[i*n + k] : stride 1, consecutive addresses        good
 *   B[k][j] -> b[k*n + j] : stride n, a jump of n*8 bytes per step  bad
 *
 * Why the B access is slow:
 *   Memory moves between RAM and cache in 64-byte lines (8 doubles).
 *   Walking down a COLUMN of B uses 8 bytes of each line and then jumps to
 *   another line, so 7/8 of the data loaded is wasted. For large n, those
 *   lines are evicted before the next j could use their neighbours.
 *   For n >= 512 every step even lands on a different 4 KiB page, which
 *   adds TLB misses, and hardware prefetchers stop at page boundaries.
 *
 * (The running sum stays in a register; this is what A1 did in all languages.)
 */
static NOINLINE void multiply_baseline(const double *a, const double *b, double *c, int64_t n)
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
/* Kernel (b): loop reordering, i-k-j                                  */
/* ------------------------------------------------------------------ */

/*
 * Same arithmetic, different loop nesting: the j and k loops are swapped.
 *
 * Why swapping is allowed:
 *   C[i][j] is still the sum over k of A[i][k] * B[k][j]. We only change
 *   WHEN each term is added, not WHICH terms are added. For a fixed (i, j)
 *   the terms even arrive in the same order k = 0..n-1 as in the baseline.
 *
 * How to read it:
 *   "row i of C = sum over k of A[i][k] * (row k of B)".
 *   For each (i, k), the single number A[i][k] scales a whole ROW of B,
 *   and that scaled row is added to row i of C.
 *
 * Memory access in the innermost loop (j changes):
 *   A[i][k] -> a_ik       : constant, loaded once into a register   best
 *   B[k][j] -> b_row[j]   : stride 1                                good
 *   C[i][j] -> c_row[j]   : stride 1                                good
 *
 * Why this is faster:
 *   - Every 64-byte cache line of B and C is fully used (8 doubles each).
 *   - Two sequential streams are the easiest pattern for the hardware
 *     prefetcher, so data is fetched before the loop asks for it.
 *   - Independent iterations over consecutive addresses can be turned into
 *     SIMD instructions by the compiler when vectorisation is enabled
 *     (e.g. -O3). The baseline cannot be vectorised: its k loop is one
 *     sequential sum, and B is not contiguous along k.
 *
 * The trade-off:
 *   C[i][j] is now loaded and stored in every inner iteration, instead of
 *   living in a register. That is cheap, because row i of C (n doubles,
 *   about 8 KB for n = 1000) stays in the L1 cache during the whole k loop.
 *
 * The weakness that motivates tiling:
 *   For EVERY row i, the two inner loops stream through ALL of B
 *   (n*n doubles = 8 MB for n = 1000). Once B no longer fits in the cache,
 *   it is re-read n times from a slower level (L3 or main memory).
 *
 * Because the j loop ACCUMULATES into C, C must start at zero. The kernel
 * clears C itself so that it is correct on its own; that is n*n stores,
 * negligible next to the n^3 multiply-adds (0.1 % for n = 1000).
 *
 * The row pointers c_row and b_row only hoist the i*n and k*n index
 * arithmetic out of the inner loop, to keep that loop easy to read.
 */
static NOINLINE void multiply_reordered(const double *a, const double *b, double *c, int64_t n)
{
    memset(c, 0, (size_t)n * (size_t)n * sizeof(double));

    for (int64_t i = 0; i < n; i++) {
        double *c_row = &c[i * n];                  /* row i of C */

        for (int64_t k = 0; k < n; k++) {
            const double a_ik = a[i * n + k];       /* fixed for the whole j loop */
            const double *b_row = &b[k * n];        /* row k of B */

            for (int64_t j = 0; j < n; j++) {
                c_row[j] += a_ik * b_row[j];        /* both stride 1 */
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Kernel (c): cache blocking (tiling)                                 */
/* ------------------------------------------------------------------ */

static int64_t min_i64(int64_t x, int64_t y)
{
    return (x < y) ? x : y;
}

/*
 * Idea:
 *   Cut every matrix into square tiles of block_size x block_size (bs x bs)
 *   and multiply tile by tile. Tile (I, J) of C is the sum over K of
 *   tile (I, K) of A times tile (K, J) of B: the same formula as for single
 *   elements, applied to small sub-matrices.
 *
 * Why it helps:
 *   Kernel (b) re-reads all of B for every row i. Here, a B tile is loaded
 *   into the cache once and then reused bs times (once for each row of the
 *   A and C tiles) before it is evicted. In the ideal case where the tiles
 *   fit in cache, the data moved from memory drops from about n^3 elements
 *   (B streamed n times) to about 2*n^3/bs (one A tile and one B tile per
 *   tile product). For bs = 64 that is roughly 32x less memory traffic.
 *
 * Loop structure: 6 loops = 3 tile loops + 3 element loops.
 *   ii, jj, kk : choose the tiles. The order ii-jj-kk means:
 *                "for each output tile C(ii,jj), add up all the products
 *                 A(ii,kk) * B(kk,jj)". The C tile stays in cache as an
 *                accumulator until it is complete, so each C value is
 *                written back to memory once; only A and B tiles stream.
 *   i, k, j    : the element loops inside one tile product, in the same
 *                i-k-j order as kernel (b), so the innermost loop is still
 *                stride 1 over a row segment of B and of C.
 *   The innermost statement is identical to kernel (b); tiling only
 *   shortens the ranges of the loops so that the data they touch is small.
 *
 * Choosing block_size:
 *   The three tiles in use (A, B and C) take 3 * bs^2 * 8 bytes and should
 *   fit in cache:
 *     bs =  32 ->  24 KiB in total (all three fit in L1)
 *     bs =  64 ->  96 KiB in total (one tile = 32 KiB ~ L1; all fit in L2)
 *     bs = 128 -> 384 KiB in total (L2)
 *   Too small: very short inner loops, more loop overhead, little reuse.
 *   Too large: tiles no longer fit and we are back to streaming B.
 *   The best value depends on the machine, so the benchmark sweeps it.
 *   If bs >= n there is only one tile and this kernel IS kernel (b).
 *   (When n is a power of two, the rows of a tile are exactly 2^p bytes
 *   apart and may compete for the same cache sets, which can cause dips.)
 *
 * Edge tiles:
 *   n does not need to be a multiple of bs. The last tile in each
 *   dimension is simply smaller, which the min() on each bound handles.
 *
 * Same results as (a) and (b):
 *   For a fixed (i, j), kk increases and k increases inside each tile, so
 *   C[i][j] still receives its terms in the order k = 0..n-1.
 */
static NOINLINE void multiply_tiled(const double *a, const double *b, double *c,
                                    int64_t n, int64_t block_size)
{
    memset(c, 0, (size_t)n * (size_t)n * sizeof(double));   /* we accumulate, as in (b) */

    for (int64_t ii = 0; ii < n; ii += block_size) {
        const int64_t i_end = min_i64(ii + block_size, n);

        for (int64_t jj = 0; jj < n; jj += block_size) {
            const int64_t j_end = min_i64(jj + block_size, n);

            for (int64_t kk = 0; kk < n; kk += block_size) {
                const int64_t k_end = min_i64(kk + block_size, n);

                /* C(ii,jj) += A(ii,kk) * B(kk,jj), in i-k-j order */
                for (int64_t i = ii; i < i_end; i++) {
                    double *c_row = &c[i * n];

                    for (int64_t k = kk; k < k_end; k++) {
                        const double a_ik = a[i * n + k];
                        const double *b_row = &b[k * n];

                        for (int64_t j = jj; j < j_end; j++) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Timer (unchanged from Assignment 1)                                 */
/* ------------------------------------------------------------------ */

/*
 * Current time in seconds from CLOCK_MONOTONIC.
 * A monotonic clock never jumps backwards (unlike the wall clock, which
 * NTP can adjust), so the difference of two readings is a valid duration.
 */
static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ------------------------------------------------------------------ */
/* Result check and output (unchanged from Assignment 1)               */
/* ------------------------------------------------------------------ */

/*
 * Sum of all elements of C. It lets us compare kernels quickly, and because
 * the result is used, the compiler cannot optimise the kernel away.
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
/* Command-line interface                                              */
/* ------------------------------------------------------------------ */

typedef enum {
    ALGO_BASELINE,
    ALGO_REORDERED,
    ALGO_TILED
} Algorithm;

typedef struct {
    const char *input_path;
    const char *output_path;    /* NULL = do not save C */
    Algorithm algo;
    int64_t block_size;         /* only used by ALGO_TILED */
} Options;

static const char *algorithm_name(Algorithm algo)
{
    switch (algo) {
    case ALGO_BASELINE:  return "baseline";
    case ALGO_REORDERED: return "reordered";
    case ALGO_TILED:     return "tiled";
    }
    return "unknown";
}

static void print_usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s <matrices_file.bin> --algo baseline|reordered|tiled\n"
            "          [--block-size B] [--output <result.bin>]\n"
            "  --block-size  tile edge for --algo tiled (default: %d)\n",
            program, DEFAULT_BLOCK_SIZE);
}

/* Returns 0 and sets *algo_out if `text` names a known algorithm, else -1. */
static int parse_algorithm(const char *text, Algorithm *algo_out)
{
    if (strcmp(text, "baseline") == 0)  { *algo_out = ALGO_BASELINE;  return 0; }
    if (strcmp(text, "reordered") == 0) { *algo_out = ALGO_REORDERED; return 0; }
    if (strcmp(text, "tiled") == 0)     { *algo_out = ALGO_TILED;     return 0; }
    return -1;
}

/* Returns 0 and sets *value_out if `text` is a positive integer, else -1. */
static int parse_positive_integer(const char *text, int64_t *value_out)
{
    char *end = NULL;
    errno = 0;
    long long value = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0) {
        return -1;
    }
    *value_out = (int64_t)value;
    return 0;
}

/*
 * Fills `opts` from argv. Returns 0 on success, -1 on a usage error
 * (a specific message is printed here; main() then prints the usage).
 * --algo is required on purpose: a benchmark run should never silently
 * fall back to a default kernel and be recorded under the wrong name.
 */
static int parse_arguments(int argc, char *argv[], Options *opts)
{
    int algo_given = 0;
    int block_size_given = 0;

    opts->input_path = NULL;
    opts->output_path = NULL;
    opts->algo = ALGO_BASELINE;
    opts->block_size = DEFAULT_BLOCK_SIZE;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        int has_value = (i + 1 < argc);

        if (strcmp(arg, "--algo") == 0 && has_value) {
            if (parse_algorithm(argv[++i], &opts->algo) != 0) {
                fprintf(stderr, "Error: unknown algorithm '%s'\n", argv[i]);
                return -1;
            }
            algo_given = 1;
        } else if (strcmp(arg, "--block-size") == 0 && has_value) {
            if (parse_positive_integer(argv[++i], &opts->block_size) != 0) {
                fprintf(stderr, "Error: --block-size must be a positive integer, got '%s'\n",
                        argv[i]);
                return -1;
            }
            block_size_given = 1;
        } else if (strcmp(arg, "--output") == 0 && has_value) {
            opts->output_path = argv[++i];
        } else if (arg[0] != '-' && opts->input_path == NULL) {
            opts->input_path = arg;
        } else {
            fprintf(stderr, "Error: unexpected or incomplete argument '%s'\n", arg);
            return -1;
        }
    }

    if (opts->input_path == NULL) {
        fprintf(stderr, "Error: missing input file\n");
        return -1;
    }
    if (!algo_given) {
        fprintf(stderr, "Error: --algo is required\n");
        return -1;
    }
    if (block_size_given && opts->algo != ALGO_TILED) {
        fprintf(stderr, "Warning: --block-size only applies to --algo tiled; ignored\n");
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Orchestration                                                       */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    Options opts;
    if (parse_arguments(argc, argv, &opts) != 0) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    /* 1. Load inputs - NOT timed (inputs are prepared outside the kernel). */
    int64_t n = 0;
    double *a = NULL;
    double *b = NULL;
    if (load_matrices(opts.input_path, &n, &a, &b) != 0) {
        return EXIT_FAILURE;
    }

    /* 2. Allocate the output - NOT timed. */
    size_t count = (size_t)n * (size_t)n;
    double *c = malloc(count * sizeof(double));
    if (c == NULL) {
        fprintf(stderr, "Error: out of memory for result matrix\n");
        free(a);
        free(b);
        return EXIT_FAILURE;
    }

    /*
     * Fill C with NaN before timing, for two reasons:
     *  - Page faults stay outside the timer. A large malloc/calloc only
     *    reserves address space; the OS maps each 4 KiB page on its first
     *    write. Writing every element here means no kernel pays that cost.
     *    (calloc alone is not enough: for big blocks it returns lazily
     *    zeroed pages, and the compiler may turn malloc + memset(0) into
     *    calloc. A NaN fill cannot be optimised away like that.)
     *  - Safety net: any element a kernel forgets to write (or, for (b) and
     *    (c), to clear) stays NaN, so the checksum becomes nan and the
     *    error cannot go unnoticed.
     */
    for (size_t idx = 0; idx < count; idx++) {
        c[idx] = NAN;
    }

    /* ============ TIMED REGION START ============ */
    double start = now_seconds();

    switch (opts.algo) {    /* one branch, taken once: negligible cost */
    case ALGO_BASELINE:  multiply_baseline(a, b, c, n);               break;
    case ALGO_REORDERED: multiply_reordered(a, b, c, n);              break;
    case ALGO_TILED:     multiply_tiled(a, b, c, n, opts.block_size); break;
    }

    double end = now_seconds();
    /* ============ TIMED REGION END ============== */

    /* 3. Report and (optionally) save - NOT timed.
     *    block_size is printed only where it has a meaning. */
    printf("algo=%s n=%lld ", algorithm_name(opts.algo), (long long)n);
    if (opts.algo == ALGO_TILED) {
        printf("block_size=%lld ", (long long)opts.block_size);
    }
    printf("time_s=%.6f checksum=%.10e\n", end - start, checksum(c, n));

    int status = EXIT_SUCCESS;
    if (opts.output_path != NULL && save_matrix(opts.output_path, c, n) != 0) {
        status = EXIT_FAILURE;
    }

    free(a);
    free(b);
    free(c);
    return status;
}
