/*
 * parallel_mult.c - Parallel and vectorised dense matrix multiplication
 *                   (Assignment 3)
 *
 * Three kernels compute C = A x B for square, row-major matrices of 64-bit
 * floats. All three use the cache-blocked (tiled) i-k-j loops of
 * Assignment 2; they only differ in HOW that same work is executed:
 *
 *   seq_tiled      : the Assignment 2 tiled kernel on one thread (baseline)
 *   omp_tiled      : the row tiles of C are shared out among p threads (OpenMP)
 *   omp_simd_tiled : omp_tiled + explicit SIMD on the innermost loop
 *                    (the optional vectorisation extension)
 *
 * What the three kernels answer together (same program, same flags):
 *   seq_tiled  vs omp_tiled at p = 1      -> cost of the OpenMP machinery
 *   omp_tiled at p vs seq_tiled           -> parallel speedup S(p) = T_seq / T_p
 *   omp_simd_tiled vs omp_tiled, same p   -> what SIMD adds on top of threads
 *
 * What does NOT change between the kernels:
 *   Every kernel does the same n^3 multiply-adds, and every C[i][j] receives
 *   its n terms in the same order (k = 0, 1, ..., n-1):
 *     - threads split the work by ROWS of C, and a row is never split, so
 *       each element is computed by exactly one thread, in the usual order;
 *     - SIMD lanes hold neighbouring COLUMNS (C[i][j], C[i][j+1], ...), so
 *       each lane still adds up its own terms in the usual order.
 *   As long as the compiler does not fuse a*b + c into one FMA instruction
 *   (-std=c11 implies -ffp-contract=off), the three results are therefore
 *   bit-for-bit identical. run_benchmark.py relies on this: all kernels
 *   must print the same checksum for the same input file.
 *
 * Scalar vs vector (why two kernels carry '#pragma GCC novector'):
 *   The program is built with -O3 -march=native -ftree-vectorize, and the
 *   i-k-j inner loop (two stride-1 streams, independent iterations) is
 *   exactly what GCC's auto-vectoriser looks for. Left alone, GCC would
 *   vectorise ALL three kernels, and "omp_tiled vs omp_simd_tiled" would
 *   compare vector code with vector code. So the two reference kernels ask
 *   for a scalar inner loop ('#pragma GCC novector', GCC >= 14) and only
 *   omp_simd_tiled asks for SIMD ('#pragma omp simd').
 *   Proof: the build writes GCC's vectorisation report (-fopt-info-vec-optimized)
 *   to results/vectorization_report.txt. Its "loop vectorized" lines must
 *   point inside multiply_omp_simd_tiled() and nowhere in the other two
 *   kernels; verify_correctness.py checks exactly that.
 *
 * Responsibilities (SOLID-lite, same structure as Assignment 2):
 *   alloc_matrix()             -> cache-line aligned n*n buffer
 *   load_matrices()            -> read A and B from the shared binary file
 *   multiply_seq_tiled()       -> kernel (a)
 *   multiply_omp_tiled()       -> kernel (b)
 *   multiply_omp_simd_tiled()  -> kernel (c)
 *   prefill_with_nan()         -> touch C and start the threads, NOT timed
 *   now_seconds()              -> CLOCK_MONOTONIC timer
 *   checksum()                 -> sum of C, cross-check between kernels
 *   save_matrix()              -> optional output for verify_correctness.py
 *   parse_arguments()          -> command-line interface
 *   main()                     -> orchestration + the timer boundary
 *
 * Matrix file format (unchanged, see common/data_generator.py):
 *   Input : int64 n | n*n float64 (A, row-major) | n*n float64 (B, row-major)
 *   Output: int64 n | n*n float64 (C, row-major)
 *   All values little-endian.
 *
 * Build (all flags fixed in benchmark_config.py; GCC >= 14 required):
 *   gcc -O3 -march=native -ftree-vectorize -fopenmp -std=c11 -Wall -Wextra \
 *       -fopt-info-vec-optimized=results/vectorization_report.txt \
 *       -o build/c/parallel_mult src/c/parallel_mult.c
 * Run:
 *   ./parallel_mult <matrices.bin> --algo seq_tiled|omp_tiled|omp_simd_tiled
 *                   [--threads P] [--block-size B] [--output result.bin]
 *   --threads is required for the omp_* kernels (seq_tiled always uses 1).
 *
 * Output (one line on stdout, space-separated key=value pairs):
 *   algo=omp_tiled n=1000 threads=4 block_size=64 time_s=0.123456789 checksum=1.2345678901e+02
 *   threads is the team size the OpenMP runtime REALLY started (checked
 *   before timing), not just the number that was asked for.
 */

/* Needed so <time.h> exposes clock_gettime() under strict -std=c11. */
#define _POSIX_C_SOURCE 199309L

#include <errno.h>
#include <math.h>      /* NAN (the macro only; no -lm needed) */
#include <omp.h>       /* omp_set_num_threads(), omp_get_num_threads(), ... */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * The scalar reference kernels depend on '#pragma GCC novector' (GCC 14+).
 * An older GCC (or another compiler) would silently ignore that pragma and
 * vectorise them, which would invalidate the SIMD comparison. Failing at
 * compile time is safer than measuring the wrong thing.
 */
#if !defined(__GNUC__) || defined(__clang__) || (__GNUC__ < 14)
#error "Build with GCC >= 14: the scalar kernels rely on '#pragma GCC novector' (Ubuntu 24.04: sudo apt install gcc-14, then set C_COMPILER = \"gcc-14\" in benchmark_config.py)"
#endif

/*
 * Default tile edge. One 64x64 tile of doubles is 32 KiB (about one L1 data
 * cache); the three tiles in use (A, B, C) take 96 KiB and fit in L2.
 * Each thread works on its own tiles, so each core needs its own 96 KiB:
 * private L1/L2 caches make that free, which is one reason tiling and
 * threading combine well.
 */
#define DEFAULT_BLOCK_SIZE 64

/* Size of one cache line on x86-64 (and most ARM64) CPUs, in bytes. */
#define CACHE_LINE_BYTES 64

/* Sanity limit for --threads (far above any core count we will meet). */
#define MAX_THREADS 1024

/*
 * Keep every kernel a separate function (never inlined into main()).
 * Same reason as in Assignment 2: inlined into main(), the six nested loops
 * ran out of registers and became much slower. With noinline, every kernel
 * is compiled on its own, under the same conditions.
 */
#define NOINLINE __attribute__((noinline))

/* ------------------------------------------------------------------ */
/* Memory                                                              */
/* ------------------------------------------------------------------ */

/*
 * Returns a buffer for `count` doubles whose first byte starts a cache line,
 * or NULL (out of memory, or the size does not fit in size_t).
 *
 * Why align (malloc only guarantees 16 bytes):
 *   Two threads that write to the SAME 64-byte cache line keep stealing it
 *   from each other, even if they write different elements ("false
 *   sharing"). Our threads write disjoint blocks of rows of C. When C starts
 *   on a line boundary, those blocks also start on line boundaries (see
 *   multiply_omp_tiled), so no line is ever written by two threads.
 *   A and B are aligned as well, for consistency (they are only read).
 *
 * C11 aligned_alloc() wants the size to be a multiple of the alignment,
 * so it is rounded up (at most 63 extra bytes). Free with free().
 */
static double *alloc_matrix(size_t count)
{
    if (count > (SIZE_MAX - CACHE_LINE_BYTES) / sizeof(double)) {
        return NULL;    /* count * 8 (+ rounding) would overflow size_t */
    }
    size_t bytes = count * sizeof(double);
    size_t rounded = (bytes + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES * CACHE_LINE_BYTES;
    return aligned_alloc(CACHE_LINE_BYTES, rounded);
}

/* ------------------------------------------------------------------ */
/* Data loading (as in Assignment 2, with aligned buffers)             */
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

    /*
     * Never trust the header: check n BEFORE computing n*n.
     * 0 < n <= INT32_MAX keeps n*n below 2^62, so it fits in size_t;
     * alloc_matrix() then checks that n*n*8 bytes fit as well.
     */
    int64_t n = 0;
    if (fread(&n, sizeof(int64_t), 1, file) != 1 || n <= 0 || n > INT32_MAX) {
        fprintf(stderr, "Error: invalid header in '%s' (need 0 < n <= %d)\n",
                path, INT32_MAX);
        fclose(file);
        return -1;
    }

    size_t count = (size_t)n * (size_t)n;   /* number of elements per matrix */
    double *a = alloc_matrix(count);
    double *b = alloc_matrix(count);
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

static int64_t min_i64(int64_t x, int64_t y)
{
    return (x < y) ? x : y;
}

/* ------------------------------------------------------------------ */
/* Kernel (a): sequential tiled baseline (Assignment 2)                */
/* ------------------------------------------------------------------ */

/*
 * The tiled kernel of Assignment 2, unchanged except for ONE line: the
 * '#pragma GCC novector' in front of the innermost loop.
 *
 * Reminder of how it works (details in assignment_2/src/c/dense_mult.c):
 *   ii, jj, kk : pick tile C(ii,jj) and add A(ii,kk) * B(kk,jj) to it;
 *   i, k, j    : that tile product in i-k-j order, so the innermost loop
 *                walks a row segment of B and of C with stride 1.
 *   The tiles of A, B and C in use (3 * bs^2 doubles) stay in cache, so a
 *   B tile is reused bs times instead of being re-read from memory.
 *
 * '#pragma GCC novector' (GCC 14+):
 *   "Do not vectorise the next loop." This kernel is the scalar reference:
 *   one multiply and one add per instruction, on one double at a time.
 *   Note that this is the only thing it changes; -O3 still optimises the
 *   loop in every other way (registers, scheduling, unrolling, ...).
 *
 * Remark: compiled here with -O3 -march=native instead of the -O2 of
 * Assignment 2, so its times are comparable with the other kernels of THIS
 * program, not with the Assignment 2 measurements.
 */
static NOINLINE void multiply_seq_tiled(const double *a, const double *b, double *c,
                                        int64_t n, int64_t block_size)
{
    memset(c, 0, (size_t)n * (size_t)n * sizeof(double));   /* we accumulate into C */

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

                        #pragma GCC novector
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
/* Kernel (b): OpenMP parallel tiled                                   */
/* ------------------------------------------------------------------ */

/*
 * Kernel (a) with the OUTERMOST tile loop (ii) shared out among threads.
 * The body of the loop is the same as in (a); the only additions are the
 * '#pragma omp parallel for' line and the per-block clearing of C.
 *
 * Which loop to parallelise, and why ii:
 *   One iteration of ii computes a whole block-row of C: rows ii .. i_end-1,
 *   all columns, all kk. Different ii values write to DIFFERENT rows of C.
 *     -> No two threads ever write the same element: no data race, so no
 *        locks, atomics or reductions are needed (they would all cost time).
 *     -> Each C element is still built by one thread with k = 0..n-1 in
 *        order, so the result is bit-identical to kernel (a).
 *   Why not kk: two threads would add into the SAME C tile at the same
 *   time, a race (lost updates) unless it is protected by atomics.
 *   Why not jj or i: this works too, but splits rows between threads, so
 *   neighbouring tiles of one row would be written by different threads
 *   (more thread boundaries, see false sharing below) and the parallel
 *   loop would start and stop many more times.
 *
 * The pragma, clause by clause:
 *   parallel for       Start a team of threads (the size set in main() with
 *                      omp_set_num_threads) and divide the ii iterations
 *                      among them. At the end of the loop there is an
 *                      implicit barrier: the call returns only when every
 *                      thread has finished, so the timer in main() measures
 *                      the complete product.
 *   schedule(static)   Thread t gets one contiguous chunk of ii tiles,
 *                      decided once, before the loop starts. Every full tile
 *                      costs the same, so the even split is already
 *                      balanced, and static costs nothing at run time.
 *                      (dynamic would hand out tiles one by one through a
 *                      shared counter: overhead for no gain here, and a
 *                      different tile-to-thread mapping in every run.)
 *   default(none)      Forces us to state the sharing of every outside
 *                      variable. Forgetting one becomes a compile error
 *                      instead of a silently shared (racy) variable.
 *   shared(...)        The pointers and sizes are only READ inside the loop.
 *                      A and B are only read, too: many threads may read
 *                      the same data at the same time, which is safe and
 *                      lets each core keep its own copy in its cache.
 *                      Everything declared INSIDE the loop (i_end, c_row,
 *                      a_ik, ...) and the loop counter ii are automatically
 *                      private: each thread has its own copy.
 *
 * Clearing C inside the parallel loop:
 *   Each thread zeroes its own block-row of C just before accumulating into
 *   it. A single memset of all of C before the loop (as in (a)) would be a
 *   serial part of the timed region, which limits the speedup (Amdahl's law).
 *
 * False sharing (why the threads do not slow each other down):
 *   Cores exchange data in whole 64-byte cache lines. If two threads write
 *   different elements that sit in the SAME line, the line bounces between
 *   their caches on every write, as if they shared the data.
 *   Here:
 *     - Nothing shared is written except C, and there are no shared
 *       counters or accumulators (the classic false-sharing trap: an array
 *       like partial_sum[thread_id] updated in the inner loop).
 *     - Thread t writes only its contiguous block of rows; neighbouring
 *       threads meet only at a block boundary, i.e. at the start of a row
 *       whose index r is a multiple of block_size. That row starts at byte
 *       r * n * 8 of C. C is 64-byte aligned (alloc_matrix), so whenever
 *       r * n is a multiple of 8, the boundary is also a cache-line
 *       boundary and no line is written by two threads. This holds for
 *       every n if block_size is a multiple of 8 (our block size is 64),
 *       and for every block size if n is a multiple of 8.
 *       Otherwise, at most one line per boundary (p - 1 lines in total) is
 *       shared, which is negligible next to the n*n*8/64 lines of C.
 *
 * Load balance:
 *   There are ceil(n / block_size) row tiles; the last one may be smaller.
 *   With fewer tiles than threads, some threads get no work (main() warns).
 */
static NOINLINE void multiply_omp_tiled(const double *a, const double *b, double *c,
                                        int64_t n, int64_t block_size)
{
    #pragma omp parallel for schedule(static) default(none) shared(a, b, c, n, block_size)
    for (int64_t ii = 0; ii < n; ii += block_size) {
        const int64_t i_end = min_i64(ii + block_size, n);

        /* Zero rows ii .. i_end-1 of C: this thread's block-row. */
        memset(&c[ii * n], 0, (size_t)(i_end - ii) * (size_t)n * sizeof(double));

        for (int64_t jj = 0; jj < n; jj += block_size) {
            const int64_t j_end = min_i64(jj + block_size, n);

            for (int64_t kk = 0; kk < n; kk += block_size) {
                const int64_t k_end = min_i64(kk + block_size, n);

                for (int64_t i = ii; i < i_end; i++) {
                    double *c_row = &c[i * n];

                    for (int64_t k = kk; k < k_end; k++) {
                        const double a_ik = a[i * n + k];
                        const double *b_row = &b[k * n];

                        /* Scalar, as in (a): this kernel measures threads only. */
                        #pragma GCC novector
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
/* Kernel (c): OpenMP parallel tiled + explicit SIMD (optional part)   */
/* ------------------------------------------------------------------ */

/*
 * Kernel (b) with ONE line changed: the innermost loop is marked
 * '#pragma omp simd' instead of '#pragma GCC novector'. Threads,
 * scheduling, clearing and false-sharing reasoning are exactly as in (b).
 *
 * What SIMD does here:
 *   One AVX2 instruction works on 4 doubles (256-bit registers), one
 *   AVX-512 instruction on 8. The loop
 *       c_row[j] += a_ik * b_row[j]
 *   becomes: broadcast a_ik into all lanes once, then per step load 4 (or 8)
 *   consecutive b_row[j], multiply, add to 4 (or 8) consecutive c_row[j],
 *   store. If the loop length is not a multiple of the vector width, a short
 *   "epilogue" handles the last few j. GCC's report tells which width it
 *   used ("loop vectorized using 32 byte vectors" = 4 doubles).
 *
 * '#pragma omp simd' (OpenMP 4.0, enabled by -fopenmp):
 *   A promise from the programmer: "the iterations of this loop are
 *   independent, execute them in SIMD lanes". Here the promise is true:
 *   iteration j reads b_row[j] and updates only c_row[j], and B and C are
 *   separate allocations, so no iteration depends on another.
 *   Without the pragma, GCC cannot prove that c_row and b_row never overlap
 *   (both are plain double pointers). Its auto-vectoriser would then add a
 *   run-time overlap check and keep a scalar copy of the loop as a fallback
 *   ("loop versioned for vectorization because of possible aliasing" in the
 *   report). With the promise, it emits the vector loop directly.
 *   Other options considered:
 *     '#pragma GCC ivdep' only says "ignore assumed dependencies"; GCC's
 *         cost model may still decide not to vectorise, and it is GCC-only.
 *     'restrict' pointers say "no aliasing" for the whole function, which
 *         is a broader promise than we need for one loop.
 *   Clauses we deliberately do NOT use:
 *     aligned(...)   rows of B and C start at i*n*8 bytes and the j range
 *                    starts at jj, so 32/64-byte alignment is not
 *                    guaranteed for every n; a false promise would be
 *                    undefined behaviour. Unaligned vector loads are fast on
 *                    current x86-64 CPUs anyway.
 *     reduction(...) there is no sum over j; each lane owns its own C[i][j].
 *
 * Why the result stays bit-identical:
 *   Lanes correspond to different j (different output elements), so the
 *   order in which each C[i][j] receives its terms does not change.
 *   (Vectorising a SUM is different: several partial sums would change the
 *   order of the additions, and GCC only does that with -ffast-math. The
 *   report also lists the checksum loop as vectorised, but only as an
 *   "in-order" reduction: it handles one vector's worth of elements per
 *   step, yet still adds them one by one in index order.)
 *   With -ffp-contract=off (implied by -std=c11), multiply and add stay two
 *   separately rounded operations, exactly as in the scalar kernels.
 */
static NOINLINE void multiply_omp_simd_tiled(const double *a, const double *b, double *c,
                                             int64_t n, int64_t block_size)
{
    #pragma omp parallel for schedule(static) default(none) shared(a, b, c, n, block_size)
    for (int64_t ii = 0; ii < n; ii += block_size) {
        const int64_t i_end = min_i64(ii + block_size, n);

        /* Zero rows ii .. i_end-1 of C: this thread's block-row. */
        memset(&c[ii * n], 0, (size_t)(i_end - ii) * (size_t)n * sizeof(double));

        for (int64_t jj = 0; jj < n; jj += block_size) {
            const int64_t j_end = min_i64(jj + block_size, n);

            for (int64_t kk = 0; kk < n; kk += block_size) {
                const int64_t k_end = min_i64(kk + block_size, n);

                for (int64_t i = ii; i < i_end; i++) {
                    double *c_row = &c[i * n];

                    for (int64_t k = kk; k < k_end; k++) {
                        const double a_ik = a[i * n + k];
                        const double *b_row = &b[k * n];

                        /* The ONLY difference from (b): vectorise this loop. */
                        #pragma omp simd
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
/* Preparing C outside the timed region                                */
/* ------------------------------------------------------------------ */

/*
 * Fills C with NaN using the same number of threads as the kernel, and
 * returns how many threads the OpenMP runtime actually started.
 *
 * This runs BEFORE the timer and does three jobs:
 *  1. Page faults stay outside the timer (as in Assignment 2). A large
 *     allocation only reserves address space; the OS maps each 4 KiB page
 *     on its first write. Writing every element here means no kernel pays
 *     that cost inside the timed region.
 *  2. The thread team is created outside the timer. OpenMP starts its
 *     worker threads at the FIRST parallel region of the program and then
 *     reuses them. Since each benchmark run is a fresh process, without this
 *     step the kernel would also time the creation of p threads. Here the
 *     team is already running when the kernel starts, so p = 1 vs seq_tiled
 *     measures the per-call cost of OpenMP, not a one-off start-up cost.
 *  3. Safety net: any element a kernel forgets to write (or to clear) stays
 *     NaN, so the checksum becomes nan and the error cannot go unnoticed.
 *
 * The threads fill C in their own static chunks of elements. On a
 * single-socket machine it does not matter which thread touches a page
 * first (all cores share the same memory); on multi-socket NUMA servers,
 * this "first touch" would also decide in whose memory each page lives.
 */
static int prefill_with_nan(double *c, int64_t n)
{
    const size_t count = (size_t)n * (size_t)n;
    int team_size = 0;

    #pragma omp parallel default(none) shared(c, count, team_size)
    {
        /* One thread records the team size; 'single' ends with a barrier. */
        #pragma omp single
        team_size = omp_get_num_threads();

        /* Each thread writes its own contiguous chunk of C. */
        #pragma omp for schedule(static)
        for (size_t idx = 0; idx < count; idx++) {
            c[idx] = NAN;
        }
    }
    return team_size;
}

/* ------------------------------------------------------------------ */
/* Timer (unchanged from Assignments 1 and 2)                          */
/* ------------------------------------------------------------------ */

/*
 * Current time in seconds from CLOCK_MONOTONIC.
 * A monotonic clock never jumps backwards (unlike the wall clock, which NTP
 * can adjust), so the difference of two readings is a valid duration.
 * It is WALL time, which is what a parallel speedup needs: CPU-time clocks
 * such as clock() add up the time of all threads and would hide the gain.
 */
static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ------------------------------------------------------------------ */
/* Result check and output (unchanged from Assignment 2)               */
/* ------------------------------------------------------------------ */

/*
 * Sum of all elements of C. It lets us compare kernels quickly, and because
 * the result is used, the compiler cannot optimise the kernel away.
 * Computed sequentially, in index order, so it is the same for every kernel
 * and thread count whenever C is the same.
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
    ALGO_SEQ_TILED,
    ALGO_OMP_TILED,
    ALGO_OMP_SIMD_TILED
} Algorithm;

typedef struct {
    const char *input_path;
    const char *output_path;    /* NULL = do not save C */
    Algorithm algo;
    int threads;                /* team size; always 1 for seq_tiled */
    int64_t block_size;         /* tile edge, used by all three kernels */
} Options;

static const char *algorithm_name(Algorithm algo)
{
    switch (algo) {
    case ALGO_SEQ_TILED:      return "seq_tiled";
    case ALGO_OMP_TILED:      return "omp_tiled";
    case ALGO_OMP_SIMD_TILED: return "omp_simd_tiled";
    }
    return "unknown";
}

static void print_usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s <matrices_file.bin> --algo seq_tiled|omp_tiled|omp_simd_tiled\n"
            "          [--threads P] [--block-size B] [--output <result.bin>]\n"
            "  --threads     number of OpenMP threads, 1..%d (required for omp_*;\n"
            "                seq_tiled always runs on 1 thread)\n"
            "  --block-size  tile edge (default: %d)\n",
            program, MAX_THREADS, DEFAULT_BLOCK_SIZE);
}

/* Returns 0 and sets *algo_out if `text` names a known algorithm, else -1. */
static int parse_algorithm(const char *text, Algorithm *algo_out)
{
    if (strcmp(text, "seq_tiled") == 0)      { *algo_out = ALGO_SEQ_TILED;      return 0; }
    if (strcmp(text, "omp_tiled") == 0)      { *algo_out = ALGO_OMP_TILED;      return 0; }
    if (strcmp(text, "omp_simd_tiled") == 0) { *algo_out = ALGO_OMP_SIMD_TILED; return 0; }
    return -1;
}

/*
 * Returns 0 and sets *value_out if `text` is an integer in [1, max_value],
 * else -1. The upper limit keeps later arithmetic far from overflow.
 */
static int parse_bounded_integer(const char *text, int64_t max_value, int64_t *value_out)
{
    char *end = NULL;
    errno = 0;
    long long value = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0 || value > max_value) {
        return -1;
    }
    *value_out = (int64_t)value;
    return 0;
}

/*
 * Fills `opts` from argv. Returns 0 on success, -1 on a usage error
 * (a specific message is printed here; main() then prints the usage).
 * --algo is required, and --threads is required for the omp_* kernels:
 * a benchmark run should never silently fall back to a default and be
 * recorded under the wrong name or thread count.
 */
static int parse_arguments(int argc, char *argv[], Options *opts)
{
    int algo_given = 0;
    int threads_given = 0;
    int64_t threads = 1;

    opts->input_path = NULL;
    opts->output_path = NULL;
    opts->algo = ALGO_SEQ_TILED;
    opts->threads = 1;
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
        } else if (strcmp(arg, "--threads") == 0 && has_value) {
            if (parse_bounded_integer(argv[++i], MAX_THREADS, &threads) != 0) {
                fprintf(stderr, "Error: --threads must be an integer in 1..%d, got '%s'\n",
                        MAX_THREADS, argv[i]);
                return -1;
            }
            threads_given = 1;
        } else if (strcmp(arg, "--block-size") == 0 && has_value) {
            /* <= INT32_MAX: the OpenMP loop computes about (n + block_size) / block_size
               iterations, which must not overflow int64_t. */
            if (parse_bounded_integer(argv[++i], INT32_MAX, &opts->block_size) != 0) {
                fprintf(stderr, "Error: --block-size must be an integer in 1..%d, got '%s'\n",
                        INT32_MAX, argv[i]);
                return -1;
            }
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

    if (opts->algo == ALGO_SEQ_TILED) {
        /* The baseline is sequential by definition. */
        if (threads_given && threads != 1) {
            fprintf(stderr, "Warning: seq_tiled always uses 1 thread; --threads %lld ignored\n",
                    (long long)threads);
        }
        opts->threads = 1;
    } else {
        if (!threads_given) {
            fprintf(stderr, "Error: --threads is required for --algo %s\n",
                    algorithm_name(opts->algo));
            return -1;
        }
        opts->threads = (int)threads;
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

    /* 2. Allocate the output (cache-line aligned) - NOT timed. */
    double *c = alloc_matrix((size_t)n * (size_t)n);
    if (c == NULL) {
        fprintf(stderr, "Error: out of memory for result matrix\n");
        free(a);
        free(b);
        return EXIT_FAILURE;
    }

    /*
     * 3. Fix the number of threads - NOT timed.
     *    omp_set_dynamic(0): the runtime may NOT choose fewer threads than
     *        requested (with "dynamic adjustment" on, it is allowed to, e.g.
     *        when the machine looks busy).
     *    omp_set_num_threads(p): every following parallel region uses p
     *        threads. Being a call in the program, it overrides the
     *        OMP_NUM_THREADS environment variable, so --threads is the single
     *        source of truth for the thread count.
     */
    omp_set_dynamic(0);
    omp_set_num_threads(opts.threads);

    /* 4. Touch C and start the thread team - NOT timed (see the function). */
    int team_size = prefill_with_nan(c, n);

    /* The runtime can still give fewer threads (e.g. OMP_THREAD_LIMIT).
       Better to stop than to record the run under the wrong thread count. */
    if (team_size != opts.threads) {
        fprintf(stderr, "Error: requested %d threads but OpenMP started %d\n",
                opts.threads, team_size);
        free(a);
        free(b);
        free(c);
        return EXIT_FAILURE;
    }

    /* Only whole row tiles are shared out; warn if some threads must idle. */
    int64_t row_tiles = (n + opts.block_size - 1) / opts.block_size;
    if (row_tiles < opts.threads) {
        fprintf(stderr, "Warning: only %lld row tile(s) for %d threads; %lld thread(s) get no work\n",
                (long long)row_tiles, opts.threads, (long long)(opts.threads - row_tiles));
    }

    /* ============ TIMED REGION START ============ */
    double start = now_seconds();

    switch (opts.algo) {    /* one branch, taken once: negligible cost */
    case ALGO_SEQ_TILED:      multiply_seq_tiled(a, b, c, n, opts.block_size);      break;
    case ALGO_OMP_TILED:      multiply_omp_tiled(a, b, c, n, opts.block_size);      break;
    case ALGO_OMP_SIMD_TILED: multiply_omp_simd_tiled(a, b, c, n, opts.block_size); break;
    }

    double end = now_seconds();
    /* ============ TIMED REGION END ============== */

    /* 5. Report and (optionally) save - NOT timed.
     *    %.9f keeps the nanosecond resolution of the clock. */
    printf("algo=%s n=%lld threads=%d block_size=%lld time_s=%.9f checksum=%.10e\n",
           algorithm_name(opts.algo), (long long)n, team_size,
           (long long)opts.block_size, end - start, checksum(c, n));

    int status = EXIT_SUCCESS;
    if (opts.output_path != NULL && save_matrix(opts.output_path, c, n) != 0) {
        status = EXIT_FAILURE;
    }

    free(a);
    free(b);
    free(c);
    return status;
}
