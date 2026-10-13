/*
 * MatMul.java - Basic O(n^3) dense matrix multiplication (Assignment 1, Java version)
 *
 * Responsibilities are split into small methods (SOLID-lite):
 *   1. loadMatrices()  -> read A and B from the shared binary file
 *   2. multiply()      -> the O(n^3) triple-loop kernel (the ONLY timed part)
 *   3. warmUp()        -> untimed runs so the JIT compiler can optimize multiply()
 *   4. checksum()      -> a simple sum of C, used to cross-check languages
 *   5. saveMatrix()    -> optionally write C to disk for verify_correctness.py
 *   6. main()          -> orchestration + the timer boundary
 *
 * Matrix file format (shared by all languages, see common/data_generator.py):
 *   Input : int64 n | n*n double (A, row-major) | n*n double (B, row-major)
 *   Output: int64 n | n*n double (C, row-major)
 *   All values little-endian.
 *
 * Memory layout:
 *   Each matrix is ONE flat double[] of n*n elements in row-major order,
 *   element (i, j) at index i*n + j - exactly the same as C and Python.
 *   We deliberately avoid double[][] because Java stores each row of a
 *   2D array as a separate object, which would make the memory layout
 *   (and cache behaviour) different from the C version.
 *
 * Build and run (from the repository root):
 *   javac -d assignment_1/build/java assignment_1/src/java/MatMul.java
 *   java -cp assignment_1/build/java MatMul data/matrices_n100.bin [--warmup 3] [--output result_java.bin]
 *
 * Output (one line on stdout, parsed by run_benchmark.py):
 *   n=<n> warmup_runs=<w> time_s=<seconds> checksum=<sum of C>
 */

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

public class MatMul {

    /*
     * Default number of untimed warm-up runs. Each run executes the full
     * kernel, so the hot inner loop runs millions of times - far beyond the
     * JIT's compilation thresholds - and the timed run uses fully optimized
     * machine code. Can be overridden with the --warmup option.
     */
    private static final int DEFAULT_WARMUP_RUNS = 3;

    /** Simple holder for the loaded input data. */
    private static final class Matrices {
        final int n;
        final double[] a;
        final double[] b;

        Matrices(int n, double[] a, double[] b) {
            this.n = n;
            this.a = a;
            this.b = b;
        }
    }

    // ------------------------------------------------------------------
    // Data loading
    // ------------------------------------------------------------------

    /**
     * Reads n, A and B from the shared binary file.
     * The whole file is read at once; ByteBuffer then decodes it with
     * explicit little-endian order, so the result is the same on any machine.
     */
    static Matrices loadMatrices(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);

        if (bytes.length < Long.BYTES) {
            throw new IOException("Invalid header in '" + path + "'");
        }
        long nLong = buffer.getLong();
        // Java arrays are indexed by int, so n*n must fit in an int.
        if (nLong <= 0 || nLong * nLong > Integer.MAX_VALUE) {
            throw new IOException("Unsupported matrix size n=" + nLong);
        }
        int n = (int) nLong;
        int count = n * n;

        long expectedBytes = Long.BYTES + 2L * count * Double.BYTES;
        if (bytes.length < expectedBytes) {
            throw new IOException("File '" + path + "' is truncated");
        }

        double[] a = new double[count];
        double[] b = new double[count];
        // A and B are stored one after the other, so two bulk reads suffice.
        buffer.asDoubleBuffer().get(a);
        buffer.position(Long.BYTES + count * Double.BYTES);
        buffer.asDoubleBuffer().get(b);

        return new Matrices(n, a, b);
    }

    // ------------------------------------------------------------------
    // Multiplication kernel
    // ------------------------------------------------------------------

    /**
     * C = A x B using the textbook i-j-k triple loop.
     * Same loop order and same running-sum accumulator as the C and Python
     * versions, so all three languages perform identical arithmetic.
     */
    static void multiply(double[] a, double[] b, double[] c, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    // ------------------------------------------------------------------
    // JIT warm-up
    // ------------------------------------------------------------------

    /**
     * Runs the kernel several times WITHOUT timing it.
     *
     * Why: the JVM starts by interpreting bytecode. Only after a method
     * (or loop) has run many times does the JIT compiler (C1, then the
     * optimizing C2) translate it to native machine code. Without warm-up,
     * the first timed run would mix interpretation, compilation time and
     * compiled execution, so it would not represent Java's steady-state speed.
     *
     * We warm up on the SAME matrices so the JIT sees exactly the loop
     * bounds and data it will see in the timed run. The results are written
     * into c, which is later used by checksum(), so the JIT cannot discard
     * these runs as dead code.
     */
    static void warmUp(Matrices m, double[] c, int runs) {
        for (int run = 0; run < runs; run++) {
            multiply(m.a, m.b, c, m.n);
        }
    }

    // ------------------------------------------------------------------
    // Result check
    // ------------------------------------------------------------------

    /** Sum of all elements of C, used to compare results across languages. */
    static double checksum(double[] c) {
        double total = 0.0;
        for (int idx = 0; idx < c.length; idx++) {
            total += c[idx];
        }
        return total;
    }

    /** Writes n and C to disk (same little-endian format as the input). */
    static void saveMatrix(Path path, double[] c, int n) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES + c.length * Double.BYTES)
                                      .order(ByteOrder.LITTLE_ENDIAN);
        buffer.putLong(n);
        buffer.asDoubleBuffer().put(c);
        Files.write(path, buffer.array());
    }

    // ------------------------------------------------------------------
    // Orchestration
    // ------------------------------------------------------------------

    private static void printUsageAndExit() {
        System.err.println("Usage: java MatMul <matrices_file.bin> [--warmup N] [--output <result.bin>]");
        System.exit(1);
    }

    public static void main(String[] args) throws IOException {
        // Arguments: <input.bin> [--warmup N] [--output <result.bin>]
        String inputPath = null;
        String outputPath = null;
        int warmupRuns = DEFAULT_WARMUP_RUNS;
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--warmup") && i + 1 < args.length) {
                warmupRuns = Integer.parseInt(args[++i]);
            } else if (args[i].equals("--output") && i + 1 < args.length) {
                outputPath = args[++i];
            } else if (inputPath == null) {
                inputPath = args[i];
            } else {
                printUsageAndExit();
            }
        }
        if (inputPath == null) {
            printUsageAndExit();
        }

        // 1. Load inputs - NOT timed (guidelines: inputs prepared outside).
        Matrices m = loadMatrices(Path.of(inputPath));

        // 2. Allocate the output - NOT timed, so we measure only arithmetic.
        double[] c = new double[m.n * m.n];

        // 3. Warm-up - NOT timed, excluded from the results.
        warmUp(m, c, warmupRuns);

        // ============ TIMED REGION START ============
        // System.nanoTime() is Java's monotonic clock: it measures elapsed
        // time and is not affected by changes to the system wall clock.
        long startNs = System.nanoTime();

        multiply(m.a, m.b, c, m.n);

        long endNs = System.nanoTime();
        // ============ TIMED REGION END ==============

        // 4. Report and (optionally) save - NOT timed.
        double elapsedSeconds = (endNs - startNs) / 1e9;
        System.out.printf("n=%d warmup_runs=%d time_s=%.6f checksum=%.10e%n",
                m.n, warmupRuns, elapsedSeconds, checksum(c));

        if (outputPath != null) {
            saveMatrix(Path.of(outputPath), c, m.n);
        }
    }
}
