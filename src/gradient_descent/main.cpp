#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <charconv>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <filesystem>
#include <omp.h>
#include "gradient_descent.hpp"

#ifdef USE_ITT
    #include <ittnotify.h>
    #define VTUNE_RESUME() __itt_resume()
    #define VTUNE_PAUSE()  __itt_pause()
#else
    #define VTUNE_RESUME()
    #define VTUNE_PAUSE()
#endif

#ifdef SILENT
    #define LOG(msg)
#else
    #define LOG(msg) std::cout << msg
#endif

namespace fs = std::filesystem;

void log_to_csv(const std::string& log_file, const std::string& ds_name,
                size_t n, size_t d, int threads,
                double load_t, double fit_t) {
    std::ofstream file(log_file, std::ios::app);
    if (file.tellp() == 0) {
        file << "dataset,samples,features,threads,load_time,fit_time,total_time\n";
    }
    file << ds_name << "," << n << "," << d << "," << threads << ","
         << std::fixed << std::setprecision(6)
         << load_t << "," << fit_t << "," << (load_t + fit_t) << "\n";
}

Matrix load_csv_to_matrix(const std::string& filepath, size_t& out_N, size_t& out_D) {
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("Error: Could not open " + filepath);

    struct stat sb;
    fstat(fd, &sb);
    size_t length = sb.st_size;
    if (length == 0) {
        close(fd);
        throw std::runtime_error("Error: File is empty");
    }

    const char* addr = static_cast<const char*>(mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0));
    if (addr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Error: mmap failed");
    }

    std::vector<size_t> line_offsets;
    line_offsets.push_back(0);

    for (size_t i = 0; i < length; ++i) {
        if (addr[i] == '\n') line_offsets.push_back(i + 1);
    }
    if (addr[length-1] != '\n') line_offsets.push_back(length);

    size_t num_samples = line_offsets.size() - 1;

    size_t cols = 0;
    size_t first_line_end = line_offsets[1] - 1;
    for (size_t i = 0; i < first_line_end; ++i) {
        if (addr[i] == ',') cols++;
    }
    cols++;

    Matrix mat(num_samples, cols);
    float* raw_mat = mat.data.get();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_samples; ++i) {
        const char* current = addr + line_offsets[i];
        const char* line_end = addr + line_offsets[i+1];

        for (size_t j = 0; j < cols; ++j) {
            while (current < line_end && (*current == ',' || *current == ' ' || *current == '\r')) {
                current++;
            }
            auto res = std::from_chars(current, line_end, raw_mat[i * cols + j]);
            current = res.ptr;
        }
    }

    munmap((void*)addr, length);
    close(fd);

    out_N = num_samples;
    out_D = cols;
    return mat;
}



int main(int argc, char* argv[]) {
    VTUNE_PAUSE();

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path.csv> [log_file.csv]" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    std::string log_filename = (argc > 2) ? argv[2] : "bench_results.csv";
    std::string ds_name = fs::path(filepath).filename().string();

    LOG("================================================\n");
    LOG("   Parallel Gradient Descent - OpenMP Version   \n");
    LOG("================================================\n");

    size_t N = 0, D = 0;

    double start_load = omp_get_wtime();
    Matrix data(0, 0);
    try {
        data = load_csv_to_matrix(filepath, N, D);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    double end_load = omp_get_wtime();
    double load_duration = end_load - start_load;

    int threads = omp_get_max_threads();

    LOG("Dataset: " << ds_name << " (" << N << "x" << D << ")\n");
    LOG("Threads: " << threads << "\n");
    LOG("Load Time: " << load_duration << "s\n");

    float learningRate = 0.1f;
    int numIterations = 5000;
    normalizeFeatures(data);

    VTUNE_RESUME();
    double start_fit = omp_get_wtime();

    std::vector<float> weights = runGradientDescent(data, learningRate, numIterations, false);

    double end_fit = omp_get_wtime();
    VTUNE_PAUSE();

    double fit_duration = end_fit - start_fit;

    log_to_csv(log_filename, ds_name, N, D, threads, load_duration, fit_duration);
    LOG("Results logged to " << log_filename << "\n");

    LOG("\n--- Execution Summary ---\n");
    LOG("Fit Execution Time: " << std::fixed << std::setprecision(6) << fit_duration << " seconds\n");

    if (!weights.empty()) {
        LOG("\nModel Coefficients (first 10):\n");
        size_t num_weights = weights.size();
        for (size_t i = 0; i < (num_weights < 10 ? num_weights : 10); ++i) {
            LOG("  Beta[" << i << "]: " << weights[i] << "\n");
        }
    }
    LOG("================================================\n");

    return 0;
}
