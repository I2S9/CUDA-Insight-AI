#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// JSON output helper
void print_json_start() {
    std::cout << "{\n";
}

void print_json_key_value(const std::string& key, const std::string& value, bool last = false) {
    std::cout << "  \"" << key << "\": \"" << value << "\"" << (last ? "\n" : ",\n");
}

void print_json_key_number(const std::string& key, double value, bool last = false) {
    std::cout << "  \"" << key << "\": " << value << (last ? "\n" : ",\n");
}

void print_json_key_array_start(const std::string& key) {
    std::cout << "  \"" << key << "\": [\n";
}

void print_json_array_number(double value, bool last = false) {
    std::cout << "    " << value << (last ? "\n" : ",\n");
}

void print_json_end() {
    std::cout << "}\n";
}

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Measure kernel execution time using CUDA events
double measure_kernel_time(int (*kernel_func)(), int warmup_iterations = 3, int measure_iterations = 10) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warmup
    for (int i = 0; i < warmup_iterations; i++) {
        kernel_func();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Measure
    std::vector<float> times;
    for (int i = 0; i < measure_iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        kernel_func();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        times.push_back(elapsed_ms);
    }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // Calculate statistics
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    
    std::sort(times.begin(), times.end());
    double median = times.size() % 2 == 0 
        ? (times[times.size()/2 - 1] + times[times.size()/2]) / 2.0
        : times[times.size()/2];
    
    double min_time = *std::min_element(times.begin(), times.end());
    double max_time = *std::max_element(times.begin(), times.end());
    
    // Calculate standard deviation
    double variance = 0.0;
    for (double t : times) {
        variance += (t - mean) * (t - mean);
    }
    double std_dev = std::sqrt(variance / times.size());
    
    // Output JSON
    print_json_start();
    print_json_key_number("mean_time_ms", mean);
    print_json_key_number("median_time_ms", median);
    print_json_key_number("min_time_ms", min_time);
    print_json_key_number("max_time_ms", max_time);
    print_json_key_number("std_dev_ms", std_dev);
    print_json_key_number("iterations", measure_iterations);
    print_json_key_array_start("all_times_ms");
    for (size_t i = 0; i < times.size(); i++) {
        print_json_array_number(times[i], i == times.size() - 1);
    }
    std::cout << "  ]\n";
    print_json_end();
    
    return mean;
}

// Load shared library and get kernel function
#ifdef _WIN32
typedef HMODULE LibraryHandle;
LibraryHandle load_library(const char* path) {
    return LoadLibraryA(path);
}

void* get_symbol(LibraryHandle handle, const char* symbol) {
    return (void*)GetProcAddress(handle, symbol);
}

void close_library(LibraryHandle handle) {
    FreeLibrary(handle);
}
#else
typedef void* LibraryHandle;
LibraryHandle load_library(const char* path) {
    return dlopen(path, RTLD_LAZY);
}

void* get_symbol(LibraryHandle handle, const char* symbol) {
    return dlsym(handle, symbol);
}

void close_library(LibraryHandle handle) {
    dlclose(handle);
}
#endif

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <library_path> <function_name> [iterations]" << std::endl;
        std::cerr << "Example: " << argv[0] << " saxpy.so test_saxpy 10" << std::endl;
        return 1;
    }
    
    const char* library_path = argv[1];
    const char* function_name = argv[2];
    int iterations = (argc > 3) ? std::atoi(argv[3]) : 10;
    
    // Check CUDA availability
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        std::cerr << "{\"error\": \"No CUDA devices found\"}" << std::endl;
        return 1;
    }
    
    // Get device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    // Load library
    void* handle = load_library(library_path);
    if (!handle) {
        std::cerr << "{\"error\": \"Failed to load library: " << library_path << "\"}" << std::endl;
        return 1;
    }
    
    // Get function pointer
    typedef int (*TestFunc)();
    TestFunc kernel_func = (TestFunc)get_symbol(handle, function_name);
    
    if (!kernel_func) {
        std::cerr << "{\"error\": \"Function not found: " << function_name << "\"}" << std::endl;
        close_library(handle);
        return 1;
    }
    
    // Measure kernel execution
    measure_kernel_time(kernel_func, 3, iterations);
    
    close_library(handle);
    return 0;
}

