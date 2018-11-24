#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include "Eigen/Dense"

#include "half.hpp"

#define CL_HPP_MINIMUM_OPENCL_VERSION   110
#define CL_HPP_TARGET_OPENCL_VERSION    120
#define CL_HPP_ENABLE_EXCEPTIONS
#include "cl2.hpp"

template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

std::string m_cl_args = "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-denorms-are-zero";
cl::Program m_program;
cl::Device m_device;
cl::Context m_context;

template<class T>
static std::string opencl_dev_type_to_string(T type) {
    if (type == CL_DEVICE_TYPE_CPU) {
        return "CPU";
    } else if (type == CL_DEVICE_TYPE_GPU) {
        return "GPU";
    } else if (type == CL_DEVICE_TYPE_ACCELERATOR) {
        return "Accelerator";
    } else {
        return "Unknown";
    }
}

static std::string trim(std::string trim_me) {
    boost::algorithm::trim(trim_me);
    return trim_me;
}

static size_t next_power_of_two(const size_t x) {
    return 2 << size_t(std::ceil(std::log2(x)) - 1);
}

static void sgemmBatched_ref(const std::vector<half_float::half>& a,
                             const std::vector<half_float::half>& b,
                             std::vector<half_float::half>& c,
                             const int m, const int n, const int k,
                             const int batch_size) {
    std::vector<float> ar(a.size());
    std::vector<float> br(b.size());
    std::vector<float> cr(c.size());

    std::copy(begin(a), end(a), begin(ar));
    std::copy(begin(b), end(b), begin(br));

    for (auto batch = 0; batch < batch_size; batch++) {
        auto offset_u = batch * m * k;
        auto offset_v = batch * n * k;
        auto offset_m = batch * m * n;
#ifdef USE_BLAS
        // Calculates C = transpose(tranpose(A) * B) in row major, or
        // C = A * transpose(B) in column major.
        for (auto i = 0; i < m; i++) {
            for (auto j = 0; j < n; j++) {
                auto acc = 0.0f;
                for (auto l = 0; l < k; l++) {
                    acc += ar[l * m + i + offset_u] * br[l * n + j + offset_v];
                }
                cr[j * m + i + offset_m] = acc;
            }
        }
#else
        auto C = EigenMatrixMap<float>(cr.data() + offset_m, m, n);
        auto A = ConstEigenMatrixMap<float>(ar.data() + offset_u, m, k);
        auto B = ConstEigenMatrixMap<float>(br.data() + offset_v, n, k);
        C.noalias() = (A * B.transpose());
#endif
    }

    std::copy(begin(cr), end(cr), begin(c));
}


static void sgemm_generate_data(std::vector<half_float::half> &x,
                                const int m, const int n,
                                const int batch_size,
                                const int m_ceil, const int n_ceil) {
    for (auto batch = 0; batch < batch_size; batch++) {
        for (auto i = 0; i < n_ceil; i++) {
            if (i < n) {
                for (auto j = 0; j < m; j++) {
                    x[batch*n_ceil*m_ceil + i*m_ceil + j] =
                        (( (i ^ j) + batch - 128) % 256) / 256.0f;
                }
                for (auto j = m; j < m_ceil; j++) {
                    x[batch*n_ceil*m_ceil + i*m_ceil + j] = 0.0f;
                }
            } else {
                for (auto j = 0; j < m_ceil; j++) {
                    x[batch*n_ceil*m_ceil + i*m_ceil + j] = 0.0f;
                }
            }
        }
    }
}

static float compare_ref(std::vector<half_float::half> &x, std::vector<half_float::half> &ref,
                         const int m, const int n, const int batch_size,
                         const int m_ceil, const int n_ceil) {
    auto sum = 0.0f;
    for (auto batch = 0; batch < batch_size; batch++) {
        for (auto j = 0; j < m; j++) {
            for (auto i = 0; i < n; i++) {
                auto r = ref[batch*n*m + j*n + i];
                auto y = x[batch*n_ceil*m_ceil + j*n_ceil + i];

                sum += (r - y) * (r - y);
                // printf("%.2f ", (float)(r-y));
            }
            // printf("\n");
        }
        // printf("\n\n");
    }
    return sum / (m * n * batch_size);
}

int main() {
    std::ifstream t("hgemm.cl");
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string sourceCode = buffer.str();

    std::vector<cl::Platform> platforms;
    try {
        cl::Platform::get(&platforms);
    } catch (const cl::Error &e) {
        printf("OpenCL: %s\n", e.what());
        throw;
    }

    auto best_version = 0.0f;
    cl::Platform best_platform;
    cl::Device best_device;
    std::string best_vendor;
    auto best_score = 0;
    auto found_device = false;
    auto id = 0;

    printf("Detected %d OpenCL platforms.\n", platforms.size());

    for (const auto &p : platforms) {
        std::string platvers = p.getInfo<CL_PLATFORM_VERSION>();
        std::string platprof = p.getInfo<CL_PLATFORM_PROFILE>();
        std::string platname = p.getInfo<CL_PLATFORM_NAME>();
        std::string platvend = p.getInfo<CL_PLATFORM_VENDOR>();
        printf("Platform version: %s\n", platvers.c_str());;
        printf("Platform profile: %s\n", platprof.c_str());
        printf("Platform name:    %s\n", platname.c_str());
        printf("Platform vendor:  %s\n", platvend.c_str());

        std::istringstream versstream(platvers);
        std::string tmp;
        float opencl_version;
        versstream >> tmp >> opencl_version;

        std::vector<cl::Device> devices;
        try {
            p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        } catch (const cl::Error &e) {
            printf("Error getting device(s): %s: %d\n", e.what(), e.err());
            devices.clear();
        }
        for (auto& d : devices) {
            printf("Device ID:     %d\n", id);
            printf("Device name:   %s\n",
                     trim(d.getInfo<CL_DEVICE_NAME>()).c_str());
            printf("Device type:   %s\n",
                     opencl_dev_type_to_string(
                         d.getInfo<CL_DEVICE_TYPE>()).c_str());
            printf("Device vendor: %s\n",
                      d.getInfo<CL_DEVICE_VENDOR>().c_str());
            printf("Device driver: %s\n",
                      d.getInfo<CL_DRIVER_VERSION>().c_str());
            printf("Device speed:  %u MHz\n",
                      d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());
            printf("Device cores:  %u CU\n",
                      d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());

            // assign score, try to find best device
            int this_score = 0;
            std::string this_vendor = d.getInfo<CL_DEVICE_VENDOR>();
            this_score += 1000 * boost::icontains(this_vendor, "advanced micro devices");
            this_score += 1000 * boost::icontains(this_vendor, "amd");
            this_score += 1000 * boost::icontains(this_vendor, "nvidia");
            this_score +=  500 * boost::icontains(this_vendor, "intel");
            this_score +=  100 * (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU);
            this_score +=  opencl_version * 10;
            printf("Device score:  %d\n", this_score);

            if ((this_score > best_score)) {
                best_version = opencl_version;
                best_platform = p;
                best_device = d;
                best_vendor = this_vendor;
                best_score = this_score;
                found_device = true;
            }
            id++;
        }
    }

    if (!found_device) {
        throw std::runtime_error("No suitable OpenCL device found.");
    }

    printf("Selected platform: %s\n",
        best_platform.getInfo<CL_PLATFORM_NAME>().c_str());
    printf("Selected device: %s\n",
        trim(best_device.getInfo<CL_DEVICE_NAME>()).c_str());
    printf("with OpenCL %2.1f capability.\n", best_version);

    cl::Context context;
    try {
        context = cl::Context(best_device);
    } catch (const cl::Error &e) {
        printf("Error creating OpenCL context: %s: %d", e.what(), e.err());
        throw std::runtime_error("Error creating OpenCL context.");
    }
    m_context = context;
    m_device = best_device;

    // Make program of the source code in the context
    cl::Kernel kernel;

    std::string max_option;
    float min_time = 100000.0;
    float min_time_error = 0.0; 

    auto test = [&](int mdimc, int ndimc, int mwg, int nwg, int kwg, int sa, int sb, int vwm, int vwn)
    {

        auto args = m_cl_args;
        auto tune_args = std::string();
        tune_args += " -DMDIMC="+std::to_string(mdimc);
        tune_args += " -DNDIMC="+std::to_string(ndimc);
        tune_args += " -DMWG="+std::to_string(mwg);
        tune_args += " -DNWG="+std::to_string(nwg);
        tune_args += " -DKWG="+std::to_string(kwg);
        tune_args += " -DSA="+std::to_string(sa);
        tune_args += " -DSB="+std::to_string(sb);
        tune_args += " -DVWM="+std::to_string(vwm);
        tune_args += " -DVWN="+std::to_string(vwn);
    
        args += tune_args;

        int batch_size = 36;
        int m = 256;
        int n = 32;
        int k = 256;
       
        // This needs to be at minimum the maximum (MNK/WG) values above.
        auto m_max = std::max(64, m);
        auto n_max = std::max(64, n);
        auto k_max = std::max(32, k);

        auto at_size = batch_size
            * next_power_of_two(k_max) * next_power_of_two(m_max);
        auto b_size = batch_size
            * next_power_of_two(k_max) * next_power_of_two(n_max);
        auto c_size = batch_size
            * next_power_of_two(m_max) * next_power_of_two(n_max);

        auto at = std::vector<half_float::half>(at_size);
        auto b = std::vector<half_float::half>(b_size);
        auto c = std::vector<half_float::half>(c_size);
        auto c_ref = std::vector<half_float::half>(c_size);
    
        auto queue = cl::CommandQueue(m_context,
                                  m_device,
                                  CL_QUEUE_PROFILING_ENABLE);
        auto event = cl::Event();

        auto aBuffer = cl::Buffer(
            m_context,
            CL_MEM_READ_WRITE, sizeof(half_float::half) * at_size, nullptr, nullptr);
        auto bBuffer = cl::Buffer(
            m_context,
            CL_MEM_READ_WRITE, sizeof(half_float::half) * b_size, nullptr, nullptr);
        auto cBuffer = cl::Buffer(
            m_context,
            CL_MEM_READ_WRITE, sizeof(half_float::half) * c_size, nullptr, nullptr);

        sgemm_generate_data(at, k, m, batch_size, k, m);
        sgemm_generate_data(b, n, k, batch_size, n, k);
        sgemmBatched_ref(at, b, c_ref, m, n, k, batch_size);


        queue.enqueueWriteBuffer(aBuffer, CL_FALSE, 0,
                                 at_size * sizeof(half_float::half), at.data());
        queue.enqueueWriteBuffer(bBuffer, CL_FALSE, 0,
                                 b_size * sizeof(half_float::half), b.data());
        queue.finish();

        auto sum_time = 0.0f;
        auto sum_error = 0.0f;

        try {
            m_program = cl::Program(m_context, sourceCode);
            m_program.build(args.c_str());

            kernel = cl::Kernel(m_program, "HgemmBatched");
            kernel.setArg(0, m);
            kernel.setArg(1, n);
            kernel.setArg(2, k);
            kernel.setArg(3, aBuffer);
            kernel.setArg(4, bBuffer);
            kernel.setArg(5, cBuffer);
            for (int i=0; i<4; i++) {
                cl::NDRange local_sgemm = {32 * mdimc / 16, ndimc / 16, 1};
                cl::NDRange size_sgemm = {32 * m / 16 * mdimc / mwg, n / 16 * ndimc / nwg, size_t(batch_size)};
    
                queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                           size_sgemm, local_sgemm,
                                           nullptr, &event);
                queue.finish();
                event.wait();
    
                queue.enqueueReadBuffer(cBuffer, CL_FALSE, 0,
                                        c_size * sizeof(half_float::half), c.data());
                queue.finish();
                auto this_error = compare_ref(c, c_ref, n, m, batch_size, n, m);
                auto elapsed = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                            event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                sum_time += elapsed;
                sum_error += this_error;
            }
        } catch (...) {
            sum_error = 10000;
            sum_time = 0.0f;
        }

        auto time = sum_time * 1e-6 / 4;
        printf("%s %f %f\n", tune_args.c_str(), (double)sum_error / 4, (double) time);
        if(min_time > time) {
            min_time = time;
            min_time_error = sum_error / 4;
            max_option = tune_args;
        }
    };

    for ( int mdimc = 16; mdimc <= 64; mdimc *= 2) {
        for (int ndimc = 16; ndimc <= 32; ndimc *= 2) {
            for ( int mwg = 16; mwg <= 64; mwg *= 2) {
                for ( int nwg = 16; nwg <= 32; nwg *= 2) {
                    if(mwg < mdimc) continue;
                    if(nwg < ndimc) continue;
                    for (int kwg = 16; kwg < 64; kwg *= 2) {
                        for(int sa = 0; sa < 2; sa++) {
                            for(int sb = 0; sb < 2; sb++) {
                                for(int vwm = 1; vwm <= 8; vwm *= 2) {
                                    for(int vwn = 1; vwn <= 8; vwn *= 2) {
                                        if(sa == 0 && vwm != 1) continue;
                                        if(sb == 0 && vwn != 1) continue;
                                        try {
                                            test(mdimc, ndimc, mwg, nwg, kwg, sa, sb, vwm, vwn);
                                        } catch(...) {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    printf("\n\nWinner : %s %f %f\n", max_option.c_str(), (double)min_time_error, (double) min_time);

    return 0;
}

