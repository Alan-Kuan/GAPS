#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <opencv2/videoio.hpp>
#include <zenoh.hxx>

#include "metadata.hpp"
#include "node/cpp_iface/subscriber.hpp"

using namespace std;
using namespace cv;

const char kDftLLocator[] = "udp/224.0.0.123:7447#iface=lo";
const char kTopicName[] = "video";
constexpr size_t kPoolSize = 3 * 1024 * 1024;  // 3 MiB

__global__ void conv(uchar* frame_in, uchar* frame_out, size_t frame_width,
                     size_t frame_height, double* kernel, size_t kernel_width);

void printUsageAndExit(char* program_name);
void dump(ofstream& out, uchar* arr, size_t size);

int main(int argc, char* argv[]) {
    if (argc < 2) printUsageAndExit(argv[0]);

    char* output_path = nullptr;
    bool dump_hash = false;
    int opt;

    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt) {
        case 'v':
            dump_hash = true;
            break;
        default:
            printUsageAndExit(argv[0]);
        }
    }

    if (optind >= argc) printUsageAndExit(argv[0]);
    output_path = argv[optind];

    cout << "Will write the output to '" << output_path << '\'' << endl;

    try {
        VideoWriter writer(output_path, VideoWriter::fourcc('m', 'p', '4', 'v'),
                           25, Size(600, 316));

        Domain domain{DeviceType::kGPU, 0};
        Subscriber sub(kTopicName, kDftLLocator, domain, kPoolSize);
        size_t frame_size = 600 * 316 * 3;
        uchar* frame_blurred_d;
        cudaMalloc(&frame_blurred_d, frame_size);
        uchar* frame_blurred = new uchar[frame_size];

        double filter[5][5] = {
            {0.003, 0.013, 0.022, 0.013, 0.003},
            {0.013, 0.059, 0.097, 0.059, 0.013},
            {0.022, 0.097, 0.159, 0.097, 0.022},
            {0.013, 0.059, 0.097, 0.059, 0.013},
            {0.003, 0.013, 0.022, 0.013, 0.003},
        };
        double* filter_d;
        cudaMalloc(&filter_d, sizeof(filter));
        cudaMemcpy(filter_d, filter, sizeof(filter), cudaMemcpyHostToDevice);

        ofstream out("recv.out");

        sub.sub([dump_hash, frame_blurred_d, frame_blurred, filter_d, &out,
                 &writer](void* data_d, size_t size) {
            dim3 grid_dim{316 / 32 + 1, 600 / 32 + 1};
            dim3 block_dim{32, 32};

            if (dump_hash) {
                cudaMemcpy(frame_blurred, data_d, size, cudaMemcpyDeviceToHost);
                dump(out, frame_blurred, size);
            }

            conv<<<grid_dim, block_dim>>>((uchar*) data_d,
                                          (uchar*) frame_blurred_d, 600, 316,
                                          filter_d, 5);
            cudaMemcpy(frame_blurred, frame_blurred_d, size,
                       cudaMemcpyDeviceToHost);

            writer.write(Mat(316, 600, CV_8UC3, frame_blurred));
        });

        cout << "Type enter to continue" << endl;
        cin.get();

        cudaFree(filter_d);
        cudaFree(frame_blurred_d);
        delete[] frame_blurred;
        writer.release();
    } catch (runtime_error& err) {
        cerr << err.what() << endl;
    } catch (zenoh::ErrorMessage& err) {
        cerr << err.as_string_view() << endl;
    }

    return 0;
}

__global__ void conv(uchar* frame_in, uchar* frame_out, size_t frame_width,
                     size_t frame_height, double* kernel, size_t kernel_width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= frame_height || col >= frame_width) return;

    size_t half_kernel_width = kernel_width / 2;

    for (int c = 0; c < 3; c++) {
        double sum = 0;
        for (size_t i = 0; i < kernel_width; i++) {
            for (size_t j = 0; j < kernel_width; j++) {
                int row_target = row + i - half_kernel_width;
                int col_target = col + j - half_kernel_width;
                if (row_target < 0 || row_target >= frame_height ||
                    col_target < 0 || col_target >= frame_width)
                    continue;
                sum +=
                    kernel[i * kernel_width + j] *
                    frame_in[(row_target * frame_width + col_target) * 3 + c];
            }
        }
        frame_out[(row * frame_width + col) * 3 + c] = (uchar) sum;
    }
}

void printUsageAndExit(char* program_name) {
    cerr << "Usage: " << program_name << " OPTIONS OUTPUT_VIDEO_PATH" << endl;
    cerr << "OPTIONS:" << endl;
    cerr << "  -v       whether dump hash of each frame for verification"
         << endl;
    cerr << "OUTPUT_VIDEO_PATH:" << endl;
    cerr << "  output path for the blurred video" << endl;
    exit(1);
}

void dump(ofstream& out, uchar* arr, size_t size) {
    int hash = 0;
    for (size_t i = 0; i < size; i++) {
        hash = (1234 * hash + (int) arr[i]) % 56789;
    }
    out << hash << endl;
}