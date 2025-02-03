#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <opencv2/videoio.hpp>
#include <zenoh.hxx>

#include "blur.hpp"
#include "env.hpp"
#include "helpers.hpp"
#include "node/subscriber.hpp"

using namespace std;
using namespace cv;

void printUsageAndExit(char* program_name);
void dump(ofstream& out, uchar* arr, size_t size);

int main(int argc, char* argv[]) {
    if (argc < 4) printUsageAndExit(argv[0]);

    int video_width, video_height;
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

    if (optind + 2 >= argc) printUsageAndExit(argv[0]);
    video_width = stoi(argv[optind]);
    video_height = stoi(argv[optind + 1]);
    output_path = argv[optind + 2];

    cout << "Will write the output to '" << output_path << '\'' << endl;

    try {
        VideoWriter writer(output_path, VideoWriter::fourcc('m', 'p', '4', 'v'),
                           25, Size(video_width, video_height));

        auto config = zenoh::Config::create_default();
        config.insert(Z_CONFIG_MODE_KEY, Z_CONFIG_MODE_PEER);
        config.insert(Z_CONFIG_LISTEN_KEY, env::kDftLLocator);
        zenoh::Session session(std::move(config));
        size_t frame_size = video_width * video_height * 3;
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

        ofstream out;
        if (dump_hash) out.open("recv.out");

        Subscriber sub(session, env::kTopicName, env::kPoolSize,
                       env::kMsgQueueCapExp,
                       [video_width, video_height, dump_hash, frame_blurred_d,
                        frame_blurred, filter_d, &out,
                        &writer](void* data_d, size_t size) {
                           if (dump_hash) {
                               cudaMemcpy(frame_blurred, data_d, size,
                                          cudaMemcpyDeviceToHost);
                               dump(out, frame_blurred, size);
                           }

                           blur((uchar*) data_d, (uchar*) frame_blurred_d,
                                video_width, video_height, filter_d, 5);
                           cudaMemcpy(frame_blurred, frame_blurred_d, size,
                                      cudaMemcpyDeviceToHost);

                           writer.write(Mat(video_height, video_width, CV_8UC3,
                                            frame_blurred));
                       });

        cout << "Ctrl+C to continue" << endl;
        hlp::waitForSigInt();

        cudaFree(filter_d);
        cudaFree(frame_blurred_d);
        delete[] frame_blurred;
        writer.release();
    } catch (runtime_error& err) {
        cerr << err.what() << endl;
    }

    return 0;
}

void printUsageAndExit(char* program_name) {
    cerr << "Usage: " << program_name
         << " [OPTIONS] VIDEO_WIDTH VIDEO_HEIGHT OUTPUT_VIDEO_PATH" << endl;
    cerr << "OPTIONS:" << endl;
    cerr << "  -v       whether dump hash of each frame for verification"
         << endl;
    cerr << "VIDEO_WIDTH:" << endl;
    cerr << "  width of the input video" << endl;
    cerr << "VIDEO_HEIGHT:" << endl;
    cerr << "  height of the input video" << endl;
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