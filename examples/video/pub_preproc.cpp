#include <unistd.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <zenoh.hxx>

#include "node/publisher.hpp"

using namespace std;
using namespace cv;

const char kDftLLocator[] = "udp/224.0.0.123:7447#iface=lo";
const char kTopicName[] = "video";
constexpr size_t kPoolSize = 4 * 1024 * 1024;  // 4 MiB

void printUsageAndExit(char* program_name);
void matToVec(Mat& m, vector<uchar>& v);
void dump(ofstream& out, vector<uchar>& arr);

int main(int argc, char* argv[]) {
    if (argc < 2) printUsageAndExit(argv[0]);

    char* input_path = nullptr;
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
    input_path = argv[optind];

    cout << "Will read the video from '" << input_path << '\'' << endl;

    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        cerr << "Failed to open the video file" << endl;
        return 1;
    }

    try {
        auto config = zenoh::Config::create_default();
        config.insert(Z_CONFIG_MODE_KEY, Z_CONFIG_MODE_PEER);
        config.insert(Z_CONFIG_LISTEN_KEY, kDftLLocator);
        zenoh::Session session(std::move(config));
        Publisher pub(session, kTopicName, kPoolSize);
        Mat frame;
        vector<uchar> frame_vec;

        ofstream out;
        if (dump_hash) out.open("send.out");

        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            matToVec(frame, frame_vec);
            uchar* buf_d = (uchar*) pub.malloc(frame_vec.size());
            cudaMemcpy(buf_d, frame_vec.data(), frame_vec.size(),
                       cudaMemcpyHostToDevice);
            pub.put(buf_d, frame_vec.size());
            if (dump_hash) dump(out, frame_vec);
            usleep(10000);
        }
        cout << "Done" << endl;
    } catch (runtime_error& err) {
        cerr << err.what() << endl;
    }

    cap.release();
    return 0;
}

void matToVec(Mat& m, vector<uchar>& v) {
    v.clear();

    if (m.isContinuous()) {
        v.assign(m.data, m.data + m.total() * m.channels());
    } else {
        for (int i = 0; i < m.rows; i++) {
            v.insert(v.end(), m.ptr<uchar>(i),
                     m.ptr<uchar>(i) + m.cols * m.channels());
        }
    }
}

void printUsageAndExit(char* program_name) {
    cerr << "Usage: " << program_name << " OPTIONS INPUT_VIDEO_PATH" << endl;
    cerr << "OPTIONS:" << endl;
    cerr << "  -v       whether dump hash of each frame for verification"
         << endl;
    cerr << "INPUT_VIDEO_PATH:" << endl;
    cerr << "  path to the input video" << endl;
    exit(1);
}

void dump(ofstream& out, vector<uchar>& arr) {
    int hash = 0;
    for (auto ele : arr) {
        hash = (1234 * hash + (int) ele) % 56789;
    }
    out << hash << endl;
}