#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <vector>
#include <fstream>
using namespace std;
using namespace cv;


Mat siftExtract(string imgName) {

    Mat img = imread(imgName, CV_LOAD_IMAGE_GRAYSCALE); 
    // resize(img, img, Size(), 0.625, 0.625);

    // feature detection
    // SiftFeatureDetector detector(0.05, 5.0);
    SiftFeatureDetector detector;
    vector<KeyPoint> keypoints;
    detector.detect(img, keypoints);

    // // feature extraction
    SiftDescriptorExtractor extractor(3.0);
    Mat descr;
    extractor.compute(img, keypoints, descr);
    return descr;
}

void printMatToFile(string path, Mat &mat) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "error: open file " << path << endl;
        exit(-1);
    }

    for (int i = 0; i < mat.rows; ++i) {
        ofs << mat.at<float>(i, 0);
        for (int j = 1; j < mat.cols; ++j) {
            ofs << " " << mat.at<float>(i, j);
        }
        ofs << endl;
    }
    ofs.flush();
    ofs.close();
}

int main(int argv, char **argc) {
    string path = argc[1];
    cout << path << endl;
    // form the output path
    size_t start = path.rfind("/") + 1;
    size_t end = path.rfind(".");
    string filenameNoExt = path.substr(start, end - start);
    string filepath = path.substr(0, start);
    string outputPath = filepath + filenameNoExt + ".opencv.sift";

    // feature extraction
    Mat descr = siftExtract(path);
    // write to output
    printMatToFile(outputPath, descr);
    return 0;
}
