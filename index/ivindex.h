#ifndef IVINDEX_H_INCLUDED
#define IVINDEX_H_INCLUDED

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <opencv/cv.h>
#include <opencv2/ml/ml.hpp>
#include <opencv2/flann/flann.hpp>
#include <assert.h>

using namespace std;
using namespace cv;

class Node {
public:
    Node(unsigned int id, unsigned int tf) {
        // highest 8 bit is tf, lowest 24 bit is imgID 
        val = tf;
        val <<= 24;
        val |= id; 
    } 

    inline unsigned int imgID() {
        return val & 0xFFFFFF; 
    }
    
    inline unsigned int tf() {
        return ((val & 0xFF000000) >> 24); 
    }

    unsigned int val;
};

class IvIndex {
public:
    IvIndex(string path, cv::Mat & codebook, size_t dbSize);
    IvIndex(cv::Mat & codebook, size_t dbSize);
    ~IvIndex();

    float l2Norm(vector<size_t> &hist);

    void addToIndex(size_t imgID, string siftpath, cv::flann::Index *flannIndex);

    void computeIdf();

    void save(string path);

    void load(string path);

    vector<size_t> argsort(vector<float> &nums, size_t top);

    vector<size_t> score(string siftpath, size_t top, cv::flann::Index *flannIndex);

    void rootNormalize(Mat &descr);

    Mat readCSV(string filename);

    void quantize(vector<string> siftpaths, string savepath, cv::flann::Index *index);

    struct Comparator {
        bool operator() (pair<float, size_t> p1, pair<float, size_t> p2) {
            return p1.first > p2.first; 
        }
    } comp;

private:
    size_t codebookSz;
    size_t featureDim;
    size_t dbSize;
    vector<vector<Node> > nodelist;
    vector<float> idf;
    vector<float> imgNorm;
};
#endif
