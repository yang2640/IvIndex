#include "ivindex.h"
#include <sys/stat.h>

vector<string> readlines(string path) {
    ifstream ifs(path);
    vector<string> lines;
    if (ifs.is_open()) {
        string line;
        while (getline(ifs, line)) {
            if (line.empty()) {
                continue;
            } 
            lines.push_back(line);
        } 
        ifs.close();
    }
    else {
        cerr << "error: can't open file " << path << endl;
    }
    return lines;
}

inline bool exists(string name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

int main() {
    /* load codebook */
    cout << "load codebook ... ..." << endl;
    string codebookPath = "data/cluster/codebook.yml";
    Mat codebook;
    FileStorage codebookFs(codebookPath, FileStorage::READ);
    codebookFs["codebook"] >> codebook;
    codebookFs.release();

    /* load flann */
    cout << "build flann ... ... " << endl;
    string flannpath = "data/cache/flann.index";
    cv::flann::IndexParams *indexParams = nullptr;
    cv::flann::Index *flannIndex = nullptr;
    if (exists(flannpath)) {
        indexParams = new cv::flann::SavedIndexParams(flannpath);
        flannIndex = new cv::flann::Index(codebook, *indexParams);
    }
    else {
        indexParams = new cv::flann::KMeansIndexParams(); 
        flannIndex = new cv::flann::Index(codebook, *indexParams);
        flannIndex->save(flannpath);
    }

    /* quantize */
    vector<string> siftpaths = readlines("data/featlist");
    string savepath = "data/cache/ivindex.txt";

    IvIndex ivindex(codebook, siftpaths.size());
    ivindex.quantize(siftpaths, savepath, flannIndex);

    if (indexParams != nullptr) {
        delete indexParams;
        indexParams = nullptr;
    }
    if (flannIndex != nullptr) {
        delete flannIndex;
        flannIndex = nullptr;
    }
    return 0;
}

