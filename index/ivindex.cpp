#include "ivindex.h"

IvIndex::IvIndex(string path, cv::Mat & codebook, size_t dbSize):
     idf(codebook.rows, 0.0), imgNorm(dbSize, 0.0) {
     load(path);
}

IvIndex::IvIndex(cv::Mat & codebook, size_t dbSize):
     idf(codebook.rows, 0.0), imgNorm(dbSize, 0.0) {

    this->codebookSz = codebook.rows;
    this->featureDim = codebook.cols;
    this->dbSize = dbSize;

    // the first node is head node, don't search head node
    for (size_t i = 0; i < codebookSz; ++i) {
        nodelist.push_back(vector<Node> ());
    }
}

IvIndex::~IvIndex() {
}

float IvIndex::l2Norm(vector<size_t> & hist) {
    float sum = 0.0;
    for (auto x: hist) {
        sum += x * x;
    }
    return sqrt(sum);
}

void IvIndex::rootNormalize(Mat &descr) {
    Mat L1Norm;
    reduce(descr, L1Norm, 1, CV_REDUCE_SUM);
    for (int i = 0; i < descr.rows; ++i) {
        descr.row(i) /= L1Norm.at<float>(i, 0);
    }
    sqrt(descr, descr);
}

Mat IvIndex::readCSV(string filename) {
    ifstream ifs(filename);
    if (!ifs.is_open()) {
        return Mat();
    }

    string line;
    Mat ret;

    while (getline(ifs, line) && !line.empty()) {
        stringstream ss(line);
        vector<float> row;
        float x = 0.0;
        while (ss >> x) {
            row.push_back(x);
        }
        Mat tmp = Mat(row).t();
        ret.push_back(tmp);
    }
    ifs.close();
    return ret;
}

void IvIndex::addToIndex(size_t imgID, string siftpath, cv::flann::Index *flannIndex) {
    /* load descr, descr is a Mat */
    Mat descr = readCSV(siftpath);
    /* check empty feature on some images*/
    if (descr.rows == 0) {
        return;
    }

    rootNormalize(descr);

    Mat indices;     //(numQueries, k, CV_32S);
    Mat dists;       //(numQueries, k, CV_32F);
    flannIndex->knnSearch(descr, indices, dists, 1, cv::flann::SearchParams(150));

    vector<size_t> imhist(codebookSz, 0);
    for (int i = 0; i < indices.rows; ++i) {
        imhist[indices.at<int>(i, 0)] += 1;
    }
    // compute image norm
    imgNorm[imgID] = l2Norm(imhist);

    // build the inverted index table
    for (size_t wordID = 0; wordID < codebookSz; ++wordID) {
        if (imhist[wordID] == 0) {
            // skip zero visual word
            continue;
        }
        vector<Node> &head = nodelist[wordID];
        head.push_back(Node(imgID, imhist[wordID]));
        idf[wordID] += 1.0;
    }
}

void IvIndex::computeIdf() { 
    for (size_t i = 0; i < codebookSz; ++i) {
        if (idf[i] == 0) {
            idf[i] = 0;
        }
        else {
            idf[i] = log10(dbSize / idf[i]);
        }
    }
}

void IvIndex::save(string path) {
    // save inverted index information to disk
    ofstream ofs(path, ios::out);
    if (ofs.is_open()) {
        ofs << codebookSz << endl;
        ofs << featureDim << endl;
        ofs << dbSize << endl;
        for (size_t i = 0; i < codebookSz; ++i) {
            ofs << idf[i] << " ";
        }
        ofs << endl;
        for (size_t i = 0; i < dbSize; ++i) {
            ofs << imgNorm[i] << " ";
        }
        ofs << endl;
        for (size_t i = 0; i < codebookSz; ++i) {
            vector<Node> &head = nodelist[i];
            for (auto x: head) {
                ofs << x.imgID() << " " << x.tf() << " ";
            }
            ofs << endl; 
        }
        ofs.close();
    }
    else {
        cerr << "error: can't open file " << path << endl;
    }
}

void IvIndex::load(string path) {
    // load inverted index information from disk
    ifstream ifs(path);
    string line;
    if (ifs.is_open()) {
        getline(ifs, line);
        codebookSz = stoi(line);

        getline(ifs, line);
        featureDim = stoi(line);

        getline(ifs, line);
        dbSize = stoi(line);

        getline(ifs, line);
        istringstream ss1(line);
        for (size_t i = 0; i < codebookSz; ++i) {
            ss1 >> idf[i];
        }

        getline(ifs, line);
        istringstream ss2(line);
        for (size_t i = 0; i < dbSize; ++i) {
            ss2 >> imgNorm[i];
        }

        for (size_t i = 0; i < codebookSz; ++i) {
            nodelist.push_back(vector<Node>());
        }
        for (size_t i = 0; i < codebookSz; ++i) {
            getline(ifs, line); 
            istringstream ss(line);
            unsigned int imgID = 0;  
            unsigned int tf = 0;
            vector<Node> &head = nodelist[i];
            while (ss >> imgID && ss >> tf) {
                head.push_back(Node(imgID, tf));
            }
        }
        ifs.close();    
    }
    else {
        cerr << "error: can't open file " << path << endl;
    }
}

vector<size_t> IvIndex::argsort(vector<float> &nums, size_t top) {
    size_t n = nums.size();
    vector<pair<float, size_t>> indexNums(n);
    for (size_t i = 0; i < n; ++i) {
        indexNums[i].first = nums[i];
        indexNums[i].second = i;
    }

    sort(indexNums.begin(), indexNums.end(), comp);

    vector<size_t> res(top, 0);
    for (size_t i = 0; i < top; ++i) {
        res[i] = indexNums[i].second;
    }
    return res; 
}

vector<size_t> IvIndex::score(string siftpath, size_t top, cv::flann::Index *flannIndex) {
    /* load descr, descr is a Mat */
    Mat descr = readCSV(siftpath);
    /* check empty feature on some images */
    if (descr.rows == 0) {
        vector<size_t> res(top, 0);
        return res;
    }
    rootNormalize(descr);

    Mat indices;     //(numQueries, k, CV_32S);
    Mat dists;       //(numQueries, k, CV_32F);
    flannIndex->knnSearch(descr, indices, dists, 1, cv::flann::SearchParams(150));

    vector<size_t> imhist(codebookSz, 0);
    for (int i = 0; i < indices.rows; ++i) {
        imhist[indices.at<int>(i, 0)] += 1;
    }

    // score for all db images
    vector<float> scores(dbSize, 0.0);

    // build the inverted index table
    for (size_t wordID = 0; wordID < codebookSz; ++wordID) {
        if (imhist[wordID] == 0) {
            // skip zero visual word
            continue;
        }

        vector<Node> &head = nodelist[wordID];
        for (auto x: head) {
            scores[x.imgID()] += x.tf() * imhist[wordID] * idf[wordID];
        }
    }

    // divied scores by imgNorm
    for (size_t i = 0; i < dbSize; ++i) {
        scores[i] /= imgNorm[i];
    }

    // sort scores in reversed order
    return argsort(scores, top);
}

void IvIndex::quantize(vector<string> siftpaths, string savepath, cv::flann::Index *index) {
    for (size_t imgID = 0; imgID < dbSize; ++imgID) {
        cout << "Process image " << imgID << " ... ... " << endl; 
        addToIndex(imgID,  siftpaths[imgID], index);
    }

    // compute idf
    computeIdf();
    // save index information
    save(savepath);
}

