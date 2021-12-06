#ifndef SIFT_H
#define SIFT_H

#include<iostream>
#include<fstream>

using namespace std;

class Logger {

    string logFile = "log";
    ofstream fout;

public:
    Logger() {
        this->fout.open(this->logFile, ios::out);
    }

    void log(const string &logString) {
        fout << logString << endl;
    }
};

extern Logger *logger;

#endif
