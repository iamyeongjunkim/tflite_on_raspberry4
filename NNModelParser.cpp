//
// Created by YEONGJUN KIM on 2020/02/22.
//

#include <iostream>
#include <fstream>
#include <vector>

#include "NNModelParser.h"
#include "errorType.h"

NNModelParser::NNModelParser() {
    std::cout << __FUNCTION__ << "(+)" << std::endl;
    std::cout << __FUNCTION__ << "(-)" << std::endl;
}

NNModelParser::~NNModelParser() {
    std::cout << __FUNCTION__ << "(+)" << std::endl;
    std::cout << __FUNCTION__ << "(-)" << std::endl;
}

int NNModelParser::read(const std::string &modelName, char*& fileBuffer, std::fpos<mbstate_t>& fileSize) {
    std::cout << __FUNCTION__ << "(+) with " << modelName << std::endl;
    std::ifstream openFile;
    openFile.open(modelName, std::ios::binary | std::ios::ate);
    if (!openFile.is_open()) {
        std::cout << "file open error!" << std::endl;
        return ReturnError::FILE_OPEN_ERROR;
    } else {
        std::cout << "file open ok!" << std::endl;
        fileSize = openFile.tellg();
        std::cout << "file size is " << fileSize << std::endl;

        openFile.seekg(0, std::ios::beg);
        char* buffer = new char[fileSize];
        if (openFile.read(buffer, fileSize)) {
            std::cout << "file read ok!" << std::endl;
            fileBuffer = buffer;
            openFile.close();
        } else {
            std::cout << "file read error!" << std::endl;
        }
    }

    std::cout << __FUNCTION__ << "(-)" << std::endl;
    return ReturnError::FILE_OPEN_OK;
}

