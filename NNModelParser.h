//
// Created by YEONGJUN KIM on 2020/02/22.
//

#ifndef TFLITEDRIVER_NNMODELPARSER_H
#define TFLITEDRIVER_NNMODELPARSER_H

class NNModelParser {
private:
public:
    NNModelParser();
    ~NNModelParser();
    int read(const std::string &modelName, char*& fileBuffer, std::fpos<mbstate_t>& fileSize);
};


#endif //TFLITEDRIVER_NNMODELPARSER_H
