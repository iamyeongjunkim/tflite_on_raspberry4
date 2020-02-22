#include <iostream>

#include "NNModelParser.h"
#include "errorType.h"

#include "flatbuffers/flatbuffers.h"
#include "schema_generated.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    const tflite::Model* model = nullptr;
    NNModelParser *nnModelParser = new NNModelParser();
    std::string modelName = "mobilenet_v1_1.0_224_quant.tflite";

    char* fileBuffer;
    std::fpos<mbstate_t> fileSize = 0;
    if (nnModelParser->read(modelName, fileBuffer, fileSize) == ReturnError::FILE_OPEN_OK) {
        std::cout << "file open ok!" << std::endl;
        if (fileBuffer != nullptr) {
            model = tflite::GetModel((const void*)fileBuffer);
            auto version = model->version();
            auto description = model->description();
            std::cout << "model version : " << version << std::endl;
            std::cout << "model description : " << description->c_str() << std::endl;

            auto subGraphs = model->subgraphs();
            auto operator_codes = model->operator_codes();
            auto buffers = model->buffers();

            uint32_t idx = 0;
            for (auto operator_code : *operator_codes) {
                std::cout << idx++ << ". " << tflite::EnumNameBuiltinOperator((tflite::BuiltinOperator)(operator_code->builtin_code())) << std::endl;
            }

            std::cout << "num of subgraphs : " << subGraphs->size() << std::endl;
            for (auto subGraph : *subGraphs) {
                std::cout << "num of inputs: " << subGraph->inputs()->size() << std::endl;
                for (auto input : *subGraph->inputs()) {
                    std::cout << "input id: " << input << std::endl;
                }
                std::cout << "num of outputs: " << subGraph->outputs()->size() << std::endl;
                for (auto output : *subGraph->outputs()) {
                    std::cout << "output id: " << output << std::endl;
                }
                std::cout << "num of operators: " << subGraph->operators()->size() << std::endl;
                std::cout << "num of tensors: " << subGraph->tensors()->size() << std::endl;
                for (auto _operator : *subGraph->operators()) {
                    for (auto input : *_operator->inputs()) {
                        std::cout << "input index: " << input << std::endl;
                        auto tensor = subGraph->tensors()->Get(input);
                        std::cout << "tensor name: " << tensor->name()->c_str() << std::endl;
                        std::cout << "tensor type: " << tflite::EnumNameTensorType(tensor->type()) << std::endl;
                        std::cout << "tensor shape: ";
                        for (auto shape : *tensor->shape()) {
                            std::cout << shape << ", ";
                        }
                        std::cout << std::endl;

                        if (tensor->quantization() != nullptr) {
                            if (tensor->quantization()->max() != nullptr) {
                                std::cout << "tensor quantization max: ";
                                for (auto max : *tensor->quantization()->max()) {
                                    std::cout << max << ", ";
                                }
                                std::cout << std::endl;
                            }

                            if (tensor->quantization()->min() != nullptr) {
                                std::cout << "tensor quantization min: ";
                                for (auto min : *tensor->quantization()->min()) {
                                    std::cout << min << ", ";
                                }
                                std::cout << std::endl;
                            }

                            if (tensor->quantization()->scale() != nullptr) {
                                std::cout << "tensor quantization scale: ";
                                for (auto scale : *tensor->quantization()->scale()) {
                                    std::cout << scale << ", ";
                                }
                                std::cout << std::endl;
                            }

                            if (tensor->quantization()->zero_point() != nullptr) {
                                std::cout << "tensor quantization zero point: ";
                                for (auto zeroPoint : *tensor->quantization()->zero_point()) {
                                    std::cout << zeroPoint << ", ";
                                }
                                std::cout << std::endl;
                            }
                        }
                    }

                    for (auto output : *_operator->outputs()) {
                        std::cout << "output index: " << output << std::endl;
                        auto tensor = subGraph->tensors()->Get(output);
                        std::cout << "tensor name: " << tensor->name()->c_str() << std::endl;
                        std::cout << "tensor type: " << tflite::EnumNameTensorType(tensor->type()) << std::endl;
                        std::cout << "tensor shape: ";
                        for (auto shape : *tensor->shape()) {
                            std::cout << shape << ", ";
                        }
                        std::cout << std::endl;

                        if (tensor->quantization() != nullptr) {
                            if (tensor->quantization()->max() != nullptr) {
                                std::cout << "tensor quantization max: ";
                                for (auto max : *tensor->quantization()->max()) {
                                    std::cout << max << ", ";
                                }
                                std::cout << std::endl;
                            }

                            if (tensor->quantization()->min() != nullptr) {
                                std::cout << "tensor quantization min: ";
                                for (auto min : *tensor->quantization()->min()) {
                                    std::cout << min << ", ";
                                }
                                std::cout << std::endl;
                            }

                            if (tensor->quantization()->scale() != nullptr) {
                                std::cout << "tensor quantization scale: ";
                                for (auto scale : *tensor->quantization()->scale()) {
                                    std::cout << scale << ", ";
                                }
                                std::cout << std::endl;
                            }

                            if (tensor->quantization()->zero_point() != nullptr) {
                                std::cout << "tensor quantization zero point: ";
                                for (auto zeroPoint : *tensor->quantization()->zero_point()) {
                                    std::cout << zeroPoint << ", ";
                                }
                                std::cout << std::endl;
                            }
                        }
                    }
                    auto optionType = _operator->builtin_options_type();
                    switch (optionType) {
                        case tflite::BuiltinOptions::BuiltinOptions_Conv2DOptions: {
                            auto option = (tflite::Conv2DOptions *) _operator->builtin_options();
                            std::cout << "stride_h : " << option->stride_h() << std::endl;
                            std::cout << "stride_w : " << option->stride_w() << std::endl;
                            std::cout << "padding: " << tflite::EnumNamePadding(option->padding()) << std::endl;
                            std::cout << "fused_activation_function: "
                                      << tflite::EnumNameActivationFunctionType(option->fused_activation_function())
                                      << std::endl;
                            break;
                        }
                        case tflite::BuiltinOptions::BuiltinOptions_Pool2DOptions: {
                            auto option = (tflite::Pool2DOptions*) _operator->builtin_options();
                            std::cout << "stride_h : " << option->stride_h() << std::endl;
                            std::cout << "stride_w : " << option->stride_w() << std::endl;
                            std::cout << "filter_height : " << option->filter_height() << std::endl;
                            std::cout << "filter_width : " << option->filter_width() << std::endl;
                            std::cout << "padding: " << tflite::EnumNamePadding(option->padding()) << std::endl;
                            std::cout << "fused_activation_function: "
                                      << tflite::EnumNameActivationFunctionType(option->fused_activation_function())
                                      << std::endl;
                            break;
                        }
                        case tflite::BuiltinOptions::BuiltinOptions_DepthwiseConv2DOptions: {
                            auto option = (tflite::DepthwiseConv2DOptions*) _operator->builtin_options();
                            std::cout << "stride_h : " << option->stride_h() << std::endl;
                            std::cout << "stride_w : " << option->stride_w() << std::endl;
                            std::cout << "depth_multiplier: " << option->depth_multiplier() << std::endl;
                            std::cout << "fused_activation_function: "
                                      << tflite::EnumNameActivationFunctionType(option->fused_activation_function())
                                      << std::endl;
                            break;
                        }
                        case tflite::BuiltinOptions::BuiltinOptions_SoftmaxOptions: {
                            auto option = (tflite::SoftmaxOptions*) _operator->builtin_options();
                            std::cout << "beta : " << option->beta() << std::endl;
                            break;
                        }
                        case tflite::BuiltinOptions::BuiltinOptions_ReshapeOptions: {
                            auto option = (tflite::ReshapeOptions*) _operator->builtin_options();
                            for (auto shape : *option->new_shape()) {
                                std::cout << shape << ", ";
                            }
                            std::cout << std::endl;
                            break;
                        }
                        default: {
                            case tflite::BuiltinOptions::BuiltinOptions_NONE: {
                                std::cout << "Not yet supported!" << std::endl;
                                break;
                            }
                        }
                    }
                }
            }
        }
    } else {
        std::cout << "file open error!" << std::endl;
    }

    delete nnModelParser;
    std::cout << "Goodbye, World!" << std::endl;
    return 0;
}
