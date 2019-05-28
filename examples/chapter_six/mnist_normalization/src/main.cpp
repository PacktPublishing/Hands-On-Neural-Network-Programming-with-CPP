#include <iostream>
#include <fstream>
#include <exception>
#include <string>
#include <algorithm>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "dataset.hpp"

uint32_t readUnsignedInt32(std::ifstream &stream, size_t position)
{
    stream.seekg(position, std::ios::beg);
    uint32_t temp;
    stream.read(reinterpret_cast<char *>(&temp), sizeof(temp));
    uint32_t result = ((temp << 8) & 0xFF00FF00) | ((temp >> 8) & 0xFF00FF);
    return (result << 16) | (result >> 16);
}

Matrix loadInput(const std::string &imagesFilePath)
{
    std::ifstream imagesStream(imagesFilePath, std::ios::in | std::ios::binary);

    if (!imagesStream.is_open())
        throw std::invalid_argument("failed to open the images file.");

    uint32_t magicNumber = readUnsignedInt32(imagesStream, 0);

    if (magicNumber != 2051)
        throw std::invalid_argument("failed to read magic number in images file.");

    uint32_t numberOfInstances = readUnsignedInt32(imagesStream, 4);
    std::cout << "This file has " << numberOfInstances << " elements.\n";

    uint32_t numberOfRows = readUnsignedInt32(imagesStream, 8);
    std::cout << "Number of rows is " << numberOfRows << ".\n";

    uint32_t numberOfCols = readUnsignedInt32(imagesStream, 12);
    std::cout << "Number of cols is " << numberOfCols << ".\n";

    const size_t size = numberOfRows * numberOfCols;

    Matrix result(size, numberOfInstances);

    std::unique_ptr<unsigned char[]> buffer(new unsigned char[size]);

    for (unsigned instance = 0; instance < numberOfInstances; ++instance)
    {
        auto ref = &buffer[0];
        auto deref = reinterpret_cast<char *>(ref);
        imagesStream.read(deref, size);
        std::copy(ref, ref + size, result.col(instance).data());
    }

    return result;
}

Matrix loadTarget(const std::string &labelsFilePath)
{
    std::ifstream labelsStream(labelsFilePath, std::ios::in | std::ios::binary);

    if (!labelsStream.is_open())
        throw std::invalid_argument("failed to open the labels file.");

    uint32_t magicNumber = readUnsignedInt32(labelsStream, 0);

    if (magicNumber != 2049)
        throw std::invalid_argument("failed to read magic number in labels file.");

    uint32_t numberOfInstances = readUnsignedInt32(labelsStream, 4);
    std::cout << "This file has " << numberOfInstances << " labels\n";

    Matrix result = Matrix::Zero(10, numberOfInstances);

    for (unsigned instance = 0; instance < numberOfInstances; ++instance)
    {
        char label;
        labelsStream.read(&label, 1);
        unsigned index = static_cast<unsigned>(label);
        result(index, instance) = 1.0;
    }

    return result;
}

void navigate(ann::Dataset &dataset)
{
    unsigned int instance = 0;

    char key = 0;

    const char * title = "MNIST";

    cv::namedWindow(title, cv::WindowFlags::WINDOW_AUTOSIZE);

    while(key != 27 && key >= 0)
    {
        auto data = dataset.X.col(instance).data();
        cv::Mat image(28, 28, CV_64FC1, data), resized;
        cv::resize(image, resized, cv::Size(280, 280));
        auto labelCol = dataset.T.col(instance);
        Matrix::Index maxRow;
        labelCol.maxCoeff(&maxRow);

        std::cout << "======================================\n";
        std::cout << std::hex;
        for (unsigned i = 0; i < 28; i++)
        {
            for (unsigned j = 0; j < 28; j++)
            {
                unsigned value = static_cast<unsigned int>(data[i * 28 + j] * 255);
                if (value <= 15)
                    std::cout << "0";
                std::cout << value;
                std::cout << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::dec << "index: " << instance << "\n";
        std::cout << "label: " << maxRow << "\n";

        cv::Mat toSave = resized;
        toSave.convertTo(toSave, CV_8UC1, 255.0); 
        cv::imwrite("output_z.jpg", toSave);

        cv::imshow(title, resized);

        do {
            key = cv::waitKey(0);
        } while(key != 2 && key != 3 && key != 27);

        if(key == 3 && instance < dataset.size()) instance++;
        if(key == 2 && instance > 0) instance--;

    }
}

ann::Dataset loadMNISTDataset(const std::string &imagesFilePath, const std::string &labelsFilePath)
{
    ann::Dataset result;
    result.X = loadInput(imagesFilePath);
    result.T = loadTarget(labelsFilePath);
    return result;
}

Vector fixedNormalization(const Vector &col)
{
    return col / 255.0;
}

Vector zscoreNormalization(const Vector &row)
{
    double min = row.minCoeff();
    double max = row.maxCoeff();

    const auto size = row.size();

    if(max == min)
        return Vector::Zero(size);

    double mean = row.mean();
    double standardDeviation = 0.0;
    for(unsigned i = 0; i < size; ++i)
        standardDeviation += pow(row(i) - mean, 2);
    standardDeviation = sqrt(standardDeviation  / (size - 1));

    return row.unaryExpr([&mean, &standardDeviation](double x) { return (x - mean)/standardDeviation; });
}

Vector minMaxNormalization(const Vector &row)
{
    double min = row.minCoeff();
    double max = row.maxCoeff();

    return row.unaryExpr([&min, &max](double x) { return (x - min)/(max - min); });
}

int main(int, char **)
{
    try
    {
        std::cout << "Loading data...\n";
        auto mnist = loadMNISTDataset("../data/mnist/train-images-idx3-ubyte", "../data/mnist/train-labels-idx1-ubyte");
        std::cout << "Normalizing data...\n";
        mnist.normalize(std::ref(fixedNormalization));
        navigate(mnist);
        std::cout << "exiting...\n";
    }
    catch (std::exception const &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}