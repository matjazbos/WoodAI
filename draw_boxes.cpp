#include <opencv2/opencv.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Annotation {
    int classId;
    float xCenter;
    float yCenter;
    float width;
    float height;
};

std::vector<Annotation> loadAnnotations(const std::string& annotationPath) {
    std::ifstream in(annotationPath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open annotation file: " + annotationPath);
    }

    std::vector<Annotation> annotations;
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        std::istringstream iss(line);
        Annotation ann{};
        if (!(iss >> ann.classId >> ann.xCenter >> ann.yCenter >> ann.width >> ann.height)) {
            std::cerr << "Skipping invalid annotation line: " << line << std::endl;
            continue;
        }

        annotations.push_back(ann);
    }

    return annotations;
}

cv::Rect yoloToRect(const Annotation& ann, int imageWidth, int imageHeight) {
    const float xCenterPx = ann.xCenter * static_cast<float>(imageWidth);
    const float yCenterPx = ann.yCenter * static_cast<float>(imageHeight);
    const float boxWidthPx = ann.width * static_cast<float>(imageWidth);
    const float boxHeightPx = ann.height * static_cast<float>(imageHeight);

    int x = static_cast<int>(std::round(xCenterPx - boxWidthPx / 2.0f));
    int y = static_cast<int>(std::round(yCenterPx - boxHeightPx / 2.0f));
    int w = static_cast<int>(std::round(boxWidthPx));
    int h = static_cast<int>(std::round(boxHeightPx));

    // Clamp to image boundaries
    x = std::max(0, std::min(x, imageWidth - 1));
    y = std::max(0, std::min(y, imageHeight - 1));

    if (x + w > imageWidth) {
        w = imageWidth - x;
    }
    if (y + h > imageHeight) {
        h = imageHeight - y;
    }

    return cv::Rect(x, y, std::max(0, w), std::max(0, h));
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: draw_boxes <image_path> <annotation_path> <output_path>" << std::endl;
        std::cerr << "Example: draw_boxes input.jpg labels.txt output.jpg" << std::endl;
        return 1;
    }

    const std::string imagePath = argv[1];
    const std::string annotationPath = argv[2];
    const std::string outputPath = argv[3];

    try {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            return 1;
        }

        const auto annotations = loadAnnotations(annotationPath);

        for (const auto& ann : annotations) {
            cv::Rect rect = yoloToRect(ann, image.cols, image.rows);

            if (rect.width <= 0 || rect.height <= 0) {
                continue;
            }

            cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);

            std::string label = "class " + std::to_string(ann.classId);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(
                label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline
            );

            int textX = rect.x;
            int textY = std::max(rect.y - 5, textSize.height + 5);

            cv::rectangle(
                image,
                cv::Point(textX, textY - textSize.height - 4),
                cv::Point(textX + textSize.width + 4, textY + baseline - 4),
                cv::Scalar(0, 255, 0),
                cv::FILLED
            );

            cv::putText(
                image,
                label,
                cv::Point(textX + 2, textY - 4),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0, 0, 0),
                1
            );
        }

        if (!cv::imwrite(outputPath, image)) {
            std::cerr << "Failed to save output image: " << outputPath << std::endl;
            return 1;
        }

        std::cout << "Saved output image to: " << outputPath << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}