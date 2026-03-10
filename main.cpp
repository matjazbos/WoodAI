#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

cv::Mat stitchImagesHorizontal(const std::vector<cv::Mat>& images) {
    if (images.empty()) 
        throw std::runtime_error("No images provided");

    int h = images[0].rows;
    int type = images[0].type();
    int totalW = 0;
    for (const auto& img : images) {
        if (img.empty()) 
            throw std::runtime_error("Empty image");
        if (img.rows != h || img.type() != type)
            throw std::runtime_error("All images must have same height and type");
        totalW += img.cols;
    }

    cv::Mat out(h, totalW, type);
    int x = 0;
    for (const auto& img : images) {
        img.copyTo(out(cv::Rect(x, 0, img.cols, img.rows)));
        x += img.cols;
    }
    return out;
}

int main() {
    std::vector<cv::Mat> images = {
        cv::imread("/workdir/WoodDataset/images/0_0.png"),
        cv::imread("/workdir/WoodDataset/images/0_1.png"),
        cv::imread("/workdir/WoodDataset/images/0_5.png"),
        cv::imread("/workdir/WoodDataset/images/0_6.png")
    };

    cv::Mat stitched = stitchImagesHorizontal(images);
    cv::imwrite("/workdir/output/stitched.png", stitched);
    return 0;
}