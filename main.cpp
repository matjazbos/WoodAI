#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <array>
#include <string>
#include <regex>
#include <filesystem>

namespace fs = std::filesystem;

const int W = 640;
const int H = 640;
const float CONF_THRESH = 0.25f;
const std::string MODEL_PATH = "/workdir/output/runs/wood_knots/weights/best.onnx";
const std::string TMP_OUTPUT_LOCATION = "/tmp/wood_ai_output/";
const fs::path IMG_DIR = "/workdir/WoodDataset/images";

std::vector<fs::path> get_matching_files(const fs::path& dir, int number) {
    std::vector<std::pair<int, fs::path>> temp;
    std::regex pattern("^" + std::to_string(number) + "_([0-9]+)\\.png$");

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string filename = entry.path().filename().string();
        std::smatch match;
        if (std::regex_match(filename, match, pattern)) {
            int index = std::stoi(match[1].str());
            temp.push_back({index, entry.path()});
        }
    }

    std::sort(temp.begin(), temp.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });

    std::vector<fs::path> result;
    for (const auto& [index, path] : temp) {
        result.push_back(path);
    }

    return result;
}

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

int infer(const std::string& image_path, Ort::Value& infer_out, cv::Mat& original_img_out){

    // ONNX Runtime session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, MODEL_PATH.c_str(), so);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    const char* input_name = input_name_alloc.get();
    const char* output_name = output_name_alloc.get();

    // Load image
    cv::Mat img = cv::imread(image_path.c_str());
    if (img.empty()) {
        std::cerr << "Failed to read image\n";
        return 1;
    }

    // Keep original for drawing
    original_img_out = img.clone();

    // Minimal preprocessing: resize, BGR->RGB, float32, [0,1], HWC->CHW
    cv::Mat resized, rgb, f32;
    cv::resize(img, resized, cv::Size(W, H));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32F, 1.0 / 255.0);

    std::vector<float> input_tensor_values(3 * W * H);
    size_t idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                input_tensor_values[idx++] = f32.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    std::array<int64_t, 4> input_shape{1, 3, H, W};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    // Inspect output first
    auto& out = outputs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();

    // std::cout << "Output shape: ";
    // for (auto d : shape) std::cout << d << " ";
    // std::cout << "\n";

    // Expecting [1, 300, 6]
    if (shape.size() != 3 || shape[0] != 1 || shape[2] != 6) {
        std::cerr << "Unexpected output shape\n";
        return 1;
    }

    infer_out = std::move(outputs[0]);
    return 0;
}

void tag_image(Ort::Value& infer_output, cv::Mat& original_img, size_t idx){
    auto info = infer_output.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();

    if (shape.size() != 3 || shape[0] != 1 || shape[2] != 6) {
        throw std::runtime_error("Invalid output shape in tag_image");
    }

    const int num_det = static_cast<int>(shape[1]);
    const int fields = static_cast<int>(shape[2]);

    float* data = infer_output.GetTensorMutableData<float>();

    // Scale from 640x640 back to original image size
    const float sx = static_cast<float>(original_img.cols) / W;
    const float sy = static_cast<float>(original_img.rows) / H;

    for (int i = 0; i < num_det; ++i) {
        float x1   = data[i * fields + 0];
        float y1   = data[i * fields + 1];
        float x2   = data[i * fields + 2];
        float y2   = data[i * fields + 3];
        float conf = data[i * fields + 4];
        int cls    = static_cast<int>(data[i * fields + 5]);

        if (conf < CONF_THRESH) {
            continue;
        }

        // Map back to original image coordinates
        int left   = std::max(0, std::min(static_cast<int>(x1 * sx), original_img.cols - 1));
        int top    = std::max(0, std::min(static_cast<int>(y1 * sy), original_img.rows - 1));
        int right  = std::max(0, std::min(static_cast<int>(x2 * sx), original_img.cols - 1));
        int bottom = std::max(0, std::min(static_cast<int>(y2 * sy), original_img.rows - 1));

        if (right <= left || bottom <= top) {
            continue;
        }

        cv::rectangle(
            original_img,
            cv::Point(left, top),
            cv::Point(right, bottom),
            cv::Scalar(0, 255, 0),
            2
        );

        std::string label = cv::format("%.2f", conf);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline
        );

        int text_y = std::max(top, text_size.height + 4);
        cv::rectangle(
            original_img,
            cv::Point(left, text_y - text_size.height - 4),
            cv::Point(left + text_size.width, text_y + baseline - 4),
            cv::Scalar(0, 255, 0),
            cv::FILLED
        );

        cv::putText(
            original_img,
            label,
            cv::Point(left, text_y - 2),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            1
        );

        std::cout << "det " << i
                  << ": [" << x1 << ", " << y1 << ", " << x2 << ", " << y2
                  << "] conf=" << conf << " cls=" << cls << "\n";
    }


    std::string output_loc = TMP_OUTPUT_LOCATION + "result_" + std::to_string(idx) + ".jpg";
    cv::imwrite(output_loc, original_img);
    std::cout << output_loc << " saved"  << std::endl;
}

int process_board(int board_id){
    auto files = get_matching_files(IMG_DIR, board_id);
    std::cout << "Found " << files.size() << " files" << std::endl;
    if (files.empty()) {
        std::cerr << "No files matched board id " << board_id << std::endl;
        return 1;
    }

    size_t idx = 0;

    for (const auto& image_path : files) {
        std::cout << image_path << std::endl;

        cv::Mat original_img;
        Ort::Value infer_out;

        int infer_status = infer(image_path, infer_out, original_img);
        if(infer_status != 0){
            return infer_status;
        }
        tag_image(infer_out, original_img, idx++);
    }

    std::vector<cv::Mat> images;

    for (size_t i = 0; i < idx; i++){

        images.push_back(cv::imread(TMP_OUTPUT_LOCATION + "result_" + std::to_string(i) + ".jpg"));
    }

    cv::Mat stitched = stitchImagesHorizontal(images);
    cv::imwrite("/workdir/output/stitched.png", stitched);

    return 0;
}


int main(int argc, char* argv[]) {

    int status;
    fs::create_directories(TMP_OUTPUT_LOCATION);

    if (argc == 2) {
        const std::string board_id_str = argv[1];
        int board_id = std::stoi(board_id_str);
        status = process_board(board_id);
    } else if (argc == 1){

    } else {
        std::cerr << "Usage: 'wood_ai' or 'wood_ai <board_id>' " << std::endl;
        status = 1;
    }

    return status;
}