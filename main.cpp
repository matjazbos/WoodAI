#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <array>
#include <string>
#include <regex>
#include <filesystem>
#include <set>
#include <csignal>

namespace fs = std::filesystem;

// Input size expected by the ONNX model.
const int W = 640;
const int H = 640;

// Minimum confidence required to keep a detection.
const float CONF_THRESH = 0.25f;

// Paths for model, intermediate outputs, final board outputs, and source images.
const std::string MODEL_PATH = "/workdir/output/runs/wood_knots/weights/best.onnx";
const std::string TMP_OUTPUT_LOCATION = "/tmp/wood_ai_output/";
const std::string BOARDS_OUTPUT_LOCATION = "/workdir/output/boards/";
const fs::path IMG_DIR = "/workdir/WoodDataset/images/";

// Find files in the input directory that match "<board_id>_<index>.png".
// Returned files are sorted by the trailing index so board segments stay in order.
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

    // Sort matched images by segment index.
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

// Concatenate a sequence of images left-to-right into one wide image.
// All images must share the same height and type.
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

// Run model inference for one image.
// On success:
// - infer_out receives the model output tensor
// - original_img_out receives a copy of the original image for later annotation
int infer(const std::string& image_path, Ort::Value& infer_out, cv::Mat& original_img_out){

    // ONNX Runtime session
    // Create runtime environment and session for the detector model.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, MODEL_PATH.c_str(), so);

    // Query model input and output names.
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

    // Flatten image into CHW tensor layout expected by many vision models.
    std::vector<float> input_tensor_values(3 * W * H);
    size_t idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                input_tensor_values[idx++] = f32.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    // Input tensor shape: batch=1, channels=3, height=H, width=W.
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

    // Execute the model and collect outputs.
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
    // Each detection row is assumed to be [x1, y1, x2, y2, conf, cls].
    if (shape.size() != 3 || shape[0] != 1 || shape[2] != 6) {
        std::cerr << "Unexpected output shape\n";
        return 1;
    }

    // Move the first output tensor to the caller.
    infer_out = std::move(outputs[0]);
    return 0;
}

// Draw detections above threshold on the original image and save the result.
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

        // Skip invalid or degenerate bounding boxes.
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

        // Use confidence as the display label.
        std::string label = cv::format("%.2f", conf);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline
        );

        // Keep label inside the image vertically.
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

        // std::cout << "det " << i
        //           << ": [" << x1 << ", " << y1 << ", " << x2 << ", " << y2
        //           << "] conf=" << conf << " cls=" << cls << "\n";
    }

    // Save this annotated segment to a temporary file for later stitching.
    std::string output_loc = TMP_OUTPUT_LOCATION + "result_" + std::to_string(idx) + ".jpg";
    cv::imwrite(output_loc, original_img);
    // std::cout << output_loc << " saved"  << std::endl;
}

// Process all image segments for a single board id, annotate them, then stitch them.
int process_board(int board_id){
    auto files = get_matching_files(IMG_DIR, board_id);
    // std::cout << "Found " << files.size() << " files" << std::endl;
    if (files.empty()) {
        std::cerr << "No files matched board id " << board_id << std::endl;
        return 1;
    }

    size_t idx = 0;

    for (const auto& image_path : files) {
        // std::cout << image_path << std::endl;

        cv::Mat original_img;
        Ort::Value infer_out;

        int infer_status = infer(image_path, infer_out, original_img);
        if(infer_status != 0){
            return infer_status;
        }
        tag_image(infer_out, original_img, idx++);
    }

    std::vector<cv::Mat> images;

    // Reload all annotated temporary images in order.
    for (size_t i = 0; i < idx; i++){
        images.push_back(cv::imread(TMP_OUTPUT_LOCATION + "result_" + std::to_string(i) + ".jpg"));
    }

    // Merge annotated segments into one board-wide image.
    cv::Mat stitched = stitchImagesHorizontal(images);
    cv::imwrite(BOARDS_OUTPUT_LOCATION + "board_" + std::to_string(board_id) + ".png", stitched);

    return 0;
}

// Extract all unique board ids from filenames shaped like "<board_id>_<index>.png".
std::vector<int> get_distinct_first_numbers(const fs::path& dir) {
    std::set<int> numbers;
    std::regex pattern(R"(^([0-9]+)_[0-9]+\.png$)");

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string filename = entry.path().filename().string();
        std::smatch match;

        if (std::regex_match(filename, match, pattern)) {
            numbers.insert(std::stoi(match[1].str()));
        }
    }

    return std::vector<int>(numbers.begin(), numbers.end());
}

// Global stop flag set by signal handlers for graceful interruption.
volatile std::sig_atomic_t g_stop_requested = 0;

// Async-signal-safe handler: only set a flag.
void signal_handler(int) {
    g_stop_requested = 1;
}

int main(int argc, char* argv[]) {
    // Register handlers so Ctrl+C / termination requests stop the loop cleanly.
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Ensure output directories exist before processing.
    fs::create_directories(TMP_OUTPUT_LOCATION);
    fs::create_directories(BOARDS_OUTPUT_LOCATION);

    int status = 0;

    if (argc == 2) {
        // One argument means: process a single board id.
        const std::string board_id_str = argv[1];
        int board_id = std::stoi(board_id_str);
        status = process_board(board_id);
    } else if (argc == 1){
        // No arguments means: process every board discovered in IMG_DIR.
        auto board_ids = get_distinct_first_numbers(IMG_DIR);
        for(const auto& board_id : board_ids) {
            // std::cout << board_id << std::endl;
            if (g_stop_requested) {
                std::cerr << "Interrupted, stopping cleanly.\n";
                status = 130;
                break;
            }
            status = process_board(board_id);
            if(status != 0){
                break;
            }
        }
    } else {
        std::cerr << "Usage: 'wood_ai' or 'wood_ai <board_id>' " << std::endl;
        status = 1;
    }

    return status;
}