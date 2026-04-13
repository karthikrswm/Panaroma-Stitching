#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

// Use standard and OpenCV namespaces to avoid prefixing with std:: and cv::
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // Verify that the program received exactly the expected number of arguments:
    // argv[0] = program name, argv[1] = input folder, argv[2] = output file
    if (argc < 3) {
        cout << "Usage: ./app <input_folder_path> <output_file_path>" << endl;
        return -1;
    }

    // Extract the input directory and desired output file path from command line arguments
    String folderPath = argv[1];
    String outputPath = argv[2];
    
    String path = folderPath + "/*.jpg";

    // 'result' will store the absolute file paths of all matching images
    vector<String> result;
    // 'raw_images' will hold the actual loaded image matrices for the stitcher
    vector<Mat> raw_images;

    // Use OpenCV's glob utility to find all files matching the pattern. 
    // false = non-recursive search (only search the top level of the folder)
    glob(path, result, false);

    // Iterate through all the discovered file paths
    for (size_t k = 0; k < result.size(); ++k) {
        // Load the image into memory using imread
        Mat image = imread(result[k]);
        
        // Skip invalid or unreadable images (prevents crashing during processing)
        if (image.empty()) continue;

        // RAM Protection: Calculate a scale factor to prevent out-of-memory errors on limited hardware.
        // 1200.0 is the maximum allowed dimension (width or height).
        double scale = 1200.0 / max(image.cols, image.rows);
        
        // If the original image is larger than 1200 pixels on its longest side, scale it down
        if (scale < 1.0) {
            // Resize the image in-place based on the calculated scale ratio
            resize(image, image, Size(), scale, scale);
        }

        // Add the carefully processed/resized image to our main stitching collection
        raw_images.push_back(image);
    }

    // Inform the user about the initial processing status
    cout << "Processing folder: " << folderPath << endl;
    cout << "Number of input images: " << raw_images.size() << endl;

    // Halt execution if the glob search found no valid JPEG files
    if(raw_images.empty()) {
        cout << "Error: No images found in the directory." << endl;
        return -1;
    }

    // 'pano' will store the final assembled panorama image matrix
    Mat pano;
    
    // Initialize the built-in Stitcher module.
    // The 'PANORAMA' mode configures the stitcher for standard spherical warping and blending,
    // which is ideal for landscapes and large scene aggregations.
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);

    cout << "Stitching in progress. This may take a moment..." << endl;
    
    // Execute the main stitching pipeline: feature detection, matching, camera estimation,
    // warping, and seam blending. Returns a status enum indicating success or failure type.
    Stitcher::Status status = stitcher->stitch(raw_images, pano);

    // Evaluate the stitcher's exit status
    if (status == Stitcher::OK) {
        cout << "Final Panorama Generated Successfully!" << endl;
        // Save the resulting stitched matrix 'pano' mapped to the designated filesystem path.
        // Using imwrite avoids needing a GUI display window or HighGUI backend, ideal for headless environments.
        imwrite(outputPath, pano);
    } else {
        // Stitcher::OK is 0. Other failure codes indicate:
        // 1 = ERR_NEED_MORE_IMGS (Not enough overlapping features to form a sequence)
        // 2 = ERR_HOMOGRAPHY_EST_FAIL (Resolution mismatch or difficult geometry)
        // 3 = ERR_CAMERA_PARAMS_ADJUST_FAIL (Failed to solve camera orientation parameters)
        cout << "Error stitching. Error Code: " << int(status) << endl;
    }

    return 0;
}