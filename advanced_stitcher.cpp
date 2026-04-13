%%writefile advanced_stitcher.cpp
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
// Include internal detail headers for manual access to the stitching pipeline stages
#include <opencv2/stitching/detail/autocalib.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/warpers.hpp>

using namespace std;
using namespace cv;
// cv::detail contains the lower-level stitching components we are explicitly controlling
using namespace cv::detail;

int main(int argc, char** argv) {
    // Validate that we have the proper command-line arguments: folder, and output file
    if (argc < 3) {
        cout << "Usage: ./stitcher_app <input_folder> <output_file_path>" << endl;
        return -1;
    }

    String folderPath = argv[1];
    String outputPath = argv[2];
    vector<String> image_paths;

    // Use glob to discover all JPEG files in the specified input directory.
    // The false parameter indicates a non-recursive search.
    glob(folderPath + "/*.jpg", image_paths, false);

    vector<Mat> images;

    cout << "Loading and downscaling images for Colab compatibility..." << endl;
    for (const auto& path : image_paths) {
        // Load each image
        Mat img = imread(path);
        
        // Skip images that cannot be loaded to avoid crashing later operations
        if (img.empty()) continue;

        // ==========================================
        // COLAB FIX 1: AGGRESSIVE DOWNSCALING
        // ==========================================
        // Shrink images to a max dimension of 400px.
        // 30 full-res images will instantly crash Colab because the manual pipeline 
        // stores multiple floating-point representations of warping masks and images in memory.
        double scale = 400.0 / max(img.cols, img.rows);
        if (scale < 1.0) {
            // Resize the image to enforce the RAM constraint
            resize(img, img, Size(), scale, scale);
        }
        images.push_back(img);
    }

    // A panorama requires a minimum of 2 images to be stitched
    if (images.size() < 2) {
        cout << "Error: Need at least 2 images to stitch." << endl;
        return -1;
    }

    cout << "Loaded " << images.size() << " images." << endl;

    // ==========================================
    // STEP 1: FEATURE DETECTION AND MATCHING
    // ==========================================
    cout << "Step 1: Finding and Matching Features (ORB)..." << endl;
    // features will store keypoints and descriptors for every single image
    vector<ImageFeatures> features(images.size());
    
    // Instantiate ORB (Oriented FAST and Rotated BRIEF), a fast and free feature extractor
    Ptr<Feature2D> finder = ORB::create();

    // Loop through each image and extract robust features (keypoints) to match overlapping areas
    for (size_t i = 0; i < images.size(); ++i) {
        computeImageFeatures(finder, images[i], features[i]);
    }

    // pairwise_matches will track which features in img A match features in img B
    vector<MatchesInfo> pairwise_matches;
    
    // Use BestOf2NearestMatcher to find correspondences between images. 0.3f is the match confidence threshold.
    Ptr<FeaturesMatcher> matcher = makePtr<BestOf2NearestMatcher>(false, 0.3f);
    
    // Perform the heavy N x N match computation between all features
    (*matcher)(features, pairwise_matches);
    
    // Free memory discarding low-confidence or unused match results
    matcher->collectGarbage();

    // ==========================================
    // STEP 2: CAMERA ESTIMATION & BUNDLE ADJUSTMENT
    // ==========================================
    cout << "Step 2: Camera Estimation and Bundle Adjustment..." << endl;
    
    // cameras will hold intrinsic (focal length) and extrinsic (rotation) params
    vector<CameraParams> cameras;
    
    // Estimate initial camera alignments based on how features move between matching image pairs
    Ptr<Estimator> estimator = makePtr<HomographyBasedEstimator>();
    if (!(*estimator)(features, pairwise_matches, cameras)) {
        cout << "Error: Homography estimation failed. Images may not overlap enough." << endl;
        return -1;
    }

    // Normalize camera rotation matrices to 32-bit floats for stability
    for (size_t i = 0; i < cameras.size(); ++i) {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }

    // Refine the crude homography cameras using Bundle Adjustment (Ray logic minimizes ray projection error)
    Ptr<BundleAdjusterBase> adjuster = makePtr<BundleAdjusterRay>();
    adjuster->setConfThresh(1.0); // Strict confidence threshold
    
    if (!(*adjuster)(features, pairwise_matches, cameras)) {
        cout << "Error: Bundle Adjustment failed." << endl;
        return -1;
    }

    // Extract all focal lengths to determine the scale for our spherical projection canvas
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i) focals.push_back(cameras[i].focal);
    sort(focals.begin(), focals.end());
    
    // Use the median focal length as the global scale for warping the images
    float warped_image_scale = static_cast<float>(focals[focals.size() / 2]);

    // ==========================================
    // COLAB FIX 2: MANUAL RAM DUMP
    // ==========================================
    cout << "Clearing intermediate RAM..." << endl;
    // Release massive data structures no longer needed after camera parameters are locked
    features.clear();
    pairwise_matches.clear();

    // ==========================================
    // STEP 3: SPHERICAL IMAGE WARPING
    // ==========================================
    cout << "Step 3: Warping to Spherical Canvas..." << endl;
    vector<Point> corners(images.size());         // Top-left coordinate of each image in the panorama
    vector<UMat> masks_warped(images.size());     // Indicates visible pixels for each warped image
    vector<UMat> images_warped(images.size());    // The curved/warped images 
    vector<Size> sizes(images.size());            // Dimensions of warped bounding boxes
    vector<UMat> masks(images.size());            // Base masks (pure white before warping)

    // Create solid white masks the exact same size as the incoming images
    for (size_t i = 0; i < images.size(); ++i) {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Setup the Spherical Warper (best for wide panoramas) mapping images onto a sphere
    Ptr<WarperCreator> warper_creator = makePtr<cv::SphericalWarper>();
    Ptr<RotationWarper> warper = warper_creator->create(warped_image_scale);

    for (size_t i = 0; i < images.size(); ++i) {
        // Prepare the camera intrinsic matrix K
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        
        // Warp the actual image, outputting the deformed image and its top-left corner location on the canvas
        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();
        
        // Warp the mask the exact same way so we know which pixels are valid image vs empty space
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    // ==========================================
    // STEP 4: EXPOSURE COMPENSATION & SEAM FINDING
    // ==========================================
    cout << "Step 4: Compensating Exposure and Finding Seams..." << endl;
    
    // Normalize brightness/exposure differences across overlapping regions so the sky doesn't patchily shift in brightness
    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
    compensator->feed(corners, images_warped, masks_warped);

    // ==========================================
    // COLAB FIX 3: LIGHTWEIGHT SEAM FINDER
    // ==========================================
    // Voronoi is lightning fast and uses almost 0 RAM compared to GraphCut.
    // Seam finding figures out exactly where to slice overlapping images to hide alignment ghosts.
    Ptr<SeamFinder> seam_finder = makePtr<detail::VoronoiSeamFinder>();

    // Voronoi expects floating point images representing gradients
    vector<UMat> images_warped_f(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
    }
    
    // The find() function modifies masks_warped in-place, slicing the masks where seams are optimal
    seam_finder->find(images_warped_f, corners, masks_warped);
    images_warped_f.clear(); // Free memory

    // ==========================================
    // STEP 5: MULTI-BAND BLENDING
    // ==========================================
    cout << "Step 5: Multi-Band Blending..." << endl;
    
    // Multi-band blender uses Laplacian pyramids to smooth low frequencies (colors) over wide ranges 
    // and high frequencies (edges) over short ranges, preventing blur at seams.
    Ptr<Blender> blender = Blender::createDefault(Blender::MULTI_BAND);

    // Determine the size of the final composite canvas using corner bounds and sizes
    Rect dst_roi = resultRoi(corners, sizes);
    blender->prepare(dst_roi);

    // Feed each processed, warped, seamlessly sliced image into the blending pyramid
    for (size_t i = 0; i < images.size(); ++i) {
        // Apply the calculated exposure brightness adjustments
        compensator->apply(i, corners[i], images_warped[i], masks_warped[i]);

        // Blenders require 16-bit signed shorts to safely handle pyramid math without overflowing
        Mat img_warped_s;
        images_warped[i].convertTo(img_warped_s, CV_16S);
        Mat mask_warped_8u;
        masks_warped[i].convertTo(mask_warped_8u, CV_8U);

        // Add the image to the blender
        blender->feed(img_warped_s, mask_warped_8u, corners[i]);
    }

    // Export the finalized panorama and a mask defining valid pixel bounds on the canvas
    Mat result, result_mask;
    blender->blend(result, result_mask);

    // Convert back down to standard 8-bit color for saving to disk
    result.convertTo(result, CV_8U);
    imwrite(outputPath, result);

    cout << "Success! Panorama saved to: " << outputPath << endl;
    return 0;
}