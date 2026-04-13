%%writefile basic-stitcher.cpp
#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/features2d.hpp"

using namespace cv;
using namespace std;

// Struct to store an image along with its precomputed SIFT features.
// This prevents redundant computation when comparing the same image against multiple others.
struct ImageNode {
    Mat img;
    vector<KeyPoint> keypoints;
    Mat descriptors;
};

// Helper function to extract features once and cache them in ImageNode structures
vector<ImageNode> computeFeatures(const vector<Mat>& images) {
    // Instantiate the SIFT detector. We enforce a strict limit of 500 features per image.
    // This limit dramatically speeds up FLANN matching and Homography estimation later.
    Ptr<SIFT> detector = SIFT::create(500);
    
    // Allocate the output nodes list
    vector<ImageNode> nodes(images.size());
    
    for (size_t i = 0; i < images.size(); i++) {
        nodes[i].img = images[i];
        if(!images[i].empty()) {
            // detectAndCompute calculates both the pixel locations of corners (keypoints) 
            // and their mathematical fingerprints (descriptors)
            detector->detectAndCompute(images[i], noArray(), nodes[i].keypoints, nodes[i].descriptors);
        }
    }
    return nodes;
}

// Function to crop out unnecessary solid black borders that accumulate during perspective warping
Mat trim(Mat image) {
    Mat result, grayImage, thresholded;
    // Convert to grayscale to evaluate pixel brightness
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    // Binarize the image: any pixel > 1 brightness becomes 255 (white). Absolute black (0) stays 0.
    threshold(grayImage, thresholded, 1, 255, THRESH_BINARY);
    
    vector<vector<Point>> contours;
    // Find the outlines of the non-black regions
    findContours(thresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // Fallback if the image lacks distinct content
    if(contours.empty()) return image;
    
    // Compute the minimum bounding rectangle that contains all the valid pixels
    Rect rect = boundingRect(contours[0]);
    // Crop the image to exactly that bounding box
    return image(rect);
}

// Core function that physically stitches two images together using Homography
Mat imageStitch(ImageNode sceneNode, ImageNode objNode) {
    // Safety check: both images need at least 4 descriptors to compute a valid homography matrix
    if(objNode.descriptors.empty() || sceneNode.descriptors.empty() || objNode.descriptors.rows < 4 || sceneNode.descriptors.rows < 4) 
        return sceneNode.img;

    // Use Fast Library for Approximate Nearest Neighbors (FLANN) for lightning-fast matching
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    
    // FLANN requires descriptors to be strictly in 32-bit floating point format
    if(objNode.descriptors.type() != CV_32F) objNode.descriptors.convertTo(objNode.descriptors, CV_32F);
    if(sceneNode.descriptors.type() != CV_32F) sceneNode.descriptors.convertTo(sceneNode.descriptors, CV_32F);
    
    // Compare descriptors and find the closest match for each feature
    matcher.match(objNode.descriptors, sceneNode.descriptors, matches);
    
    vector<Point2f> keypointsobj, keypointsscene;
    // Isolate the physical (x, y) coordinates of the matching feature pairs
    for(int i = 0; i < matches.size(); i++) {
        keypointsobj.push_back(objNode.keypoints[matches[i].queryIdx].pt);
        keypointsscene.push_back(sceneNode.keypoints[matches[i].trainIdx].pt);
    }
    
    // Guarantee enough points exist
    if(keypointsobj.size() < 4) return sceneNode.img;
    
    // Find the 3x3 transformation matrix defining how to deform 'obj' to perspective match 'scene'
    // RANSAC algorithm filters out wildly incorrect feature matches (outliers)
    Mat homography = findHomography(keypointsobj, keypointsscene, RANSAC);
    if(homography.empty()) return sceneNode.img;
    
    // Determine the corner coordinates of the incoming image
    vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f((float)objNode.img.cols, 0);
    obj_corners[2] = Point2f((float)objNode.img.cols, (float)objNode.img.rows);
    obj_corners[3] = Point2f(0, (float)objNode.img.rows);
    
    // Predict where those 4 corners will land in the target 'scene' space after transformation
    vector<Point2f> obj_corners_transformed(4);
    perspectiveTransform(obj_corners, obj_corners_transformed, homography);
    
    // Determine the absolute boundaries of the joint canvas (combining both images)
    float min_x = 0, min_y = 0;
    float max_x = (float)sceneNode.img.cols;
    float max_y = (float)sceneNode.img.rows;
    
    for(int i = 0; i < 4; i++) {
        if (obj_corners_transformed[i].x < min_x) min_x = obj_corners_transformed[i].x;
        if (obj_corners_transformed[i].y < min_y) min_y = obj_corners_transformed[i].y;
        if (obj_corners_transformed[i].x > max_x) max_x = obj_corners_transformed[i].x;
        if (obj_corners_transformed[i].y > max_y) max_y = obj_corners_transformed[i].y;
    }
    
    // Generate an offset Translation matrix. 
    // If transformed coordinates went negative, we shift everything right/down so they start at (0,0)
    Mat T = Mat::eye(3, 3, CV_64F);
    T.at<double>(0, 2) = -min_x;
    T.at<double>(1, 2) = -min_y;
    
    Size new_size((int)(max_x - min_x), (int)(max_y - min_y));
    
    // Safety check to prevent memory explosions caused by insane homography scaling
    if (new_size.width > 8000 || new_size.height > 8000) return sceneNode.img;

    // Warp the new image into the calculated combined canvas size
    Mat result;
    warpPerspective(objNode.img, result, T * homography, new_size);
    
    // Create a bounding box representing where the original 'scene' image belongs on the canvas
    Mat roi(result, Rect((int)-min_x, (int)-min_y, sceneNode.img.cols, sceneNode.img.rows));
    
    // Mask out the empty areas so we can securely overlay 'scene' on top of the warped 'obj'
    Mat mask;
    cvtColor(sceneNode.img, mask, COLOR_BGR2GRAY);
    threshold(mask, mask, 1, 255, THRESH_BINARY); 
    
    // Paste the rigid 'scene' image rigidly on top, using the warped 'obj' as the background fill
    sceneNode.img.copyTo(roi, mask);
    
    // Crop excess black bordering
    return trim(result);
}

// Predicts which images heavily overlap by mapping out coordinate-validated (inlier) feature matches
Mat findInliersMatrix(vector<ImageNode> nodes) {
    // Output matrix where cell (i, j) = the number of reliable overlapping features between image i and image j
    Mat inlierMatrix = Mat::zeros(nodes.size(), nodes.size(), CV_64F);
    FlannBasedMatcher matcher;
    
    cout << "  Matching pairs ";
    // Compare every image against every other image
    for(int i = 0; i < nodes.size(); i++) {
        for(int j = 0 ; j < nodes.size(); j++) {
            if(i == j || nodes[i].descriptors.rows < 4 || nodes[j].descriptors.rows < 4) continue;
            
            // Ensure descriptors are CV_32F for FLANN
            if(nodes[i].descriptors.type() != CV_32F) nodes[i].descriptors.convertTo(nodes[i].descriptors, CV_32F);
            if(nodes[j].descriptors.type() != CV_32F) nodes[j].descriptors.convertTo(nodes[j].descriptors, CV_32F);
            
            // Match the two images' feature lists
            vector<DMatch> matches;
            matcher.match(nodes[i].descriptors, nodes[j].descriptors, matches);
            
            vector<Point2f> keypointsobj, keypointsscene;
            for(int k = 0; k < matches.size(); k++)  {
                keypointsobj.push_back(nodes[i].keypoints[matches[k].queryIdx].pt);
                keypointsscene.push_back(nodes[j].keypoints[matches[k].trainIdx].pt);
            }
            
            int inliersSize = 0;
            if (keypointsobj.size() >= 4) {
                vector<uchar> inliersStatus;
                // Calculate homography. inliersStatus will contain '1' for points that successfully conform
                // to the derived perspective matrix, and '0' for garbage matches.
                findHomography(keypointsobj, keypointsscene, RANSAC, 3, inliersStatus);
                for(int k = 0; k < inliersStatus.size(); k++) {
                    if(inliersStatus[k] == 1) inliersSize++;
                }
            }
            // Record the inlier score
            inlierMatrix.at<double>(i, j) = inliersSize;
        }
        cout << "." << flush; // Print a dot to show it hasn't frozen
    }
    cout << " Done!" << endl;
    return inlierMatrix;
}

// Executes a hierarchical "tournament style" stitching process
vector<Mat> pairwiseImageStitch(vector<ImageNode> nodes) {
    // Figure out which image pairs share the most visual data
    Mat inlierMatrix = findInliersMatrix(nodes);
    
    vector<int> imagesOrder;
    vector<bool> matched(nodes.size(), false); 
    
    // Greedily lock-in pairs with the highest absolute connectivity.
    // For every image 'i', we find the single image 'j' that overlaps with it the absolute best.
    for(int i = 0; i < nodes.size(); i++) {
        if(matched[i]) continue;
        
        Mat row = inlierMatrix.row(i);
        double maxVal; 
        Point maxLoc;
        // Search this row (image i's connections) for the peak overlap
        minMaxLoc(row, NULL, &maxVal, NULL, &maxLoc);
        
        int indexOfRowMax = maxLoc.x;
        
        // If there's an actual match and target isn't already grouped
        if(maxVal > 0 && !matched[indexOfRowMax]) {
            imagesOrder.push_back(i);
            imagesOrder.push_back(indexOfRowMax);
            matched[i] = true;
            matched[indexOfRowMax] = true;
            
            // Wipe data for both images so they can't be chosen again
            inlierMatrix.row(i).setTo(0);
            inlierMatrix.col(i).setTo(0);
            inlierMatrix.row(indexOfRowMax).setTo(0);
            inlierMatrix.col(indexOfRowMax).setTo(0);
        }
    }
    
    // Add any orphan images cleanly at the end so they aren't lost
    for(int i = 0; i < nodes.size(); i++) {
        if(!matched[i]) {
            imagesOrder.push_back(i);
        }
    }
    
    vector<Mat> stitchedImages;
    cout << "  Stitching ";
    
    // Run the physical stitching math on adjacent pairs 
    for(int i = 0 ; i < imagesOrder.size() - 1; i += 2) {
        stitchedImages.push_back(imageStitch(nodes[imagesOrder[i]], nodes[imagesOrder[i+1]]));
        cout << "*" << flush; // Print a star for every successful stitch execution
    }
    
    // If an odd number of images existed, pass the last one directly to the next round unchanged
    if(imagesOrder.size() % 2 != 0) {
        stitchedImages.push_back(nodes[imagesOrder.back()].img);
    }
    cout << " Done!" << endl;
    
    // We reduced the array size by roughly half. This new array is returned for the next iteration.
    return stitchedImages;
}

int main(int argc, char** argv) {
    // Validate command-line arguments: directory of input images, and where to dump the panorama
    if (argc < 3) {
        cout << "Usage: ./app <input_folder_path> <output_file_path>" << endl;
        return -1;
    }

    String folderPath = argv[1];
    String outputPath = argv[2]; 
    String path = folderPath + "/*.jpg";
    
    vector<cv::String> result;
    vector<cv::Mat> images;
    
    // Recursively collect JPEG images from input folder
    cv::glob(path, result, false);
    
    // Load physical matrices into memory
    for (size_t k = 0; k < result.size(); ++k) {
        cv::Mat image = cv::imread(result[k]);
        if (image.empty()) continue;
        images.push_back(image);
    }
    
    cout << "Processing folder: " << folderPath << endl;
    cout << "Target Output: " << outputPath << endl;
    cout << "Number of input images: " << images.size() << endl;
    
    if(images.empty()) return -1;
    
    // Force every raw loaded image down to an 800x800 square to standardize extraction dimensions
    for(int i = 0; i < images.size(); i++) {
        resize(images[i], images[i], Size(800, 800));
    }
    
    vector<Mat> stitchedImages = images;
    int iteration = 1;
    
    // Run a while loop that hierarchically stitches N images down until 1 giant image remains
    while(stitchedImages.size() > 1) {
        cout << "--- Stitching Iteration " << iteration << " (Images: " << stitchedImages.size() << ") ---" << endl;
        
        // Cache the SIFT features of our current pool of sub-panoramas
        vector<ImageNode> currentNodes = computeFeatures(stitchedImages);
        // Execute a round of pairing, turning N images into ~N/2 images
        stitchedImages = pairwiseImageStitch(currentNodes);
        
        // RAM FIX 3: BULLETPROOF SCALING
        // Force every composite/stitched canvas back down to a maximum of 800px.
        // If we didn't do this, iteration 1 produces 1600px images, iteration 2 produces 3200px images,
        // which radically increases execution time exponentially and causes out-of-memory crashes.
        for(size_t i = 0; i < stitchedImages.size(); i++) {
            double max_dim = max(stitchedImages[i].cols, stitchedImages[i].rows);
            if (max_dim > 800) {
                double scale = 800.0 / max_dim;
                resize(stitchedImages[i], stitchedImages[i], Size(), scale, scale);
            }
        }
        
        iteration++;
    }
    
    // Only 1 image remains in the pool: our final panorama.
    if(stitchedImages.size() == 1) {
        cout << "Final Panorama Generated Successfully." << endl;
        imwrite(outputPath, stitchedImages[0]);
    }
    
    return 0;
}