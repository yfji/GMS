#include "gms_matcher.h"

void runImagePair() {
	Mat img1 = imread("./data/027.jpg");
	Mat img2 = imread("./data/103.jpg");

	imresize(img1, 480);
	imresize(img2, 480);

	vector<KeyPoint> kp1; 
	vector<KeyPoint> kp2;
	vector<DMatch> matches=matchImage(img1, img2, kp1, kp2, 8);

	cv::Mat match_img=DrawInlier(img1, img2, kp1, kp2, matches, 2);
	cv::imshow("match", match_img);
	cv::waitKey();
}


int main()
{
	runImagePair();
    return 0;
}

