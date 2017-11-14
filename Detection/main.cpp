// main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<conio.h>           

#include "Blob.h"

#define SHOW_STEPS            

// global variables 
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

// function prototypes 
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs);
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount, int &truckCount, int &left, int &right, int &carCountrev, int &truckdown, int &left2, int &right2, int &intHorizontalLinePosition2, int &direction);
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
void drawCarCountOnImage(int &carCount, int &carCountrev, cv::Mat &imgFrame2Copy);
void drawTruckCountOnImage(int &truckup, int &truckdown, cv::Mat &imgFrame2Copy);


int main(void) {

	cv::VideoCapture capVideo;

	cv::Mat imgFrame1;
	cv::Mat imgFrame2;

	int direction; //1 for UP and 2 for DOWN

	std::vector<Blob> blobs, commonblobs;
	std::vector<Blob> truckblobs;

	cv::Point crossingLine[2], crossingLine2[2];

	int carCount = 0;
	int carCountrev = 0;
	int truckup = 0, truckdown = 0;

	capVideo.open("car4.mp4");

	if (!capVideo.isOpened()) {                                                 // if unable to open video file
		std::cout << "error reading video file" << std::endl << std::endl;      // show error message
		_getch();                  
		return(0);                                                              // and exit program
	}

	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
		std::cout << "error: video file must have at least two frames";
		_getch();                  
		return(0);
	}

	capVideo.read(imgFrame1);
	capVideo.read(imgFrame2);

	int intHorizontalLinePosition = (int)std::round((double)imgFrame1.rows * 0.55); //55
	int intHorizontalLinePosition2 = (int)std::round((double)imgFrame1.rows * 0.45); //55

	crossingLine[0].x = 0;
	crossingLine[0].y = intHorizontalLinePosition;			//First line for oncoming(downward) trafic

	crossingLine[1].x = ((int)imgFrame1.cols / 2 - 40);
	crossingLine[1].y = intHorizontalLinePosition;

	crossingLine2[0].x = ((int)imgFrame1.cols/2 + 60);		//Second line for outgoing(upward) traffic 
	crossingLine2[0].y = intHorizontalLinePosition2;

	crossingLine2[1].x = imgFrame1.cols - 1;
	crossingLine2[1].y = intHorizontalLinePosition2;

	//PARAMETERS USED TO DETECTHE CARS AND SUV's IN UNIVERSITY PARKING LOT NEAR CACS

	//int intHorizontalLinePosition = (int)std::round((double)imgFrame1.rows * 0.50); //55
	//int intHorizontalLinePosition2 = (int)std::round((double)imgFrame1.rows * 0.45); //55

	//crossingLine[0].x = 350;
	//crossingLine[0].y = intHorizontalLinePosition;			

	//crossingLine[1].x = ((int)imgFrame1.cols / 2 + 220);
	//crossingLine[1].y = intHorizontalLinePosition;

	//crossingLine2[0].x = ((int)imgFrame1.cols / 2 + 260);		
	//crossingLine2[0].y = intHorizontalLinePosition2;

	//crossingLine2[1].x = imgFrame1.cols - 1;
	//crossingLine2[1].y = intHorizontalLinePosition2;

	char chCheckForEscKey = 0;

	bool blnFirstFrame = true;

	int frameCount = 2;

	while (capVideo.isOpened() && chCheckForEscKey != 27) {

		std::vector<Blob> currentFrameBlobs, currentBlobs;
		std::vector<Blob> currentFrameBlobsTruck;

		cv::Mat imgFrame1Copy = imgFrame1.clone();
		cv::Mat imgFrame2Copy = imgFrame2.clone();

		cv::Mat imgDifference;
		cv::Mat imgThresh;

		cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);		//converting image to black and white
		cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);

		cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);	//smoothing the image and removing the noise in the video
		cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

		cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);		//finding absolute difference of the images

		cv::threshold(imgDifference, imgThresh, 15, 255.0, CV_THRESH_BINARY); //35	//applying threshold so that the we will get all the objects in white color and remaining in black color

		cv::Canny(imgThresh, imgThresh, 50, 50 * 2, 3);		//finds the edges of the image for better detection

		cv::imshow("imgThresh1", imgThresh);

		cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

		for (unsigned int i = 0; i < 1; i++) {
			cv::dilate(imgThresh, imgThresh, structuringElement3x3);	//dilation fills in the gaps in the object for better reconigation
			cv::dilate(imgThresh, imgThresh, structuringElement3x3);
			cv::erode(imgThresh, imgThresh, structuringElement3x3);		//erode will trim the edges
		}

		cv::imshow("imgThresh2", imgThresh);

		cv::Mat imgThreshCopy = imgThresh.clone();

		std::vector<std::vector<cv::Point> > contours;

		cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);		//finding contours from the output of above operations

		drawAndShowContours(imgThresh.size(), contours, "imgContours");

		std::vector<std::vector<cv::Point> > convexHulls(contours.size());

		for (unsigned int i = 0; i < contours.size(); i++) {
			cv::convexHull(contours[i], convexHulls[i]);		//creating convex hulls for the contours
		}

		drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");

		for (auto &convexHull : convexHulls) {
			Blob possibleBlob(convexHull);
			if(possibleBlob.currentBoundingRect.area() > 1000){
			currentBlobs.push_back(possibleBlob);			//saving all the blobs that have area greater than 1000 into a vector
			}
		}

		drawAndShowContours(imgThresh.size(), currentBlobs, "imgCurrentFrameBlobs");

		if (blnFirstFrame == true) {						//If this is the first iteration save all the blobs to commonblobs vector
			for (auto &currentFrameBlob : currentBlobs) {
				commonblobs.push_back(currentFrameBlob);
			}
		}
		else {
			matchCurrentFrameBlobsToExistingBlobs(commonblobs, currentBlobs);	//if this is second iteration and so on match all the blobs found in this iteration to the existing blobs so that we can track and update the positon of the blobs that are found in this iteration to previous ones
		}

		drawAndShowContours(imgThresh.size(), commonblobs, "imgBlobs");

		imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

		drawBlobInfoOnImage(commonblobs, imgFrame2Copy);

		bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(commonblobs, intHorizontalLinePosition, carCount , truckup, crossingLine2[0].x, crossingLine2[1].x, carCountrev, truckdown,crossingLine[0].x, crossingLine[1].x, intHorizontalLinePosition2, direction);
		//checking whether any blobs crossed the line or not

		if (blnAtLeastOneBlobCrossedTheLine == true)
		{
			if (direction == 1)	//if corssed flash the correct line to green color
			{
				cv::line(imgFrame2Copy, crossingLine2[0], crossingLine2[1], SCALAR_GREEN, 2);
				cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
			}
			else
			{
				cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
				cv::line(imgFrame2Copy, crossingLine2[0], crossingLine2[1], SCALAR_RED, 2);
			}
			direction = 0; //Making it NULL
		}
		else
		{
			cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
			cv::line(imgFrame2Copy, crossingLine2[0], crossingLine2[1], SCALAR_RED, 2);
		}

		drawCarCountOnImage(carCount, carCountrev, imgFrame2Copy);		//displaying all the present data on the image
		drawTruckCountOnImage(truckup, truckdown, imgFrame2Copy);

		cv::imshow("imgFrame2Copy", imgFrame2Copy);

		currentBlobs.clear();			//clearing all the temporary data

		imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

		if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {
			capVideo.read(imgFrame2);
		}
		else {
			std::cout << "end of video\n";
			break;
		}
		blnFirstFrame = false;
		frameCount++;
		chCheckForEscKey = cv::waitKey(1);
	}

	if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
		cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
	}
	// note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows

	return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs) {

	for (auto &existingBlob : existingBlobs) {

		existingBlob.blnCurrentMatchFoundOrNewBlob = false;

		existingBlob.predictNextPosition();			//predict the next position of all the existing blobs using weighted averaging
	}

	for (auto &currentFrameBlob : currentFrameBlobs) {		//pick one blob

		int intIndexOfLeastDistance = 0;
		double dblLeastDistance = 100000.0;	

		for (unsigned int i = 0; i < existingBlobs.size(); i++) {

			if (existingBlobs[i].blnStillBeingTracked == true) {		
				//compare that blob with all the existing blobs and find the distance between the blob and all the predicted positions of existing blobs

				double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

				if (dblDistance < dblLeastDistance) {		//find the existingblob with lease least distance between it and currentblob
					dblLeastDistance = dblDistance;
					intIndexOfLeastDistance = i;
				}
			}
		}

		if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {	//if the distance is less than the half the diagonal size of existing blob then update the centerposition of the existing blob with current blobs center position and update its area, bounding rectangle and counter details
			addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
		}
		else {
			addNewBlob(currentFrameBlob, existingBlobs);	//If not consider it as new blob and add it to th existing blobs
		}

	}

	for (auto &existingBlob : existingBlobs) {	//if a blob is not found for consecutive 5 frames stop tracking it.

		if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
			existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
		}

		if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
			existingBlob.blnStillBeingTracked = false;
		}

	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {

	existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
	existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

	existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

	existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
	existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

	existingBlobs[intIndex].blnStillBeingTracked = true;
	existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {

	currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

	existingBlobs.push_back(currentFrameBlob);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
double distanceBetweenPoints(cv::Point point1, cv::Point point2) {

	int intX = abs(point1.x - point2.x);	//general distance formulae
	int intY = abs(point1.y - point2.y);

	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
	cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

	cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {

	cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	std::vector<std::vector<cv::Point> > contours;

	for (auto &blob : blobs) {
		if (blob.blnStillBeingTracked == true) {
			contours.push_back(blob.currentContour);
		}
	}

	cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

	cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount, int &truckCount, int &left, int &right, int &carCountrev, int &truckdown, int &left2, int &right2, int &intHorizontalLinePosition2, int &direction) {
	bool blnAtLeastOneBlobCrossedTheLine = false;

	for (auto blob : blobs) {

		if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
			int prevFrameIndex = (int)blob.centerPositions.size() - 2;
			int currFrameIndex = (int)blob.centerPositions.size() - 1;

			if ((blob.centerPositions[prevFrameIndex].y > intHorizontalLinePosition2 && blob.centerPositions[currFrameIndex].y <= intHorizontalLinePosition2) && ((blob.centerPositions[currFrameIndex].x > left && blob.centerPositions[currFrameIndex].x < right) && (blob.centerPositions[prevFrameIndex].x > left && blob.centerPositions[prevFrameIndex].x < right))) {
			//upwards traffic counter
		    //check the Y coordinate of the centerposition in previous frame is greater than line’s position and lesser than line’s position in the current frame that means the blob crossed the line and is moving UPWARDS

			//COMMENTED NUMBERS ARE THE PARAMETERS USED TO DETECT CAR's AND SUV's FOR THE VIDEO TAKEN FROM PARKING LOT IN CACS

				if (blob.currentBoundingRect.area() > 6000 &&				//18500
					blob.dblCurrentAspectRatio > 0.2 &&
					blob.dblCurrentAspectRatio < 4.0 &&
					blob.currentBoundingRect.width > 50 &&			//160
					blob.currentBoundingRect.height > 30 &&			//105
					blob.dblCurrentDiagonalSize > 60.0 &&
					(cv::contourArea(blob.currentContour) / (double)blob.currentBoundingRect.area()) > 0.50) {
					//currentFrameBlobsTruck.push_back(blob);
					blnAtLeastOneBlobCrossedTheLine = true;
					std::cout << "\n\nTruck Upward: ";
					std::cout << "Width and Height:" << blob.currentBoundingRect.width << "and" << blob.currentBoundingRect.height << "Width:" << blob.currentBoundingRect.area();
					truckCount++;
				}
				else
				{
					if (blob.currentBoundingRect.area() > 1000 &&			//1000
						blob.currentBoundingRect.area() < 3000 &&
						blob.dblCurrentAspectRatio > 0.2 &&
						blob.dblCurrentAspectRatio < 4.0 &&
						blob.currentBoundingRect.width > 30 &&		//130	
						blob.currentBoundingRect.height > 30 &&		//90
						blob.dblCurrentDiagonalSize > 40.0 &&
						(cv::contourArea(blob.currentContour) / (double)blob.currentBoundingRect.area()) > 0.50) {
						//currentFrameBlobs.push_back(blob);
						blnAtLeastOneBlobCrossedTheLine = true;
						std::cout << "\n\nCar Upward: ";
						std::cout << "Width and Height:" << blob.currentBoundingRect.width << "and" << blob.currentBoundingRect.height << "Width:" << blob.currentBoundingRect.area();
						carCount++;
					}
				}

				direction = 1;
			}
			else
			{
				if ((blob.centerPositions[currFrameIndex].y > intHorizontalLinePosition && blob.centerPositions[prevFrameIndex].y <= intHorizontalLinePosition) && ((blob.centerPositions[currFrameIndex].x > left2 && blob.centerPositions[currFrameIndex].x < right2) && (blob.centerPositions[prevFrameIndex].x > left2 && blob.centerPositions[prevFrameIndex].x < right2)))
				{
					std::cout << "Second Condition";
					if (blob.currentBoundingRect.area() > 20000 &&				//20000
						blob.dblCurrentAspectRatio > 0.2 &&
						blob.dblCurrentAspectRatio < 4.0 &&
						blob.currentBoundingRect.width > 50 &&			//130
						blob.currentBoundingRect.height > 130 &&			//105
						blob.currentBoundingRect.height < 185 &&
						blob.dblCurrentDiagonalSize > 60.0 &&
						(cv::contourArea(blob.currentContour) / (double)blob.currentBoundingRect.area()) > 0.50) {
						//currentFrameBlobsTruck.push_back(blob);
						blnAtLeastOneBlobCrossedTheLine = true;
						std::cout << "\n\nTruck Downward: ";
						std::cout << "Width and Height:" << blob.currentBoundingRect.width << "and" << blob.currentBoundingRect.height << "Width:" << blob.currentBoundingRect.area();
						truckdown++;
					}
					else
					{
						if (blob.currentBoundingRect.area() > 4000 &&			//1000						
							blob.dblCurrentAspectRatio > 0.2 &&
							blob.dblCurrentAspectRatio < 4.0 &&
							blob.currentBoundingRect.width > 50 &&				//150
							blob.currentBoundingRect.height > 50 &&				//100
							blob.dblCurrentDiagonalSize > 40.0 &&
							(cv::contourArea(blob.currentContour) / (double)blob.currentBoundingRect.area()) > 0.50) {
							//currentFrameBlobs.push_back(blob);
							blnAtLeastOneBlobCrossedTheLine = true;
							std::cout << "\n\nCar Downward: ";
							std::cout << "Width: " << blob.currentBoundingRect.width << " and Height:" << blob.currentBoundingRect.height << "Area: " << blob.currentBoundingRect.area();
							carCountrev++;
						}
					}
					direction = 2;	//to flash the line correctly this variable is the counter
				}
			}
		}

	}

	return blnAtLeastOneBlobCrossedTheLine;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy) {

	for (unsigned int i = 0; i < blobs.size(); i++) {

		if (blobs[i].blnStillBeingTracked == true) {	//displays rectanle around the image
			cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);
			//Debugging Purpose and intially detect area and height and width

			//int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
			//double dblFontScale = blobs[i].dblCurrentDiagonalSize / 60.0;
			//int intFontThickness = (int)std::round(dblFontScale * 1.0);
			//
			////cv::putText(imgFrame2Copy, std::to_string(blobs[i].currentBoundingRect.area()), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
			////cv::putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawCarCountOnImage(int &carCount, int &carCountrev, cv::Mat &imgFrame2Copy) {	//displays all the car counters

	int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
	double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
	int intFontThickness = (int)std::round(dblFontScale * 1.5);

	cv::Size textSize1 = cv::getTextSize(std::to_string(carCount), intFontFace, dblFontScale, intFontThickness, 0);
	cv::Size textSize2 = cv::getTextSize(std::to_string(carCountrev), intFontFace, dblFontScale, intFontThickness, 0);

	cv::Point ptTextBottomLeftPosition;
	cv::Point ptTextBottomRightPosition;

	ptTextBottomRightPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize1.width * 1.25);
	ptTextBottomRightPosition.y = (int)((double)textSize1.height * 1.25);

	ptTextBottomLeftPosition.x = 0;
	ptTextBottomLeftPosition.y = (int)((double)textSize2.height * 1.25);

	cv::putText(imgFrame2Copy, std::to_string(carCount), ptTextBottomRightPosition, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);

	cv::putText(imgFrame2Copy, std::to_string(carCountrev), ptTextBottomLeftPosition, intFontFace, dblFontScale, SCALAR_RED, intFontThickness);

}

void drawTruckCountOnImage(int &truckup, int &truckdown, cv::Mat &imgFrame2Copy) {	//displays all the truck counters

	int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
	double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
	int intFontThickness = (int)std::round(dblFontScale * 1.5);

	cv::Size textSize1 = cv::getTextSize(std::to_string(truckup), intFontFace, dblFontScale, intFontThickness, 0);

	cv::Point ptTextBottomLeftPosition;
	cv::Point ptTextBottomRightPosition;

	ptTextBottomRightPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize1.width * 1.25);
	ptTextBottomRightPosition.y = imgFrame2Copy.rows;

	ptTextBottomLeftPosition.x = 0;
	ptTextBottomLeftPosition.y = imgFrame2Copy.rows;

	cv::putText(imgFrame2Copy, std::to_string(truckup), ptTextBottomRightPosition, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);

	cv::putText(imgFrame2Copy, std::to_string(truckdown), ptTextBottomLeftPosition, intFontFace, dblFontScale, SCALAR_RED, intFontThickness);

}










