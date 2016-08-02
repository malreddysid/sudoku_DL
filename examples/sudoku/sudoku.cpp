#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace caffe;

const int NUMRECT_DIM = 20;
const int NUMRECT_PIX = NUMRECT_DIM * NUMRECT_DIM;

#define UNASSIGNED 0
#define N 9

bool FindUnassignedLocation(int **grid, int &row, int &col);
bool isSafe(int **grid, int row, int col, int num);

bool solve(int **grid)
{
    char valueExists[9];
    
    // check rows
    for (int i = 0; i < 9; i++) {
        memset(valueExists, 0, 9 * sizeof(char));
        for (int j = 0; j < 9; j++) {
            int value = grid[i][j];
            if (!value) continue;
            char* exist = &valueExists[value - 1];
            if (*exist) {
                return false;
            }
            *exist = 1;
        }
    }
    // check cols
    for (int j = 0; j < 9; j++) {
        memset(valueExists, 0, 9 * sizeof(char));
        for (int i = 0; i < 9; i++) {
            int value = grid[i][j];
            if (!value) continue;
            char* exist = &valueExists[value - 1];
            if (*exist) {
                return false;
            }
            *exist = 1;
        }
    }
    // check blocks
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            memset(valueExists, 0, 9 * sizeof(char));
            for (int i = 3 * a; i < 3 * (a + 1); i++) {
                for (int j = 3 * b; j < 3 * (b + 1); j++) {
                    int value = grid[i][j];
                    if (!value) continue;
                    char* exist = &valueExists[value - 1];
                    if (*exist) {
                        return false;
                    }
                    *exist = 1;
                }
            }
        }
    }
    
    
    int row, col;
    if (!FindUnassignedLocation(grid, row, col))
        return true;
    for (int num = 1; num <= 9; num++)
    {
        if (isSafe(grid, row, col, num))
        {
            grid[row][col] = num;
            if (solve(grid))
                return true;
            grid[row][col] = UNASSIGNED;
        }
    }
    return false;
}

bool FindUnassignedLocation(int **grid, int &row, int &col)
{
    for (row = 0; row < N; row++)
        for (col = 0; col < N; col++)
            if (grid[row][col] == UNASSIGNED)
                return true;
    return false;
}

bool UsedInRow(int **grid, int row, int num)
{
    for (int col = 0; col < N; col++)
        if (grid[row][col] == num)
            return true;
    return false;
}

bool UsedInCol(int **grid, int col, int num)
{
    for (int row = 0; row < N; row++)
        if (grid[row][col] == num)
            return true;
    return false;
}

bool UsedInBox(int **grid, int boxStartRow, int boxStartCol, int num)
{
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            if (grid[row + boxStartRow][col + boxStartCol] == num)
                return true;
    return false;
}

bool isSafe(int **grid, int row, int col, int num)
{
    return !UsedInRow(grid, row, num) && !UsedInCol(grid, col, num) &&
    !UsedInBox(grid, row - row % 3, col - col % 3, num);
}

void drawLine(Vec2f line, Mat &img, Scalar rgb = CV_RGB(0,0,255))
{
    if(line[1]!=0)
    {
        float m = -1/tan(line[1]);
        
        float c = line[0]/sin(line[1]);
        
        cv::line(img, Point(0, c), Point(img.size().width, m*img.size().width+c), rgb);
    }
    else
    {
        cv::line(img, Point(line[0], 0), Point(line[0], img.size().height), rgb);
    }
    
}

void mergeRelatedLines(vector<Vec2f> *lines, Mat &img)
{
    vector<Vec2f>::iterator current;
    for(current=lines->begin();current!=lines->end();current++)
    {
        if((*current)[0]==0 && (*current)[1]==-100) continue;
        float p1 = (*current)[0];
        float theta1 = (*current)[1];
        Point pt1current, pt2current;
        if(theta1>CV_PI*45/180 && theta1<CV_PI*135/180)
        {
            pt1current.x=0;
            
            pt1current.y = p1/sin(theta1);
            
            pt2current.x=img.size().width;
            pt2current.y=-pt2current.x/tan(theta1) + p1/sin(theta1);
        }
        else
        {
            pt1current.y=0;
            
            pt1current.x=p1/cos(theta1);
            
            pt2current.y=img.size().height;
            pt2current.x=-pt2current.y/tan(theta1) + p1/cos(theta1);
            
        }
        vector<Vec2f>::iterator    pos;
        for(pos=lines->begin();pos!=lines->end();pos++)
        {
            if(*current==*pos) continue;
            if(fabs((*pos)[0]-(*current)[0])<20 && fabs((*pos)[1]-(*current)[1])<CV_PI*10/180)
            {
                float p = (*pos)[0];
                float theta = (*pos)[1];
                Point pt1, pt2;
                if((*pos)[1]>CV_PI*45/180 && (*pos)[1]<CV_PI*135/180)
                {
                    pt1.x=0;
                    pt1.y = p/sin(theta);
                    pt2.x=img.size().width;
                    pt2.y=-pt2.x/tan(theta) + p/sin(theta);
                }
                else
                {
                    pt1.y=0;
                    pt1.x=p/cos(theta);
                    pt2.y=img.size().height;
                    pt2.x=-pt2.y/tan(theta) + p/cos(theta);
                }
                if(((double)(pt1.x-pt1current.x)*(pt1.x-pt1current.x) + (pt1.y-pt1current.y)*(pt1.y-pt1current.y)<64*64) &&
                   ((double)(pt2.x-pt2current.x)*(pt2.x-pt2current.x) + (pt2.y-pt2current.y)*(pt2.y-pt2current.y)<64*64))
                {
                    // Merge the two
                    (*current)[0] = ((*current)[0]+(*pos)[0])/2;
                    
                    (*current)[1] = ((*current)[1]+(*pos)[1])/2;
                    
                    (*pos)[0]=0;
                    (*pos)[1]=-100;
                }
            }
        }
    }
}

void getImage(char *image_file, int **puzzle)
{
    Mat image = imread(image_file, CV_LOAD_IMAGE_GRAYSCALE);
    GaussianBlur(image, image, Size(5,5), 0);
    Mat thresholded_image;
    adaptiveThreshold(image, thresholded_image, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 5);
    bitwise_not(thresholded_image, thresholded_image);
    Mat dilated_image;
    Mat element = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
    dilate(thresholded_image, dilated_image, element);
    
    // Use floodfill to idenify the borders
    int max = -1;
    Point maxPt;
    for(int i = 0; i < dilated_image.size().height; i++)
    {
        uchar *row = dilated_image.ptr(i);
        for(int j = 0; j < dilated_image.size().width; j++)
        {
            if(row[j] >= 128)
            {
                int area = floodFill(dilated_image, Point(j, i), CV_RGB(0,0,64));
                if(area > max)
                {
                    max = area;
                    maxPt = Point(j,i);
                }
            }
        }
    }
    
    floodFill(dilated_image, maxPt, CV_RGB(255,255,255));
    
    for(int i = 0; i < dilated_image.size().height; i++)
    {
        uchar *row = dilated_image.ptr(i);
        for(int j = 0; j < dilated_image.size().width; j++)
        {
            if(row[j] == 64 && i != maxPt.x && j != maxPt.y)
            {
                floodFill(dilated_image, Point(j,i), CV_RGB(0,0,0));
            }
        }
    }
    
    Mat eroded_image, lined_image;
    erode(dilated_image, eroded_image, element);
    erode(dilated_image, lined_image, element);
    
    // Detect the lines
    vector<Vec2f> lines;
    HoughLines(eroded_image, lines, 1, CV_PI/180, 200);
    mergeRelatedLines(&lines, lined_image); // Add this line
    for(int i=0;i<lines.size();i++)
    {
        drawLine(lines[i], eroded_image, CV_RGB(0,0,128));
    }
    
    
    // Detecting the extremes
    Vec2f topEdge = Vec2f(1000,1000); 
    Vec2f bottomEdge = Vec2f(-1000,-1000); 
    Vec2f leftEdge = Vec2f(1000,1000);    double leftXIntercept=100000;
    Vec2f rightEdge = Vec2f(-1000,-1000);        double rightXIntercept=0;
    for(int i=0;i<lines.size();i++)
    {
        
        Vec2f current = lines[i];
        
        float p=current[0];
        
        float theta=current[1];
        
        if(p==0 && theta==-100)
            continue;
        double xIntercept;
        xIntercept = p/cos(theta);
        if(theta>CV_PI*80/180 && theta<CV_PI*100/180)
        {
            if(p<topEdge[0])
                
                topEdge = current;
            
            if(p>bottomEdge[0])
                bottomEdge = current;
        }
        else if(theta<CV_PI*10/180 || theta>CV_PI*170/180)
        {
            if(xIntercept>rightXIntercept)
            {
                rightEdge = current;
                rightXIntercept = xIntercept;
            }
            else if(xIntercept<=leftXIntercept)
            {
                leftEdge = current;
                leftXIntercept = xIntercept;
            }
        }
    }
    drawLine(topEdge, lined_image, CV_RGB(0,0,0));
    drawLine(bottomEdge, lined_image, CV_RGB(0,0,0));
    drawLine(leftEdge, lined_image, CV_RGB(0,0,0));
    drawLine(rightEdge, lined_image, CV_RGB(0,0,0));
    
    Point left1, left2, right1, right2, bottom1, bottom2, top1, top2;
    
    int height=lined_image.size().height;
    
    int width=lined_image.size().width;
    
    if(leftEdge[1]!=0)
    {
        left1.x=0;        left1.y=leftEdge[0]/sin(leftEdge[1]);
        left2.x=width;    left2.y=-left2.x/tan(leftEdge[1]) + left1.y;
    }
    else
    {
        left1.y=0;        left1.x=leftEdge[0]/cos(leftEdge[1]);
        left2.y=height;    left2.x=left1.x - height*tan(leftEdge[1]);
        
    }
    
    if(rightEdge[1]!=0)
    {
        right1.x=0;        right1.y=rightEdge[0]/sin(rightEdge[1]);
        right2.x=width;    right2.y=-right2.x/tan(rightEdge[1]) + right1.y;
    }
    else
    {
        right1.y=0;        right1.x=rightEdge[0]/cos(rightEdge[1]);
        right2.y=height;    right2.x=right1.x - height*tan(rightEdge[1]);
        
    }
    
    bottom1.x=0;    bottom1.y=bottomEdge[0]/sin(bottomEdge[1]);
    
    bottom2.x=width;bottom2.y=-bottom2.x/tan(bottomEdge[1]) + bottom1.y;
    
    top1.x=0;        top1.y=topEdge[0]/sin(topEdge[1]);
    top2.x=width;    top2.y=-top2.x/tan(topEdge[1]) + top1.y;
    
    // Next, we find the intersection of  these four lines
    double leftA = left2.y-left1.y;
    double leftB = left1.x-left2.x;
    
    double leftC = leftA*left1.x + leftB*left1.y;
    
    double rightA = right2.y-right1.y;
    double rightB = right1.x-right2.x;
    
    double rightC = rightA*right1.x + rightB*right1.y;
    
    double topA = top2.y-top1.y;
    double topB = top1.x-top2.x;
    
    double topC = topA*top1.x + topB*top1.y;
    
    double bottomA = bottom2.y-bottom1.y;
    double bottomB = bottom1.x-bottom2.x;
    
    double bottomC = bottomA*bottom1.x + bottomB*bottom1.y;
    
    // Intersection of left and top
    double detTopLeft = leftA*topB - leftB*topA;
    
    CvPoint ptTopLeft = cvPoint((topB*leftC - leftB*topC)/detTopLeft, (leftA*topC - topA*leftC)/detTopLeft);
    
    // Intersection of top and right
    double detTopRight = rightA*topB - rightB*topA;
    
    CvPoint ptTopRight = cvPoint((topB*rightC-rightB*topC)/detTopRight, (rightA*topC-topA*rightC)/detTopRight);
    
    // Intersection of right and bottom
    double detBottomRight = rightA*bottomB - rightB*bottomA;
    CvPoint ptBottomRight = cvPoint((bottomB*rightC-rightB*bottomC)/detBottomRight, (rightA*bottomC-bottomA*rightC)/detBottomRight);// Intersection of bottom and left
    double detBottomLeft = leftA*bottomB-leftB*bottomA;
    CvPoint ptBottomLeft = cvPoint((bottomB*leftC-leftB*bottomC)/detBottomLeft, (leftA*bottomC-bottomA*leftC)/detBottomLeft);
    
    int maxLength = (ptBottomLeft.x-ptBottomRight.x)*(ptBottomLeft.x-ptBottomRight.x) + (ptBottomLeft.y-ptBottomRight.y)*(ptBottomLeft.y-ptBottomRight.y);
    int temp = (ptTopRight.x-ptBottomRight.x)*(ptTopRight.x-ptBottomRight.x) + (ptTopRight.y-ptBottomRight.y)*(ptTopRight.y-ptBottomRight.y);
    
    if(temp>maxLength) maxLength = temp;
    
    temp = (ptTopRight.x-ptTopLeft.x)*(ptTopRight.x-ptTopLeft.x) + (ptTopRight.y-ptTopLeft.y)*(ptTopRight.y-ptTopLeft.y);
    
    if(temp>maxLength) maxLength = temp;
    
    temp = (ptBottomLeft.x-ptTopLeft.x)*(ptBottomLeft.x-ptTopLeft.x) + (ptBottomLeft.y-ptTopLeft.y)*(ptBottomLeft.y-ptTopLeft.y);
    
    if(temp>maxLength) maxLength = temp;
    
    maxLength = sqrt((double)maxLength);
    
    Point2f src[4], dst[4];
    src[0] = ptTopLeft;            dst[0] = Point2f(0,0);
    src[1] = ptTopRight;        dst[1] = Point2f(maxLength-1, 0);
    src[2] = ptBottomRight;        dst[2] = Point2f(maxLength-1, maxLength-1);
    src[3] = ptBottomLeft;        dst[3] = Point2f(0, maxLength-1);
    
    // Warp the image
    
    Mat undistorted = Mat(Size(maxLength, maxLength), CV_8UC1);
    cv::warpPerspective(image, undistorted, cv::getPerspectiveTransform(src, dst), Size(maxLength, maxLength));
    
    // Now repeat the floodfill to identify and remove the borderes. Then we are left with only numbers. It hepls with the accuracy.
    Mat undistortedThreshed = undistorted.clone();
    adaptiveThreshold(undistorted, undistortedThreshed, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 101, 1);
    erode(undistortedThreshed, undistortedThreshed, element);
    
    
    dilate(undistortedThreshed, dilated_image, element);
    
    max = -1;
    for(int i = 0; i < dilated_image.size().height; i++)
    {
        uchar *row = dilated_image.ptr(i);
        for(int j = 0; j < dilated_image.size().width; j++)
        {
            if(row[j] >= 128)
            {
                int area = floodFill(dilated_image, Point(j, i), CV_RGB(0,0,64));
                if(area > max)
                {
                    max = area;
                    maxPt = Point(j,i);
                }
            }
        }
    }
    
    floodFill(dilated_image, maxPt, CV_RGB(255,255,255));
    
    for(int i = 0; i < dilated_image.size().height; i++)
    {
        uchar *row = dilated_image.ptr(i);
        for(int j = 0; j < dilated_image.size().width; j++)
        {
            if(row[j] == 64 && i != maxPt.x && j != maxPt.y)
            {
                floodFill(dilated_image, Point(j,i), CV_RGB(0,0,0));
            }
        }
    }
    
    erode(dilated_image, eroded_image, element);
    
    undistortedThreshed = undistortedThreshed - eroded_image;
    erode(undistortedThreshed, undistortedThreshed, element);
    //dilate(undistortedThreshed, undistortedThreshed, element);
    
    Mat resized = Mat(Size((undistortedThreshed.cols/9)*9, (undistortedThreshed.rows/9)*9), CV_8UC1);
    cv::resize(undistortedThreshed, resized, Size((undistortedThreshed.cols/9)*9, (undistortedThreshed.rows/9)*9));

    // Setting CPU or GPU to use Caffe
    {
        LOG(ERROR) << "Using CPU";
        Caffe::set_mode(Caffe::CPU);
    }
            
    // Get the net
    Net<float> caffe_test_net("models/sudoku/deploy.prototxt", caffe::TEST);
    // Get trained net
    caffe_test_net.CopyTrainedLayersFrom("models/sudoku/sudoku_iter_10000.caffemodel");

    // Split the image into 81 parts, so as to identify the numbers.
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            puzzle[i][j] = 0;
            Mat cell = Mat(Size(resized.cols/9, resized.rows/9), CV_8UC1);
            for(int ii = 0; ii < cell.rows; ii++)
            {
                for(int jj = 0; jj < cell.cols; jj++)
                {
                    cell.data[ii*cell.cols+jj] = resized.data[i*cell.rows*resized.cols + ii*resized.cols + jj + j*cell.cols];
                }
            }
            
           int area = countNonZero(cell);
            if(area < cell.rows * cell.cols / 24)
            {
                continue;
            }
            
            // Save the cell as an image to make an inference later using Caffe
            cv::imwrite("examples/sudoku/cell.jpg", cell);
            
            // Get datum
            Datum datum;
            if (!ReadImageToDatum("examples/sudoku/cell.jpg", 1, 28, 28, false, &datum)) {
                LOG(ERROR) << "Error during file reading";
            }
            
            
            // Get the blob
            Blob<float>* blob = new Blob<float>(1, datum.channels(), datum.height(), datum.width());
            
            // Get the blobproto
            BlobProto blob_proto;
            blob_proto.set_num(1);
            blob_proto.set_channels(datum.channels());
            blob_proto.set_height(datum.height());
            blob_proto.set_width(datum.width());
            int size_in_datum = std::max<int>(datum.data().size(),
                                              datum.float_data_size());
            
            for (int ii = 0; ii < size_in_datum; ++ii) {
                blob_proto.add_data(0.);
            }
            const string& data = datum.data();
            if (data.size() != 0) {
                for (int ii = 0; ii < size_in_datum; ++ii) {
                    blob_proto.set_data(ii, blob_proto.data(ii) + (uint8_t)data[ii]);
                }
            }
            
            // Set data into blob
            blob->FromProto(blob_proto);
            
            // Fill the vector
            vector<Blob<float>*> bottom;
            bottom.push_back(blob);
            float type = 0.0;
            
            const vector<Blob<float>*>& result =  caffe_test_net.Forward(bottom, &type);
            
            // Here I can use the argmax layer, but for now I do a simple for loop
            float max = 0;
            float max_i = 0;
            for (int ii = 0; ii < 10; ++ii) {
                float value = result[0]->cpu_data()[ii];
                if (max < value){
                    max = value;
                    max_i = ii;
                }
            }
            
            // Condition to detemine if the cell contains a digit or not.
            if(max > 0.95)
            	puzzle[i][j] = max_i;
        }
    }
    return;
}


int main()
{
    char image[] = "examples/sudoku/samples/sudoku1.jpg";
    int **sudoku = (int **) malloc(sizeof(int *) * 9);
    for(int i = 0; i < 9; i++)
    {
        sudoku[i] = (int *) malloc(sizeof(int) * 9);
    }
    
    getImage(image, sudoku);
    // Print puzzle
    
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            if(sudoku[i][j] == 0)
                printf("   |");
            else
                printf(" %d |", sudoku[i][j]);
        }
        printf("\n");
    }
    
    int possible = solve(sudoku);
    
    if(possible)
    {
        printf("\nSolved sudoku:\n");
        for(int i = 0; i < 9; i++)
        {
            for(int j = 0; j < 9; j++)
            {
                if(sudoku[i][j] == 0)
                    printf("   |");
                else
                    printf(" %d |", sudoku[i][j]);
            }
            printf("\n");
        }
    }
    else
    {
        printf("Not possible\n");
    }
    return 0;
}

