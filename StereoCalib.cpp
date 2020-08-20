//
// Created by syk on 17-10-23.
//

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "tinydir.h"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

using namespace std;
using namespace cv;

static void StereoCalib(const vector<string>& imagelist, Size boardSize, bool useCalibrated=true, bool showRectified=true)
{
    if(imagelist.size()%2 != 0)
    {
        cout << "Error: the image list contains odd(non-even) number of elements\n";
        return;
    }
    bool displayCorners = false;
    const int maxScale = 2;
    const float squareSize = 50.f;   //Set this to your actual square size

    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;
    Size imageSize;

    int i, j, k, nimages = (int)imagelist.size()/2;

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<string> goodImageList;

//    sykfix: Get the current path, but is redundancy
    char current_absolute_path[200];
    if (NULL == getcwd(current_absolute_path, 200))
    {
        printf("***Error***");
        exit(-1);
    }


    for(i = j = 0; i < nimages; i++)
    {
        for(k = 0; k < 2; k++)
        {
            //sykdebug: k is the flag to tell whether success, k == 2
            const string& filename = imagelist[i*2 + k];//string(current_absolute_path)+
            Mat img = imread(filename, 0);
            //cout<<filename<<endl;
            if(img.empty())
                break;
            if(imageSize == Size())
                imageSize = img.size();
            else if(img.size() != imageSize)
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            vector<Point2f>& corners = imagePoints[k][j];
            for(int scale = 1; scale <= maxScale; scale++)
            {
                Mat timg;
                if(scale == 1)
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale);
                found = findChessboardCorners(timg, boardSize, corners,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
                if(found)
                {
                    if(scale > 1)       //sykdebug: for debug? add imshow?
                    {
                        Mat cornerMat(corners);
                        cornerMat *= 1./scale;
                    }
                    break;
                }
            }
            if(!found)
                break;
            cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 0.01));
            if(displayCorners)
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
                double sf = 640./MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf);
                imshow("corners", cimg1);
                char c = (char)waitKey(500);
                if(c == 27 || c == 'q' || c == 'Q')
                    exit(-1);
            }
            else
                putchar('.');

        }
        if(k == 2)
        {
            goodImageList.push_back(imagelist[i*2]);
            goodImageList.push_back(imagelist[i*2+1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if(nimages < 2)
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }
    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);
    for(i = 0; i < nimages; i++)
    {
        for(j = 0; j < boardSize.height; j++)
            for(k = 0; k < boardSize.width; k++)
                objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
    }

    cout << "Running stereo calibration ...\n";
    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;

    //Singe View calibration
    Mat cameraMatrix_single =Mat::eye(3,3,CV_64F);
    vector<vector<Point3f> > objectPoints_single;
    objectPoints_single.resize(nimages);
    Mat distCoeffs_single;
    Mat rvecs, tvecs,stdDeviationsIntrinsics,stdDeviationsExtrinsics,perViewErrors;
    vector<vector<Point2f> > imagePoints_single=imagePoints[0];
    for(i = 0; i < nimages; i++)
    {
        for(j = 0; j < boardSize.height; j++)
            for(k = 0; k < boardSize.width; k++)
                objectPoints_single[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
    }

    calibrateCamera(objectPoints_single,imagePoints_single,imageSize,cameraMatrix_single,
            distCoeffs_single,rvecs, tvecs,stdDeviationsIntrinsics,stdDeviationsExtrinsics,perViewErrors);
    /*
     * CV_EXPORTS_AS(calibrateCameraExtended) double calibrateCamera( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     OutputArray stdDeviationsIntrinsics,
                                     OutputArray stdDeviationsExtrinsics,
                                     OutputArray perViewErrors,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) );
     * */
    //Core function

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
        cameraMatrix[0], distCoeffs[0],
        cameraMatrix[1], distCoeffs[1],
        imageSize, R, T, E, F,

        CV_CALIB_ZERO_TANGENT_DIST +
        CV_CALIB_RATIONAL_MODEL +
        CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5,//CV_CALIB_FIX_K3 +//CV_CALIB_SAME_FOCAL_LENGTH +//        CV_CALIB_FIX_ASPECT_RATIO +
        TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5)
    );
    //T.at<double>(2,0)=50;
    cout<<T.at<double>(2,0)<<endl;
    cout << "done with RMS error=" << rms << endl;
    cout << "Stereo Ex parameter" << rms << endl;
    cout<<"\nR"<< R<<"\nT"<<T<<endl;
    cout << "In parameter compare"  << endl;
    cout << "\nStereo_Left" << endl;
    cout<< cameraMatrix[0]<<endl;
    cout<< distCoeffs[0]<<endl;
    cout << "\nStereo_Right" << endl;
    cout<< cameraMatrix[1]<<endl;
    cout<< distCoeffs[1]<<endl;



    /*
     o objectPoints– 校正的图像点向量组.
o imagePoints1–通过第一台相机观测到的图像上面的向量组.
o imagePoints2–通过第二台相机观测到的图像上面的向量组.
npoints--角点数
o cameraMatrix1– 输入或者输出第一个相机的内参数矩阵
o distCoeffs1– 输入/输出第一个相机的畸变系数向量
o cameraMatrix2– 输入或者输出第二个相机的内参数矩阵
o distCoeffs2– 输入/输出第二个相机的畸变系数向量
o imageSize– 图像文件的大小——只用于初始化相机内参数矩阵。
o R– 输出第一和第二相机坐标系之间的旋转矩阵。
o T– 输出第一和第二相机坐标系之间的旋转矩阵平移向量
o E–输出本征矩阵。
o F–输出基础矩阵。
o term_crit– 迭代优化算法终止的标准。
o flags–
不同的FLAG,可能是零或以下值的结合:
§ CV_CALIB_FIX_INTRINSIC要确认cameraMatrix? and distCoeffs?所以只有R, T, E , 和F矩阵被估计出来
§ CV_CALIB_USE_INTRINSIC_GUESS根据指定的FLAG优化一些或全部的内在参数。初始值是由用户提供。
§ CV_CALIB_FIX_PRINCIPAL_POINT在优化过程中确定主点。
§ CV_CALIB_FIX_FOCAL_LENGTH确定和 .
§ CV_CALIB_FIX_ASPECT_RATIO优化 . 确定的比值.
§ CV_CALIB_SAME_FOCAL_LENGTH执行以及 .
§ CV_CALIB_ZERO_TANGENT_DIST设置每个相机切向畸变系数为零并且设为固定值。
§ CV_CALIB_FIX_K1,...,CV_CALIB_FIX_K6在优化中不改变相应的径向畸变系数. 如果设置CV_CALIB_USE_INTRINSIC_GUESS , 使用distCoeffs矩阵提供的系数。否则将其置零.
§ CV_CALIB_RATIONAL_MODEL能够输出系数k4，k5，k6。提供向后兼容性,这额外FLAG应该明确指定校正函数使用理性模型和返回8个系数。如果FLAG没有被设置,该函数计算并只返回5畸变系数。
     */
    /*
     * CALIBRATION QUALITY CHECK
     * because the output fundermental matrix implicitly
     * includes all the output information,
     * we can check the quality of calibration using the
     * epipolar geometry constraint: m2^t*F*m1=0
     * */
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for(i = 0; i < nimages; i++)
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for(k = 0; k < 2; k++)
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            // undistort the corner point
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for(j = 0; j < npt; j++)
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                imagePoints[0][i][j].y*lines[1][j][1] +lines[1][j][2]) +
                fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average reprojection err = " << err / npoints << endl;

    //save intrinsic parameters
    FileStorage fs("intrinsics.yml", CV_STORAGE_WRITE);
    if (fs.isOpened())
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
           "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";
    // OpenCv can handle left-right
    // or up-down camer arrangements

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    //Core Function
    Size myimageSiz= imageSize;
    myimageSiz.height=myimageSiz.height*0.1;
    myimageSiz.width=myimageSiz.width*0.1;
    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]
    );


/*
o cameraMatrix1– 第一个相机矩阵.
o distCoeffs1– 第一个相机畸变参数.
o cameraMatrix2– 第二个相机矩阵.
o distCoeffs2– 第二个相机畸变参数.
o imageSize– 用于校正的图像大小.
o R– 第一和第二相机坐标系之间的旋转矩阵。
o T– 第一和第二相机坐标系之间的平移矩阵.
 以上是输入
o R1– 输出第一个相机的3x3矫正变换(旋转矩阵) .
o R2– 输出第二个相机的3x3矫正变换(旋转矩阵) .
o P1–在第一台相机的新的坐标系统(矫正过的)输出 3x4 的投影矩阵
o P2–在第二台相机的新的坐标系统(矫正过的)输出 3x4 的投影矩阵
o Q–输出深度视差映射矩阵，如下：
矩阵Q是一个任意提供的矩阵(比如, stereoRectify()所能得出的矩阵).
o flags– 操作的 flag可以是零或者是CV_CALIB_ZERO_DISPARITY . 如果设置了CV_CALIB_ZERO_DISPARITY,函数的作用是使每个相机的主点在校正后的图像上有相同的像素坐标。如果未设置标志，功能还可以改变图像在水平或垂直方向（取决于极线的方向）来最大化有用的图像区域。
o alpha– 自由缩放参数。如果是-1或没有，该函数执行默认缩放。否则，该参数应在0和1之间。alpha=0，校正后的图像进行缩放和偏移，只有有效像素是可见的（校正后没有黑色区域）。alpha= 1意味着校正图像的抽取和转移，所有相机原始图像素像保留在校正后的图像（源图像像素没有丢失）。显然，任何中间值产生这两种极端情况之间的中间结果。
o newImageSize– 校正后新的图像分辨率。相同的尺寸应传递给initUndistortRectifyMap()（见OpenCV样品目录stereo_calib.cpp样品）。当（0,0）传递（默认），它设置为原始图像大小。设置为较大的值能帮助你保存原始图像的细节，特别是当有一个大的径向畸变时。
o validPixROI1– 校正后的图像可选的输出矩形，里面所有像素都是有效的。如果alpha= 0，ROIs覆盖整个图像。否则，他们可能会比较小。
o validPixROI2– 校正后的图像可选的输出矩形，里面所有像素都是有效的。如果alpha= 0，ROIs覆盖整个图像。否则，他们可能会比较小。
 */
    fs.open("extrinsics.yml", CV_STORAGE_WRITE);
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

    if (!showRectified)
        return;
    Mat rmap[2][2];
    // if by calibrated(bouguet's method)
    if (useCalibrated)
    {
        // already computed everything
    }
        // or else hartley's method
    else
        // use intrinsic parameters of each camera, but
        // compute the rectification transformation directly
        // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for (k = 0; k < 2; k++)
        {
            for (i = 0; i < nimages; i++)
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);
        R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }
    // Precompute maps for cv::remap()
    cout<<distCoeffs[0]<<endl;
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
    Mat canvas;
    double sf;
    int w, h;
    if (!isVerticalStereo)
    {
        sf = 600. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w * 2, CV_8UC3);
    }
    else
    {
        sf = 300. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h * 2, w, CV_8UC3);
    }


    for (i = 0; i < nimages; i++) {
        for (k = 0; k < 2; k++) {
            Mat img = imread(goodImageList[i * 2 + k], 0), rimg, cimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
            //imshow("单目相机校正结果", rimg);
            //waitKey();
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            if(k==0)
                imwrite("./left/"+to_string(i)+".png",cimg);
            else
                imwrite("./right/"+to_string(i)+".png",cimg);
        }
    }

    for (i = 0; i < nimages; i++)
    {
        for (k = 0; k < 2; k++)
        {
            Mat img = imread(goodImageList[i * 2 + k], 0), rimg, cimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
            imshow("单目相机校正结果", rimg);
            waitKey();
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
            if (useCalibrated)
            {
                Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                          cvRound(validRoi[k].width*sf), cvRound(validRoi[j].height*sf));
                rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
            }
        }
        if (!isVerticalStereo)
            for (j = 0; j < canvas.rows; j += 16)
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for (j = 0; j < canvas.cols; j += 16)
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("双目校正结果", canvas);
        waitKey();
        char c = (char)waitKey();
        if (c == 27 || c == 'q' || c == 'Q')
            break;
    }
}

void StereoCut(    vector<string> &leftimagelistinput, vector<string> &rightimagelistinput, Rect &rect){

    for (int i=0;i<leftimagelistinput.size();i++){
        Mat tmp=imread(leftimagelistinput[i],cv::IMREAD_UNCHANGED);
        Mat temp(tmp, rect);
        imwrite("./left_rect/"+to_string(i)+".png",temp);
    }

    for (int i=0;i<rightimagelistinput.size();i++){
        Mat tmp=imread(rightimagelistinput[i],cv::IMREAD_UNCHANGED);
        Mat temp(tmp, rect);
        imwrite("./right_rect/"+to_string(i)+".png",temp);
    }





}





static bool readStringList(const string& filename, vector<string>& l)
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if(n.type() != FileNode::SEQ)
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it)
        l.push_back((string)*it);
    return true;
}

//This function only works in clion, and should define the work dirctory in clion first
int getImagelist(vector<string> &imagelist, string Prefix)
{
    imagelist.clear();
    //Get current path
//    char *current_absolute_path = "/home/syk/CLionProjects/BinocularCalibration";
    char current_absolute_path[200];
    if (NULL == getcwd(current_absolute_path, 200))
    {
        printf("***Error***");
        exit(-1);
    }

    tinydir_dir dir;
    size_t i;
    if (tinydir_open_sorted(&dir, TINYDIR_STRING(current_absolute_path)) == -1)
    {
        perror("Error opening file");
        tinydir_close(&dir);
    }
    for (i = 0; i < dir.n_files; i++)
    {
        tinydir_file file;
        if (tinydir_readfile_n(&dir, &file, i) == -1)
        {
            perror("Error getting file");
            return 0;
        }
        if (file.is_dir && file.name == Prefix)
        {
            if (tinydir_open_subdir_n(&dir, i) == -1)
            {
                perror("Error opening subdirectory");
                return 0;
            }
            break;
        }
    }
    for (i = 0; i < dir.n_files; i++)
    {
        tinydir_file file;
        if (tinydir_readfile_n(&dir, &file, i) == -1)
        {
            perror("Error getting file");
            return 0;
        }
        if(string(file.name) != "." && string(file.name) != "..")
            imagelist.push_back(string(current_absolute_path) + "/" + Prefix + "/" + (file.name));
    }
    return 1;
}

#define TEST2

#ifdef TEST1

int main(int argc, char** argv)
{
    Size boardSize;
    string imagelistfn;
    bool showRectified = true;
    for(int i = 1; i < argc; i++)
    {
        if(string(argv[i]) == "-w")
        {
            if(sscanf(argv[++i], "%d", &boardSize.width) != 1 || boardSize.width <= 0)
            {
                cout << "invalid board width" << endl;
                return -1;
            }
        }
        else if(string(argv[i]) == "-h")
        {
            if(sscanf(argv[++i], "%d", &boardSize.height) != 1 || boardSize.height <= 0)
            {
                cout << "invalid board height" << endl;
                return -1;
            }
        }
        else if(string(argv[i]) == "-nr")
            showRectified = false;
        else if(string(argv[i]) == "--help")
            return -1;
        else if(argv[i][0] = '-')
        {
            cout << "invalid option " << argv[i] << endl;
            return 0;
        }
        else
            imagelistfn = argv[i];
    }
    if(imagelistfn == "")
    {
        imagelistfn = "stereo_calib.xml";
        boardSize = Size(9, 6);
    }
    else if(boardSize.width <= 0 || boardSize.height <= 0)
    {
        cout << "if you specified XML file with chessboards, you should also specify the board width and height" << endl;
        return 0;
    }
    vector<string> imagelist;
    bool ok = readStringList(imagelistfn, imagelist);
    if(!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
        return -1;
    }
    StereoCalib(imagelist, boardSize, true, showRectified);
    return 0;
}

#elif defined TEST2
int main()
{
    Size boardSize =  Size(9, 8);//Size(9, 8);
    bool showRectified = true;
    string leftprefix = "leftimages", rightprefix = "rightimages";
    string leftinput = "left", rightinput = "right";
    if(boardSize.width <= 0 || boardSize.height <= 0)
    {
        cout << "if you specified XML file with chessboards, you should also specify the board width and height" << endl;
        return 0;
    }
    vector<string> leftimagelist;
    vector<string> rightimagelist;
    vector<string> leftimagelistinput;
    vector<string> rightimagelistinput;
    vector<string> imagelist;
    getImagelist(rightimagelistinput, rightinput);
    getImagelist(leftimagelistinput, leftinput);
    getImagelist(rightimagelist, rightprefix);
    getImagelist(leftimagelist, leftprefix);




//    mixing two imagelist into one imagelist :<left, right, left, right...>
    if(leftimagelist.size() != rightimagelist.size())
    {
        cerr << "left images number don't equal to right images number" << endl;
        return 0;
    }
    else
    {
        for(int i = 0; i < leftimagelist.size(); i++)
        {
            imagelist.push_back(leftimagelist[i]);
            imagelist.push_back(rightimagelist[i]);
        }

    }
    StereoCalib(imagelist, boardSize, true, showRectified);
    Rect rect(310,246,970-310, 734-246); //(left up, right up, left down, right down )
    StereoCut(leftimagelistinput, rightimagelistinput, rect);
    return 0;
}

#endif
