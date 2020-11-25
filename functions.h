//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
#include "CserialCommunication.h"
#define VIDEOSTREAM  "/dev/video0" // "/home/pi/CampusCompetition/live1.avi" //   
//#define VIDEOMODEON 
//#define DEBUGMODE 

// Morphological filter Reference: https://www.cnblogs.com/wangxinyu0628/p/5928824.html
void Homography(cv::Mat image, cv::Mat &Opened , int thresh) //mask
{
    cv::Mat element_9(thresh, thresh, CV_8U, cv::Scalar(1));
    cv::morphologyEx(image, Opened, cv::MORPH_OPEN, element_9);
}

// Hole fill Reference: https://blog.csdn.net/hust_bochu_xuchao/article/details/51967846
void fillHole(const cv::Mat srcimage, cv::Mat &dstimage)
{
	cv::Size m_Size = srcimage.size();
	cv::Mat temimage = cv::Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcimage.type());
	srcimage.copyTo(temimage(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)));
	cv::floodFill(temimage, cv::Point(0, 0), cv::Scalar(255));
	cv::Mat cutImg;
	temimage(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)).copyTo(cutImg);
	dstimage = srcimage | (~cutImg);
}

// find contours with bgr image
void find_rec_contours_bgr(cv::Mat image, cv::Mat & imgBinary, int color[3], std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Vec4i> &hierarchy) {
    cv::Mat bgr = image.clone();
    cv::Mat imgThresholded;
    cv::inRange(bgr,cv::Scalar(color[0]-127,color[1]-127,color[2]-127),cv::Scalar(color[0],color[1],color[2]),imgThresholded);   
    cv::threshold(imgThresholded,imgThresholded, 120,255, cv::THRESH_BINARY);
    Homography(imgThresholded,imgBinary,20);
    cv::findContours(imgBinary, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
}

// find contours with hsv image
void find_rec_contours(cv::Mat image, cv::Mat & imgBinary, int color[3], std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Vec4i> &hierarchy) {
    cv::Mat bgr = image.clone();
    cv::Mat imgThresholded;
    cv::Mat red_imgThresholded_low;
    cv::Mat red_imgThresholded_high;
    int s_low = 75;
    int h_low = 150;
    int s_high = 255;
    int v_high = 255;
    cv::cvtColor(bgr,imgThresholded,CV_BGR2HSV);
    if (color[0] == 255){cv::inRange(imgThresholded,cv::Scalar(100,s_low,220),cv::Scalar(124,s_high,v_high),imgThresholded);
    fillHole(imgThresholded,imgThresholded);
    Homography(imgThresholded,imgBinary,10);
    #ifdef VIDEOMODEON
    cv::imshow("Filter Blue Img", imgBinary); //blue
    #endif
    }
    if (color[1] == 255){cv::inRange(imgThresholded,cv::Scalar(35,s_low,h_low),cv::Scalar(77,s_high,v_high),imgThresholded);
    fillHole(imgThresholded,imgThresholded);
    Homography(imgThresholded,imgBinary,10);
    #ifdef VIDEOMODEON
    cv::imshow("Filter Greeen Img", imgBinary); //green
    #endif
    }
    if (color[2] == 255){
        cv::inRange(imgThresholded,cv::Scalar(0,s_low,h_low),cv::Scalar(10,s_high,v_high),red_imgThresholded_low); //nightversion
        cv::inRange(imgThresholded,cv::Scalar(165,s_low,h_low),cv::Scalar(180,s_high,v_high),red_imgThresholded_high); //nightversion
        // cv::inRange(imgThresholded,cv::Scalar(0,75,h_low),cv::Scalar(10,190,v_high),red_imgThresholded_low); //daytime version
        // cv::inRange(imgThresholded,cv::Scalar(165,75,h_low),cv::Scalar(180,190,v_high),red_imgThresholded_high); //daytime version
        cv::addWeighted(red_imgThresholded_low,1.0,red_imgThresholded_high,1.0,0.0,imgThresholded);
        fillHole(imgThresholded,imgThresholded);
        Homography(imgThresholded,imgBinary,10);
        #ifdef VIDEOMODEON
        cv::imshow("Filter Red Img", imgBinary); //red
        #endif
    }
    
    cv::threshold(imgThresholded,imgThresholded, 120,255, cv::THRESH_BINARY);
    
    cv::findContours(imgBinary, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
}

void get_center_points(std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point2f> &mc , std::vector<float> &area_size)
{
    std::vector<cv::Moments> mu(contours.size());

    for (int i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i], false);
    }
    for (int i = 0; i < contours.size(); i++)
    {
        mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
        area_size[i] = mu[i].m00;
    }
    
}

void max_double_value(double &max_moment, std::vector<float> &area_size)
{
    for (int i = 0; i < area_size.size(); i++)
    {
        if (area_size[i] > max_moment)
        {
            max_moment = area_size[i];
        }
    }
}

void record_similar_value(double &max_moment, std::vector<cv::Point2f> &mc,std::vector<float> &area_size, std::vector<cv::Point2f> &determined_mc)
{
    for (int i = 0; i < area_size.size(); i++)
    {
        if ( (max_moment - area_size[i] ) / max_moment < 0.5)  
        {
            determined_mc.insert(determined_mc.end(),mc[i]);
        }
    }
}

void get_optimal_points(std::vector<cv::Point2f> &blue_mc, std::vector<cv::Point2f> &green_mc ,std::vector<cv::Point2f> &red_mc, 
     std::vector<float> &blue_area_size, std::vector<float> &green_area_size, std::vector<float> &red_area_size,
     std::vector<cv::Point2f> &blue_determined_mc,std::vector<cv::Point2f> &green_determined_mc, std::vector<cv::Point2f> &red_determined_mc)
{
    double max_moment = DBL_MIN;
    max_double_value(max_moment,blue_area_size);
    max_double_value(max_moment,green_area_size);
    max_double_value(max_moment,red_area_size);
    record_similar_value(max_moment, blue_mc , blue_area_size , blue_determined_mc);
    record_similar_value(max_moment, green_mc , green_area_size , green_determined_mc);
    record_similar_value(max_moment, red_mc , red_area_size , red_determined_mc);
}

double getDistance(std::vector<cv::Point2i> points)
{
    double distance;
    distance = powf((points[0].x - points[1].x), 2) + powf((points[0].y - points[1].y), 2);
    distance = sqrtf(distance);
    return distance;
}

void calculate_result(std::vector<cv::Point2f> &blue_determined_mc,std::vector<cv::Point2f> &green_determined_mc,std::vector<cv::Point2f> &red_determined_mc,int max_point_x,int max_point_y, bool two_mimute_flag)
{
    int translated_solid_numbers[9] = {};
    cv::Point2i combined_point[9] = {};
    int blue_numbers = blue_determined_mc.size();
    int green_numbers = green_determined_mc.size();
    int red_numbers = red_determined_mc.size();
    cv::Point2i temp_point;
    int temp_value;
    for (int i = 0; i<blue_numbers ; i++)
    {
        translated_solid_numbers[i] = 1;
        combined_point[i] = blue_determined_mc[i];
    }
    for (int i = 0; i<green_numbers ; i++)
    {
        translated_solid_numbers[blue_numbers+i] = 2;
        combined_point[blue_numbers+i] = green_determined_mc[i];
    }
    for (int i = 0; i<red_numbers ; i++)
    {
        translated_solid_numbers[blue_numbers+green_numbers+i] = 0;
        combined_point[blue_numbers+green_numbers+i] = red_determined_mc[i];
    }
    cv::Point2i zero_point = {0,0};
    cv::Point2i max_point = {max_point_x,max_point_y};
    
    double min_moment = DBL_MAX;
    for (int j =0; j<9; j ++)
    {
        std::vector<cv::Point2i> temp = {combined_point[j],zero_point};
        if (getDistance(temp) < min_moment)
        {
            min_moment = getDistance(temp);
            temp_point = combined_point[0];
            temp_value = translated_solid_numbers[0];
            combined_point[0] = combined_point[j];
            translated_solid_numbers[0] = translated_solid_numbers[j];
            combined_point[j] = temp_point;
            translated_solid_numbers[j] = temp_value;
        }
    }
    
    min_moment = DBL_MAX;
    for (int j = 1; j<9; j ++)
    {
        std::vector<cv::Point2i> temp = {combined_point[j],combined_point[0]};
        if (getDistance(temp) < min_moment)
        {
            min_moment = getDistance(temp);
            temp_point = combined_point[1];
            temp_value = translated_solid_numbers[1];
            combined_point[1] = combined_point[j];
            translated_solid_numbers[1] = translated_solid_numbers[j];
            combined_point[j] = temp_point;
            translated_solid_numbers[j] = temp_value;
        }
    }
    
    min_moment = DBL_MAX;
    for (int j = 2; j<9; j ++)
    {
        std::vector<cv::Point2i> temp = {combined_point[j],combined_point[0]};
        if (getDistance(temp) < min_moment)
        {
            min_moment = getDistance(temp);
            temp_point = combined_point[2];
            temp_value = translated_solid_numbers[2];
            combined_point[2] = combined_point[j];
            translated_solid_numbers[2] = translated_solid_numbers[j];
            combined_point[j] = temp_point;
            translated_solid_numbers[j] = temp_value;
        }
    }
    
    
    if (combined_point[1].x > combined_point[2].x )
    {
        temp_point = combined_point[2];
        temp_value = translated_solid_numbers[2];
        combined_point[2] = combined_point[3];
        translated_solid_numbers[2] = translated_solid_numbers[3];
        combined_point[3] = temp_point;
        translated_solid_numbers[3] = temp_value;
    }else
    {
        temp_point = combined_point[1];
        temp_value = translated_solid_numbers[1];
        combined_point[1] = combined_point[3];
        translated_solid_numbers[1] = translated_solid_numbers[3];
        combined_point[3] = temp_point;
        translated_solid_numbers[3] = temp_value;
        temp_point = combined_point[1];
        temp_value = translated_solid_numbers[1];
        combined_point[1] = combined_point[2];
        translated_solid_numbers[1] = translated_solid_numbers[2];
        combined_point[2] = temp_point;
        translated_solid_numbers[2] = temp_value;
    }

    
    min_moment = DBL_MAX;
    for (int j = 8; j>1; j --)
    {
        if (j != 3)
        {
            std::vector<cv::Point2i> temp = {combined_point[j],max_point};
            if (getDistance(temp) < min_moment)
            {
                min_moment = getDistance(temp);
                temp_point = combined_point[8];
                temp_value = translated_solid_numbers[8];
                combined_point[8] = combined_point[j];
                translated_solid_numbers[8] = translated_solid_numbers[j];
                combined_point[j] = temp_point;
                translated_solid_numbers[j] = temp_value;
            }
        }
    }

    
    min_moment = DBL_MAX;
    for (int j = 7; j>1; j --)
    {
        if (j != 3)
        {
            std::vector<cv::Point2i> temp = {combined_point[j],combined_point[8]};
            if (getDistance(temp) < min_moment)
            {
                min_moment = getDistance(temp);
                temp_point = combined_point[7];
                temp_value = translated_solid_numbers[7];
                combined_point[7] = combined_point[j];
                translated_solid_numbers[7] = translated_solid_numbers[j];
                combined_point[j] = temp_point;
                translated_solid_numbers[j] = temp_value;
            }
        }
    }
    
    min_moment = DBL_MAX;
    for (int j = 6; j>1; j --)
    {
        if (j != 3)
        {
            std::vector<cv::Point2i> temp = {combined_point[j],combined_point[8]};
            if (getDistance(temp) < min_moment)
            {
                min_moment = getDistance(temp);
                temp_point = combined_point[6];
                temp_value = translated_solid_numbers[6];
                combined_point[6] = combined_point[j];
                translated_solid_numbers[6] = translated_solid_numbers[j];
                combined_point[j] = temp_point;
                translated_solid_numbers[j] = temp_value;
            }
        }
    }

    
    if (combined_point[6].x > combined_point[7].x )
    {
        temp_point = combined_point[6];
        temp_value = translated_solid_numbers[6];
        combined_point[6] = combined_point[5];
        translated_solid_numbers[6] = translated_solid_numbers[5];
        combined_point[5] = temp_point;
        translated_solid_numbers[5] = temp_value;
    }else
    {
        temp_point = combined_point[7];
        temp_value = translated_solid_numbers[7];
        combined_point[7] = combined_point[5];
        translated_solid_numbers[7] = translated_solid_numbers[5];
        combined_point[5] = temp_point;
        translated_solid_numbers[5] = temp_value;
        temp_point = combined_point[6];
        temp_value = translated_solid_numbers[6];
        combined_point[6] = combined_point[7];
        translated_solid_numbers[6] = translated_solid_numbers[7];
        combined_point[7] = temp_point;
        translated_solid_numbers[7] = temp_value;
    }

    min_moment = DBL_MAX;
    for (int j = 6; j>1; j --)
    {
        if (j != 3 && j != 5)
        {
            std::vector<cv::Point2i> temp = {combined_point[j],combined_point[8]};
            if (getDistance(temp) < min_moment)
            {
                min_moment = getDistance(temp);
                temp_point = combined_point[6];
                temp_value = translated_solid_numbers[6];
                combined_point[6] = combined_point[j];
                translated_solid_numbers[6] = translated_solid_numbers[j];
                combined_point[j] = temp_point;
                translated_solid_numbers[j] = temp_value;
            }
        }
    }
    
    
    temp_point = combined_point[6];
    temp_value = translated_solid_numbers[6];
    combined_point[6] = combined_point[4];
    translated_solid_numbers[6] = translated_solid_numbers[4];
    combined_point[4] = temp_point;
    translated_solid_numbers[4] = temp_value;

    
    if (combined_point[6].x > combined_point[2].x )
    {
        temp_point = combined_point[6];
        temp_value = translated_solid_numbers[6];
        combined_point[6] = combined_point[2];
        translated_solid_numbers[6] = translated_solid_numbers[2];
        combined_point[2] = temp_point;
        translated_solid_numbers[2] = temp_value;
    }

    std::cout << translated_solid_numbers[0] 
    << translated_solid_numbers[1]
    << translated_solid_numbers[2]
    << translated_solid_numbers[3]
    << translated_solid_numbers[4]
    << translated_solid_numbers[5]
    << translated_solid_numbers[6]
    << translated_solid_numbers[7]
    << translated_solid_numbers[8]
    << std::endl;

    int results = 0;
    for (int i = 8; i>-1; i --)
    {
        results += translated_solid_numbers[i]*powf((two_mimute_flag ? 2 : 3),(8-i));
    }
    std::cout << "Origin_Calculated_Number:" << results << std::endl;
    std::cout << "Output_Number:" << results%7 << std::endl;
    Serial_Communication(results);

}