#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
#include "functions.h"

#ifdef DEBUGMODE
    #include <time.h>
#endif

int main() {
    #ifdef DEBUGMODE
    double duration;
    clock_t test_time_start, test_time_end;
    #endif

    // for camera setup
    cv::VideoCapture cap(VIDEOSTREAM); // video
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);//width
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);//height
    cap.set(cv::CAP_PROP_FPS, 2);// frame rate
    cap.set(cv::CAP_PROP_BRIGHTNESS, 50);//lightness 
    cap.set(cv::CAP_PROP_CONTRAST,50);// 
    cap.set(cv::CAP_PROP_SATURATION, 50);//contrast
    cap.set(cv::CAP_PROP_HUE, 0);//coloer tune 50
    cap.set(cv::CAP_PROP_EXPOSURE,50);
    assert(cap.isOpened());

    while (cap.isOpened()) {
        #ifdef DEBUGMODE
        test_time_start = clock();
        #endif
        cv::Mat image;
        cap >> image;
        
        int max_point_x = image.cols;
        int max_point_y = image.rows;
        
        std::vector<std::vector<cv::Point>> blue_contours;
        std::vector<cv::Vec4i> blue_hierarchy;
        cv::Mat blue_imgBinary;
        int blue_color[3] = {255,127,127};
        std::vector<std::vector<cv::Point>> green_contours;
        std::vector<cv::Vec4i> green_hierarchy;
        cv::Mat green_imgBinary;
        int green_color[3] = {127,255,127};
        std::vector<std::vector<cv::Point>> red_contours;
        std::vector<cv::Vec4i> red_hierarchy;
        cv::Mat red_imgBinary;
        int red_color[3] = {127,127,255};
       
        find_rec_contours(image, blue_imgBinary, blue_color, blue_contours, blue_hierarchy);
        find_rec_contours(image, green_imgBinary, green_color, green_contours, green_hierarchy);
        find_rec_contours(image, red_imgBinary, red_color, red_contours, red_hierarchy);
        // moment calculation https://blog.csdn.net/qq_34914551/article/details/78916084
        if (blue_contours.size() +  green_contours.size() + red_contours.size() > 8){ 
            
            std::vector<cv::Point2f> blue_mc(blue_contours.size());
            std::vector<cv::Point2f> green_mc(green_contours.size());
            std::vector<cv::Point2f> red_mc(red_contours.size());

            std::vector<float> blue_area_size(blue_contours.size());
            std::vector<float> green_area_size(green_contours.size());
            std::vector<float> red_area_size(red_contours.size());
            
            get_center_points(blue_contours, blue_mc, blue_area_size);
            get_center_points(green_contours, green_mc, green_area_size);
            get_center_points(red_contours, red_mc, red_area_size);
            
            /// contours: 
            cv::Mat dstImage = cv::Mat::zeros(blue_imgBinary.size(), CV_8UC1);
            for (int i = 0; i < blue_hierarchy.size(); i++)
            {
                drawContours(dstImage, blue_contours, i, cv::Scalar(255), cv::FILLED, 8, blue_hierarchy);
                cv::putText(dstImage, "1", blue_mc[i] , cv::FONT_HERSHEY_COMPLEX , 1,  cv::Scalar(127), 1, 8, 0);
            }
            for (int i = 0; i < green_hierarchy.size(); i++)
            {
                drawContours(dstImage, green_contours, i, cv::Scalar(255), cv::FILLED, 8, green_hierarchy);
                cv::putText(dstImage, "2", green_mc[i] , cv::FONT_HERSHEY_COMPLEX , 1,  cv::Scalar(127), 1, 8,0);
            }
            for (int i = 0; i < red_hierarchy.size(); i++)
            {
                drawContours(dstImage, red_contours, i, cv::Scalar(255), cv::FILLED, 8, red_hierarchy);
                cv::putText(dstImage, "0", red_mc[i] , cv::FONT_HERSHEY_COMPLEX , 1,  cv::Scalar(127), 1, 8, 0);
            }
            
            #ifdef VIDEOMODEON
            cv::imshow("Result", dstImage);
            #endif
            std::vector<cv::Point2f> blue_determined_mc;
            std::vector<cv::Point2f> green_determined_mc;
            std::vector<cv::Point2f> red_determined_mc;
            
            get_optimal_points(blue_mc, green_mc ,red_mc, blue_area_size, green_area_size, red_area_size, blue_determined_mc,green_determined_mc,red_determined_mc);
            

            if ((blue_determined_mc.size()+green_determined_mc.size()+ red_determined_mc.size()) == 9)
            {
                calculate_result(blue_determined_mc,green_determined_mc,red_determined_mc,max_point_x,max_point_y, green_determined_mc.size() == 0);

            }
            
        }
        
        // image display
        #ifdef VIDEOMODEON
        cv::imshow("Realtime Img", image);
        #endif
        #ifdef DEBUGMODE
        test_time_end = clock();
        duration = (double) (test_time_end - test_time_start);
        std::cout << "time cost for single loop:" << (duration/CLOCKS_PER_SEC) << "s" << std::endl;
        Serial_Communication(4);
        #endif
        char key = (char) cv::waitKey(30);
        if (key == 27)
            break;

    }

    return 0;
}