//
//  main.cpp
//
//
//  Created by Songyou Peng on 06/10/15.
//  Copyright © 2015 Songyou Peng. All rights reserved.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;


int main(int argc, const char * argv[]) {

    Mat input, input_border, back_mask, fore_mask, output;
    input = imread("/Users/Songyou/Desktop/Samples/lena.jpg");
    imshow("input", input);
    int height = input.rows;
    int width = input.cols;
    output = input;
    
    SparseMatrix<double> A(height*width, height*width), W(height*width, height*width), D(height*width, height*width), Is(height*width, height*width), L(height*width, height*width);
    A.setZero();
    W.setZero();
    D.setZero();
    Is.setZero();
    L.setZero();
    
    
    VectorXd b(height*width), x(height*width);
    b.setZero();
    x.setZero();
    //A.coeffRef(1, 1) = exp(1);
    //VectorXd b, x;
    SparseLU<SparseMatrix<double>> solver;
    //cout.precision(10);
    //cout << A.rows() <<endl;
    //cout << A.coeffRef(1, 1) <<endl;
    //imshow("Baby",image1);
    //waitKey(0);
    
    Mat Weight_Matrix;//Initialize the weight matrix and Diagonal matrix, whose height and width are equal to the image's height*width
    //Weight_Matrix = Mat::zeros(height*width, height*width, CV_64F);
    //A = Mat::zeros(height*width, height*width, CV_64F);
    //D = Mat::zeros(height*width, height*width, CV_64F);
    //Is = Mat::zeros(height*width, height*width, CV_64F);
    //L = Mat::zeros(height*width, height*width, CV_64F);
    //b = Mat::zeros(height*width, 1, CV_64F);
    //x = Mat::zeros(height*width, 1, CV_64F);
    const float beta=1.0, epsilon = pow(10,-6);
    float xb=100, xf=1;//background and foreground label
    copyMakeBorder(input, input_border, 1, 1, 1, 1, BORDER_REPLICATE);//Adding borders to original image in order to process image easily
    int height_border = input_border.rows;
    int width_border = input_border.cols;
    
    back_mask = imread("/Users/Songyou/Desktop/Samples/b.jpg", 0);
    fore_mask = imread("/Users/Songyou/Desktop/Samples/f.jpg", 0);
    
    for (int i = 1; i < (height_border-1); ++i)
    {
        for (int j = 1; j < (width_border-1); ++j)
        {
            int sigma = 0;
            int max_in_3_channels[4] = {0};
            int neighbor[12] = {0};
            //            Mat c;
            //            double max, min;
            //            c = input(Rect(i-1,j-1,3,3));
            //            minMaxIdx(c, &min, &max);
            
            for (int t = 0; t < 3; ++t)//visit R G B 3 channels
            {
                
                
                //4 neighbors in t th channels
                neighbor[t*4+0] = abs(input_border.ptr<uchar>(i-1,j)[t]-input_border.ptr<uchar>(i,j)[t]);
                //cout << neighbor[t*4+0]<<endl;
                neighbor[t*4+1] = abs(input_border.ptr<uchar>(i,j-1)[t]-input_border.ptr<uchar>(i,j)[t]);
                neighbor[t*4+2] = abs(input_border.ptr<uchar>(i+1,j)[t]-input_border.ptr<uchar>(i,j)[t]);
                neighbor[t*4+3] = abs(input_border.ptr<uchar>(i,j+1)[t]-input_border.ptr<uchar>(i,j)[t]);
                
                
            }
            
            for (int p=0; p < 3; ++p)//Acquire the max value in 3 channels of a pixel
            {
                if(neighbor[p*4] >= max_in_3_channels[0])
                    max_in_3_channels[0] = neighbor[p*4];
                if(neighbor[p*4+1] >= max_in_3_channels[1])
                    max_in_3_channels[1] = neighbor[p*4+1];
                if(neighbor[p*4+2] >= max_in_3_channels[2])
                    max_in_3_channels[2] = neighbor[p*4+2];
                if(neighbor[p*4+3] >= max_in_3_channels[3])
                    max_in_3_channels[3] = neighbor[p*4+3];
            }
            
            for(int p=0; p<4;++p)
            {
                //Sigma is equal to the max value of the substraction with 4 neighbors in RGB 3 channels
                if (max_in_3_channels[p] >= sigma)
                    sigma=max_in_3_channels[p];
            }
            
            
            //Form weigth matrix, considering different situation of image border
            //cout.precision(10);
            //cout <<exp(-beta*float(max_in_3_channels[0])/float(sigma))+epsilon<<endl;
            if (i > 1)
            {
                W.coeffRef((i-1)*height+(j-1),(i-2)*height+j-1) = exp(-beta*float(max_in_3_channels[0])*float(max_in_3_channels[0])/float(sigma))+epsilon;
                D.coeffRef((i-1)*height+(j-1), (i-1)*height+(j-1)) += W.coeffRef((i-1)*height+(j-1),(i-2)*height+j-1);
            }
//            cout.precision(10);
//            cout <<exp(-beta*float(max_in_3_channels[1])/float(sigma))+epsilon<<endl;
            if (j > 1)
            {
                W.coeffRef((i-1)*height+(j-1),(i-1)*height+(j-2)) = exp(-beta*float(max_in_3_channels[1])*float(max_in_3_channels[1])/float(sigma))+epsilon;
                D.coeffRef((i-1)*height+(j-1), (i-1)*height+(j-1)) += W.coeffRef((i-1)*height+(j-1),(i-1)*height+(j-2));
            }
            if (i < (height_border - 2) )
            {
                W.coeffRef((i-1)*height+(j-1),i*height+(j-1)) = exp(-beta*float(max_in_3_channels[2])*float(max_in_3_channels[2])/float(sigma))+epsilon;
                D.coeffRef((i-1)*height+(j-1), (i-1)*height+(j-1)) += W.coeffRef((i-1)*height+(j-1),i*height+(j-1));
            }
            if (j < (width_border - 2))
            {
                W.coeffRef((i-1)*height+(j-1),(i-1)*height+j) = exp(-beta*float(max_in_3_channels[3])*float(max_in_3_channels[3])/float(sigma))+epsilon;
                D.coeffRef((i-1)*height+(j-1), (i-1)*height+(j-1)) += W.coeffRef((i-1)*height+(j-1),(i-1)*height+j);
            }
            
            
            /*********************************Form Seed relate matrix************************************************/
            
            if ((int)fore_mask.at<uchar>(i-1,j-1) == 0)//When the pixel in fore mask is 0, then Is(i,i) = 1, b(i) = xf
            {
                Is.coeffRef((i-1)*height+(j-1), (i-1)*height+(j-1)) = 1.0;
                b((i-1)*height+(j-1)) =xf;
            }
            
            if ((int)back_mask.at<uchar>(i-1,j-1) == 0)//When the pixel in back mask is 0, then Is(i,i) = 1, b(i) = xb
            {
                Is.coeffRef((i-1)*height+(j-1), (i-1)*height+(j-1)) = 1.0;
                b((i-1)*height+(j-1)) =xb;
            }
            
        }
    }
    
    
    L = D - W;
    A = L * L + Is;
    
    solver.compute(A);
    if(solver.info()!=Success)
        exit(0);
    
    x = solver.solve(b);//Solve fomulation (7)
    
    if(solver.info()!=Success)
        exit(0);

    for (int i =0; i<height*width; ++i)
    {
        if(x(i) >50)//it is background pixel
        {
            int w = i % height;
            int h = i / height;
            output.ptr<uchar>(h,w)[0] = 0;
            output.ptr<uchar>(h,w)[1] = 0;
            output.ptr<uchar>(h,w)[2] = 0;
        }
    }
    
    imshow("output", output);
    waitKey(0);
        
    
//    
//    cout.precision(10);
//    for(int i = 0; i < height*width; ++i)
//        cout << x(i) << endl;
//        
    return 0;
}
