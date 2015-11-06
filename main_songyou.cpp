//
//  main.cpp
//  SparseMatrix2_xcode
//
//  Created by Songyou Peng on 28/10/15.
//  Copyright © 2015 Songyou Peng. All rights reserved.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Sparse>

//using namespace cv;
//using namespace Eigen;

#define NUM_NEIGHBOR 4

int main(int argc, const char * argv[]) {
    
    
    //Initialize input and output image.
    cv::Mat input, back_mask, fore_mask, output;
    input = cv::imread("/Users/Songyou/Desktop/Samples/test.jpg");//input image
    cv::imshow("input", input);
    int height = input.rows;
    int width = input.cols;
    output = input;
    input.convertTo(input, CV_64F);//Convert unsigned char(0-255) image to double so we can process easier.
    
    back_mask = cv::imread("/Users/Songyou/Desktop/Samples/test_b.jpg", 0);//Input background mask
    fore_mask = cv::imread("/Users/Songyou/Desktop/Samples/test_f.jpg", 0);//Input foreground mask
    
    
    
    //Initialize Sparse Matrix relate matrices and vectors.
    std::cout << "Start to initialize sparse matrix" << std::endl;
    Eigen::SparseMatrix<double> A(height * width, height * width), W(height * width, height * width), D(height * width, height * width), Is(height * width, height * width), L(height * width, height * width);
    A.setZero(); W.setZero(); D.setZero(); Is.setZero(); L.setZero();
    
    Eigen::VectorXd b(height * width), x(height * width);
    b.setZero(); x.setZero();
    
    //IMPORTANT! Reservation extremely increase the speed!
    //Reserve a certain number of cell in every column of sparse matrix
    W.reserve(Eigen::VectorXf::Constant(height * width, 4));
    D.reserve(Eigen::VectorXf::Constant(height * width, 1));
    Is.reserve(Eigen::VectorXf::Constant(height * width, 1));
    L.reserve(Eigen::VectorXf::Constant(height * width , 5));
    
    std::cout << "Initialization of sparse matrix finished" << std::endl;
    //double tmpSigma = -1000;
    
    //Initialize the variables that we will use in the segmentation process
    const double beta = 0.1, epsilon = pow(10, -6);
    const double xb = 3, xf = 1;//background and foreground label
    double sigma = -1000.0;//Initialize sigma
    int dy[NUM_NEIGHBOR] = {0,-1,0,1};//dy and dx represent the neighbor of a pixel
    int dx[NUM_NEIGHBOR] = {-1,0,1,0};
    
    std::cout << "Start to calculate sigma" << std::endl;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            
            cv::Mat neighbor(4, 1, CV_64FC1, cv::Scalar(0.0));
            
            for (int p = 0; p < NUM_NEIGHBOR; ++p)
            {
                double tmpsigma = -1000;
                if (i + dx[p] < 0 || i + dx[p] >= height || j + dy[p] < 0 || j + dy[p] >= width) continue;//Deal with border of image
                
                neighbor.at< double >(p) = norm(input.at< cv::Vec3d >(i, j), input.at< cv::Vec3d >(i + dx[p], j + dy[p]), cv::NORM_INF);
                tmpsigma = norm(neighbor, cv::NORM_INF);
                if(tmpsigma > sigma) sigma = tmpsigma;
            }
        }
    }
    std::cout << "sigma caulculation finished, sigma of the image is: " << sigma << std::endl << std::endl;;
    
    std::cout << "Start to calculate the Sparse matrix that we need" << std::endl << "......" << std::endl;;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            
            cv::Mat neighbor(4, 1, CV_64FC1, cv::Scalar(0.0));
            
            for (int p = 0; p < NUM_NEIGHBOR; ++p)
            {
                if (i + dx[p] < 0 || i + dx[p]>= height || j + dy[p]<0 || j + dy[p] >= width) continue;
                
                neighbor.at< double >(p) = norm(input.at< cv::Vec3d >(i, j), input.at< cv::Vec3d >(i + dx[p], j + dy[p]), cv::NORM_INF);
                
            }
            
            //put value into W & D
            for (int p = 0; p < NUM_NEIGHBOR; ++p)
            {
                if (i + dx[p] < 0 || i + dx[p] >= height || j + dy[p]<0 || j + dy[p] >= width) continue;
                W.coeffRef(i * width + j, (i + dx[p]) * width + j + dy[p]) = exp(-beta * neighbor.at< double >(p) * neighbor.at< double >(p) / sigma) + epsilon;
                D.coeffRef(i * width + j, i * width + j) += W.coeffRef(i * width + j, (i + dx[p]) * width + j + dy[p]);
                
            }
            
            
            /*********************************Form Seed relate matrix************************************************/
            
            if ((int)fore_mask.at<uchar>(i,j) < 100)//When the pixel in fore mask is 0, then Is(i,i) = 1, b(i) = xf
            {
                Is.coeffRef(i * width + j, i * width + j) = 1.0;
                b(i * width + j) =xf;
            }
            
            if ((int)back_mask.at<uchar>(i,j) < 100)//When the pixel in back mask is 0, then Is(i,i) = 1, b(i) = xb
            {
                Is.coeffRef(i * width + j, i * width + j) = 1.0;
                b(i * width + j) =xb;
                
            }
            
            
            
            
        }
    }
    
    L = D - W;
    //L = L*L;
    A = L * L + Is;
    
    std::cout << "Calculation finished! Now solve linear system" << std::endl << std:: endl;
    
    //SparseLU<SparseMatrix<double>> solver;
    //SimplicialLLT<SparseMatrix<double>> solver;
    Eigen::SimplicialLDLT< Eigen::SparseMatrix<double> > solver;
    
    solver.compute(A);
    if(solver.info()!=Eigen::Success)
        std::cout << "bad" << std::endl;
    
    x = solver.solve(b);//Solve fomulation (7)
    
    if(solver.info() != Eigen::Success)
        exit(0);
    
    std::cout << "Soving successfully! Please look at the output." << std::endl;
    //Try to find lobioa
    for (int i = 0; i< height * width; ++i)
    {
        if(x(i) > ((xb + xf)/2))//it is background pixel
        {
            int w = i % width;
            int h = i / width;
            output.ptr< uchar >(h,w)[0] = 0;
            output.ptr< uchar >(h,w)[1] = 0;
            output.ptr< uchar >(h,w)[2] = 0;
        }
    }
    
    cv::imshow("output", output);
    cv::waitKey(0);
    
    
    return 0;
}
