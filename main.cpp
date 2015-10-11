#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cmath>
#include <set>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Sparse>

#include "utils.h"

using namespace std;
using namespace cv;
using namespace Eigen;

int main()
{
    cout << "setting constants" << endl;
    const double sigma = 0.1;
    const double beta = 0.0001;
    const double eps = 0.000001;
    const double Xb = 3.;
    const double Xf = 1.;
    const int dx[] = {1, 0, -1, 0};
    const int dy[] = {0, 1, 0, -1};
    cout << "reading images" << endl;
    Mat image = imread("../se_project_prototype/lena.jpg", CV_LOAD_IMAGE_COLOR);
    Mat bseedsImage = imread("../se_project_prototype/b.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat fseedsImage = imread("../se_project_prototype/f.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    cout << image.at<Vec3f>(0, 0)[0] << endl;
    image.convertTo(image, CV_32F);
    cout << image.at<Vec3f>(0, 0)[0] << endl;

    int N = image.rows;
    int M = image.cols;

    cout << "declaring sparse matrices" << endl;
    SparseMatrix <double> Wij; Wij.resize(N*M, N*M);// Wij.reserve(N*M); // weight adjacency matrix
    SparseMatrix <double> D; D.resize(N*M, N*M); //D.reserve(N*M*(N*M-1)); //diagonal matrix of valency weights;
    //SparseMatrix <double> L; L.resize(N*M, N*M); L.reserve(N*M);//L = D - W;
    SparseMatrix <double> Is; Is.resize(N*M, N*M); //Is.reserve(N*M*(N*M-1));//diagonal matrix (pixel in bseeds || pixel in fseeds):1 ? 0;
    VectorXd b; b.resize(N*M); //b[i] = Xb if i in bseeds; elseif i in fseeds b[i] = Xf; else b[i] = 0;
    vector <Triplet<double> > WijTriplet;
    vector <Triplet<double> > DTriplet;
    vector <Triplet<double> > IsTriplet;
    set <int> seeds;
    set <int> fseeds;
    set <int> bseeds;
    cout << "initializing sets" << endl;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < M; j++){
            int colorB = int(bseedsImage.at<uchar>(i, j));
            int colorF = int(fseedsImage.at<uchar>(i, j));
            if(colorB < 200){
                seeds.insert(i*M+j);
                bseeds.insert(i*M+j);
            }
            if(colorF < 200){
                seeds.insert(i*M+j);
                fseeds.insert(i*M+j);
            }
        }
    cout << "initializing sparse matrices" << endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            double sw = 0.;
            for(int k = 0; k < 4; k++){
                if((i+dy[k] < 0 || i+dy[k] >= N)||(j+dx[k] < 0 || j+dx[k] >= M)) continue;
                double w;
                w = norm(image.at<Vec3f>(i, j), image.at<Vec3f>(i+dy[k], j+dx[k]), NORM_INF);
                //Wij.insert(i*M+j, (i+dy[k])*M+j+dx[k]) = w;
                w = exp(-(beta*w*w)/(sigma)) + eps;
                WijTriplet.push_back(Triplet<double>(i*M+j, (i+dy[k])*M+j+dx[k], w));
                sw += w;
            }
            //D.insert(i*M+j, i*M+j) = sw;
            DTriplet.push_back(Triplet<double>(i*M+j, i*M+j, sw));
            //seeds.count(i*M+j) != 0 ? Is.insert(i*M+j, i*M+j) = 1. : Is.insert(i*M+j, i*M+j) = 0.;
            seeds.count(i*M+j) != 0 ? IsTriplet.push_back(Triplet<double>(i*M+j, i*M+j, 1.)) : IsTriplet.push_back(Triplet<double>(i*M+j, i*M+j, 0.));
            if(fseeds.count(i*M+j) != 0)
                b(i*M+j) = Xf;
            else if(bseeds.count(i*M+j) != 0)
                b(i*M+j) = Xb;
            else
                b(i*M+j) = 0;
        }
    }

    Wij.setFromTriplets(WijTriplet.begin(), WijTriplet.end());
    D.setFromTriplets(DTriplet.begin(), DTriplet.end());
    Is.setFromTriplets(IsTriplet.begin(), IsTriplet.end());

    SparseMatrix <double> L = D - Wij;
    L = L * L;

    cout << "Starting to solve equation..." << endl;

    SparseMatrix <double> A = Is + L;
    cout << A.rows() << " " << A.cols() << endl;
    //SparseQR<SparseMatrix<double>, AMDOrdering<int> > solverA;
    SimplicialLDLT<SparseMatrix<double> > solverA;
    solverA.compute(A);
    if(solverA.info()!=Success){
        cout << "Failed to compute A" << endl;
        return -1;
    }
    cout << "Solver computed A successfully" << endl;
    VectorXd x = solverA.solve(b);

    if(solverA.info() != Success){
        cout << "Failed to solve Ax = b" << endl;
        return -1;
    }

    cout << "Ax = b solved!" << endl;
    cout << "Applying new labels (colors)" << endl;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < M; j++){
            Vec3f color = image.at<Vec3f>(i, j);
            if(x(i*M+j)>=(Xb+Xf)/2){
                color[0] = 0.; color[1] = 0.; color[2] = 0.;
            }
//            else {
//                color[0] = 250.; color[1] = 250.; color[2] = 250.;
//            }
            image.at<Vec3f>(i, j) = color;
        }
    cout << "finished!" << endl;
    imwrite("segmented.jpg", image);
    namedWindow("image");
    imshow("image", image);
    waitKey(0);
    return 0;
}
