#include "utils.h"

void printNMArray(int **a, int n, int m)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
            cout << a[i][j] << " ";
        cout << endl;
    }
}

void matToArray(const Mat &input, int ** output)
{
    vector<uchar> pix;
    pix.assign(input.datastart, input.dataend);
    int m = input.cols, j = 0, i = 0;
    for(vector<uchar>::iterator it = pix.begin(); it < pix.end(); it++)
    {
        output[i][j] = int(*it);
        j++;
        if(j == m)
            j = 0, i++;
    }
}

/*
 *
    cout << "Initialization." << endl;
    vector < pair < int, vector < pair <int, double> > > > v; //graph
    vector <double> dw; // weighted valency
    vector <double> bg, fg; //background, foreground
    const double sigma = 0.1;
    const double eps = 0.000001;
    const double beta = 1.0;
    const double Xb = 1.;
    const double Xf = -1.;
    const int dx[] = {1, 0, -1, 0};
    const int dy[] = {0, 1, 0, -1};

    Mat image = imread("../se_project_prototype/lena.jpg", CV_LOAD_IMAGE_COLOR);
    if(!image.data)
    {
        cerr << "Error loading image..." << endl;
        return -1;
    }
    vector< vector<double> > D(image.rows*image.cols, vector<double>(image.rows*image.cols, 0)); //diagonal matrix containing d[i], i=0..NxM-1
    vector< vector<double> > Is(image.rows*image.cols, vector<double>(image.rows*image.cols, 0)); //Is diagonal matrix
    vector< vector<double> > L(image.rows*image.cols, vector<double>(image.rows*image.cols, 0)); // graph Laplacian matrix
    vector< vector<double> > L2(image.rows*image.cols, vector<double>(image.rows*image.cols, 0)); // L*L
    vector< vector<double> > W(image.rows*image.cols, vector<double>(image.rows*image.cols, 0)); //adjacency matrix of graph
    vector< vector<double> > IsLL(image.rows*image.cols, vector<double>(image.rows*image.cols, 0)); // Is-L*L

    vector <double> b(image.rows*image.cols, 0); //Ax = b

    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            double sw = 0.;
            for(int k = 0; k < 4; k++){
                if((i+dy[k] < 0 || i+dy[k] > image.rows)||(j+dx[k] < 0 || j+dx[k] > image.cols)) continue;
                double w;
                w = norm(image.at<Vec3f>(i, j), image.at<Vec3f>(i+dy[k], j+dx[k]), NORM_INF);
                W[i*image.cols+j][(i+dy[k])*image.cols+j+dx[k]] = w;
                //neighbors.push_back(make_pair((i+dy[k])*image.cols+j+dx[k], w));
                sw += exp(-(beta*w*w)/(sigma)) + eps;
            }
            D[i*image.cols+j][i*image.cols+j] = sw;
        }
    }
    cout << "Initialization.." << endl;

    for(int i = 0; i < image.rows*image.cols; i++)
        for(int j = 0; j < image.rows*image.cols; j++)
            L[i][j] = D[i][j] - W[i][j];


    Mat backIm = imread("../se_project_prototype/b.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat foreIm = imread("../se_project_prototype/f.jpg", CV_LOAD_IMAGE_GRAYSCALE);



    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            if(backIm.at<double>(i,j) < 250){
                Is[i*image.cols+j][i*image.cols+j] = 1.;
                bg.push_back(i*image.cols+j);
                b[i*image.cols+j] = Xb;
            }
            if(foreIm.at<double>(i,j) < 250){
                Is[i*image.cols+j][i*image.cols+j] = 1.;
                fg.push_back(i*image.cols+j);
                b[i*image.cols+j] = Xf;
            }
        }
    }

    cout << "Initialization..." << endl;

    //compute L*L;
    for(int i = 0; i < image.rows; i++)
        for(int j = 0; j < image.cols; j++){
            double t = 0.;
            for(int k = 0; k < image.rows; k++){
                t += L[i][k]*L[k][j];
            }
            L2[i][j] = t;
        }
    //compute Is + L2
    for(int i = 0; i < image.rows; i++)
        for(int j = 0; j < image.cols; j++)
            IsLL[i][j] = Is[i][j] + L2[i][j];

    //solve IsLL*x = b
    IsLL.push_back(b);
    vector<double> x = gauss(IsLL);
    for(int i = 0; i < image.rows; i++)
        for(int j = 0; i < image.cols; j++){
            Vec3f color = image.at<Vec3f>(i, j);
            if(x[i*image.cols+j]>=0){
                color[0] = 0.; color[1] = 0.; color[2] = 0.;
            } else {
                color[0] = 250.; color[1] = 250.; color[2] = 250.;
            }
            image.at<Vec3f>(i, j) = color;
        }
    cout << "Done." << endl;
*/
