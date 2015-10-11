#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void printNMArray(int **a, int n = 0, int m = 0);

void matToArray(const Mat &input, int ** output);

#endif // UTILS_H
