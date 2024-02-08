// -----------------------------------------------------------------------------
//
// Generates TSP instances Tnm as described in the paper
// "Hard to Solve Instances of the Euclidean Traveling Salesman Problem"
// by Hougardy and Zhong (arXiv:1808.02859 [cs.DM]). The values for n and m
// are chosen according to equation (8) of that paper.
//
// usage: Tnm <#vertices>
// where #vertices is an integer > 50 with remainder 1 mod 3.
//
// The instance is output in TSPLIB format to a file named "Tnm<#vertices>.tsp".
//
// Use "g++ -std=c++11 Tnm.cpp" to compile this code with gcc.
//
// Stefan Hougardy and Xianghui Zhong, September 2018
//
// -----------------------------------------------------------------------------



#include <iostream>
#include <stdexcept>
#include <cmath>
#include <fstream>


static int NumPoints;


void AddScaledPoint(std::ofstream & TnmFile, double x, double y)
{
  const int ScaleFactor = 10000;

  NumPoints++;
  TnmFile << NumPoints << " " << static_cast<long>(std::round(x * ScaleFactor)) <<
                          " " << static_cast<long>(std::round(y * ScaleFactor)) << "\n";
}


void CreateTnmInstance(int N)
{
  std::string filename = "Tnm" + std::to_string(N) + ".tsp";
  std::ofstream TnmFile(filename);

  int n = (3 * N - 40) / 10;
  int m = (N + 2) / 3 - n;

  TnmFile << "NAME : " << filename << "\n";
  TnmFile << "COMMENT : " << N << "-city problem, points on edges of tetrahedron. n = "
          << n << "  m = " << m << " (Hougardy + Zhong)\n";
  TnmFile << "TYPE : TSP\n";
  TnmFile << "DIMENSION : " << N << "\n";
  TnmFile << "EDGE_WEIGHT_TYPE : EUC_2D\n";
  TnmFile << "NODE_COORD_SECTION\n";


  NumPoints = 0;

  for (int i = 1; i  <= n; i++)
  {
    AddScaledPoint(TnmFile, n      - i / 2.,     i * sqrt(3) / 2.);
    AddScaledPoint(TnmFile, n / 2. - i / 2., (n-i) * sqrt(3) / 2.);
    AddScaledPoint(TnmFile,             i,                   0);
  }

  for (int j = 1; j < m; j++)
  {
    AddScaledPoint(TnmFile,     j * n / (2. * m), j * n / (2. * sqrt(3) * m));
    AddScaledPoint(TnmFile, n - j * n / (2. * m), j * n / (2. * sqrt(3) * m));
    AddScaledPoint(TnmFile,               n / 2., n * sqrt(3) / 2. - j * n / (sqrt(3) * m));
  }

  AddScaledPoint(TnmFile, n / 2., n / (2. * sqrt(3)));

  TnmFile << "EOF\n";
  TnmFile.close();
}


int main(int argc, char* argv[])
{
  if (argc != 2)
    throw std::runtime_error("no argument given. Need #vertices as argument.");

  int N = std::atoi(argv[1]);

  if ((N < 50) or (N % 3 != 1))
    throw std::runtime_error("the argument must be an integer > 50 with remainder 1 mod 3.");

  CreateTnmInstance(N);
}