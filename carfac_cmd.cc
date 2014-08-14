// Copyright 2013 The CARFAC Authors. All Rights Reserved.
// Author: Alex Brandmeyer
//
// This file is part of an implementation of Lyon's cochlear model:
// "Cascade of Asymmetric Resonators with Fast-Acting Compression"
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Mangled by Axel Tidemann and Boye Annfelt Hoverstad.

#include "carfac.h"

#include <string>
#include <fstream>
#include <iostream>

#include <Eigen/Core>

#include "agc.h"
#include "car.h"
#include "common.h"
#include "ihc.h"

// Reads a size rows by columns Eigen matrix from a text file written
// using the Matlab dlmwrite function.
ArrayXX LoadMatrix(const std::string& filename, int rows, int columns) {
  std::ifstream file(filename.c_str());
  ArrayXX output(rows, columns);
  CARFAC_ASSERT(file.is_open());
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      file >> output(i, j);
    }
  }
  file.close();
  return output;
}

void _filterloop(const ArrayX &b, const ArrayX &a, const ArrayXX& input, ArrayXX& output) {
  int order = b.rows();
  int rows = input.rows();
  int cols = input.cols();

  for(int i = 0; i < cols; i++)
    {
      ArrayX x = input.col(i);
      ArrayX y = output.col(i);

      for(int k = 0; k < rows; k++)
	y(k+1) = (b * x.segment(k - order + 1, order)).sum() - (a * y.segment(k - order, order)).sum();

      output.col(i) = y;
    }
}

void filter(const ArrayX &beta, const ArrayX &alpha, const ArrayXX& input, ArrayXX& output) {

  float a0 = alpha(0);

  // For convenience with multiplication, we reverse the coefficients.
  ArrayX a = alpha.segment(1, alpha.rows() - 1).reverse() / a0;
  ArrayX b = beta.reverse() / a0;

  _filterloop(b, a, input, output);
  output = output.colwise().reverse().eval();
  _filterloop(b, a, input.colwise().reverse().eval(), output);
  output = output.colwise().reverse().eval();
}

void WriteMatrix(const std::string& filename, const ArrayXX& matrix, int stride) {
  ArrayX b(1);
  b << 1.;
  ArrayX a(2);
  a <<  1., -0.995; //If you extend this vector, pad the matrix with N-2 zeros.

  ArrayXX filtered = ArrayXX::Zero(matrix.rows() + b.rows(), matrix.cols());
  filter(b, a, matrix, filtered);
  ArrayXX decimated(matrix.rows()/stride, matrix.cols());
  
  for(int i = 0; i < decimated.rows(); i++)
    decimated.row(i) = filtered.row(i*stride);

  std::ofstream ofile(filename.c_str());
  const int kPrecision = 9;
  ofile.precision(kPrecision);
  if (ofile.is_open()) {
    Eigen::IOFormat ioformat(kPrecision, Eigen::DontAlignCols);
    ofile << decimated.format(ioformat) << std::endl;
  }
  ofile.close();
}

// Reads a two dimensional vector of audio data from a text file
// containing the output of the Matlab wavread() function.
ArrayXX LoadAudio(const std::string& filename, int num_samples, int num_ears) {
  // The Matlab audio input is transposed compared to the C++.
  return LoadMatrix(filename, num_samples, num_ears).transpose();
}

// Writes the CARFAC NAP output to a text file.
void WriteNAPOutput(const CARFACOutput& output, const std::string& filename,
                    int ear, int stride) {
  WriteMatrix(filename, output.nap()[ear].transpose(), stride);
}

int main(int argc, char *argv[])
{
  CARParams car_params_;
  IHCParams ihc_params_;
  AGCParams agc_params_;
  bool open_loop_;
  std::string fname = argv[1];
  int num_samples = atoi(argv[2]);
  int num_ears = atoi(argv[3]); // 1
  int num_channels = atoi(argv[4]); //71
  FPType sample_rate = atoi(argv[5]); //22050
  int stride = atoi(argv[6]); //441, which is about ~20ms of audio at 22.05kHz
  ArrayXX sound_data = LoadAudio(fname + "-audio.txt", num_samples, num_ears);
  CARFAC carfac(num_ears, sample_rate, car_params_, ihc_params_, agc_params_);
  CARFACOutput output(true, true, false, false);
  carfac.RunSegment(sound_data, open_loop_, &output);
  //If you need more ears, this is where to loop it. Change the output filename accordingly.
  WriteNAPOutput(output, fname + "-output.txt", 0, stride); 
}
