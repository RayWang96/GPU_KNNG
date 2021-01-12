#ifndef XMUKNN_FILETOOL_HPP
#define XMUKNN_FILETOOL_HPP
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

namespace xmuknn {
class FileTool {
 public:
  static void ReadFVecs(const string &data_path, float **vectors_ptr,
                        int *num_ptr, int *dim_ptr, int read_num = -1) {
    float *&vectors = *vectors_ptr;
    int &num = *num_ptr;
    int &dim = *dim_ptr;
    ifstream in(data_path, ios::binary);
    if (!in.is_open()) {
      cerr << "Can't open " << data_path << endl;
      exit(-1);
    }
    in.read((char *)&dim, 4);
    if (read_num == -1) {
      in.seekg(0, ios::end);
      ios::pos_type file_tail_pos = in.tellg();
      size_t file_size = (size_t)file_tail_pos;
      read_num = file_size / (size_t)(dim + 1) / 4;
    }
    in.seekg(0, ios::beg);
    num = read_num;
    vectors = new float[(size_t)read_num * (size_t)dim];
    int tmp = 0;
    for (int i = 0; i < read_num; i++) {
      in.read((char *)&tmp, 4);
      in.read((char *)(vectors + (size_t)i * dim), dim * 4);
    }
    in.close();
  }

  static void WriteFVecs(const string &data_path, const float *vectors,
                         const int num, const int dim) {
    ofstream out(data_path, ios::binary);
    for (int i = 0; i < num; i++) {
      out.write((char*)&dim, 4);
      out.write((char*)vectors, dim * 4);
    }
    out.close();
  }

  static void ReadIVecs(const string &data_path, int **vectors_ptr,
                        int *num_ptr, int *dim_ptr, int read_num = -1) {
    int *&vectors = *vectors_ptr;
    int &num = *num_ptr;
    int &dim = *dim_ptr;
    ifstream in(data_path, ios::binary);
    if (!in.is_open()) {
      cerr << "Can't open " << data_path << endl;
      exit(-1);
    }
    in.read((char *)&dim, 4);
    if (read_num == -1) {
      in.seekg(0, ios::end);
      ios::pos_type file_tail_pos = in.tellg();
      size_t file_size = (size_t)file_tail_pos;
      read_num = file_size / (size_t)(dim + 1) / 4;
    }
    in.seekg(0, ios::beg);
    num = read_num;
    vectors = new int[(size_t)read_num * (size_t)dim];
    int tmp = 0;
    for (int i = 0; i < read_num; i++) {
      in.read((char *)&tmp, 4);
      in.read((char *)(vectors + (size_t)i * dim), dim * 4);
    }
    in.close();
  }

  static void WriteIVecs(const string &data_path, const int *vectors,
                         const int num, const int dim) {
    ofstream out(data_path, ios::binary);
    for (int i = 0; i < num; i++) {
      out.write((char*)&dim, 4);
      out.write((char*)vectors, dim * 4);
    }
    out.close();
  }

  static void ReadVecs(const string &data_path, float **vectors_ptr,
                       int *num_ptr, int *dim_ptr, const bool show_process = true) {
    float *&vecs = *vectors_ptr;
    int &num = *num_ptr;
    int &dim = *dim_ptr;
    std::ifstream in(data_path);
    if (!in.is_open()) {
      throw(std::string("Failed to open ") + data_path);
    }
    in >> num >> dim;
    std::cerr << num << " " << dim << std::endl;
    vecs = new float[num * dim];
    for (int i = 0; i < num; i++) {
      if (show_process) {
        if (i % 12345 == 0) std::cerr << "\r" << i;
        if (i == num - 1) std::cerr << "\r" << num;
      }
      for (int j = 0; j < dim; j++) {
        in >> vecs[i * dim + j];
      }
    }
    if (show_process) {
      std::cerr << std::endl;
    }
    in.close();
  }

  static void ReadVecs(float *&vecs, int &size, int &dim,
                       const std::string &data_path, bool show_process = true) {
    std::ifstream in(data_path);
    if (!in.is_open()) {
      throw(std::string("Failed to open ") + data_path);
    }
    in >> size >> dim;
    std::cerr << size << " " << dim << std::endl;
    vecs = new float[size * dim];
    for (int i = 0; i < size; i++) {
      if (show_process) {
        if (i % 12345 == 0) std::cerr << "\r" << i;
        if (i == size - 1) std::cerr << "\r" << size;
      }
      for (int j = 0; j < dim; j++) {
        in >> vecs[i * dim + j];
      }
    }
    if (show_process) {
      std::cerr << std::endl;
    }
    in.close();
  }
};
}  // namespace xmuknn

#endif