#ifndef XMUKNN_FILETOOL_HPP
#define XMUKNN_FILETOOL_HPP
#include <assert.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>

#include "nndescent_element.cuh"
using namespace std;

class FileTool {
 public:
  static int GetFVecsNum(const string &data_path) {
    ifstream in(data_path, ios::binary);
    if (!in.is_open()) {
      cerr << "Can't open " << data_path << endl;
      return 0;
    }
    int dim;
    in.read((char *)&dim, 4);
    in.seekg(0, ios::end);
    size_t file_size = (size_t)in.tellg();
    int num = (int)(file_size / (size_t)(dim + 1) / 4);
    in.close();
    return num;
  }

  static int GetFVecsDim(const string &data_path) {
    ifstream in(data_path, ios::binary);
    if (!in.is_open()) {
      cerr << "Can't open " << data_path << endl;
      return 0;
    }
    int dim;
    in.read((char *)&dim, 4);
    in.close();
    return dim;
  }

  static int GetIVecsNum(const string &data_path) {
    ifstream in(data_path, ios::binary);
    if (!in.is_open()) {
      cerr << "Can't open " << data_path << endl;
      return 0;
    }
    int dim;
    in.read((char *)&dim, 4);
    in.seekg(0, ios::end);
    size_t file_size = (size_t)in.tellg();
    int num = (int)(file_size / (size_t)(dim + 1) / 4);
    in.close();
    return num;
  }

  static int GetKNNListsNum(const string &knn_graph_path) {
    ifstream in(knn_graph_path, ios::binary);
    if (!in.is_open()) {
      cerr << "Can't open " << knn_graph_path << endl;
      return 0;
    }
    int neighb_num;
    in.read((char *)&neighb_num, 4);
    in.seekg(0, ios::end);
    size_t file_size = (size_t)in.tellg();
    int num = (int)(file_size / (size_t)(neighb_num * 2 + 1) / 4);
    in.close();
    return num;
  }

  static void CreateBlankKNNGraph(const string &data_path, const int num,
                                  const int dim) {
    string cmd = "fallocate -l ";
    size_t byte_size = (size_t)num * (dim + 1) * sizeof(NNDElement);
    cmd += to_string(byte_size);
    cmd += " ";
    cmd += data_path;
    int code = system(cmd.c_str());
    if (code != 0){
      cerr << "fallocate error" << endl;
      exit(-1);
    }

    ofstream out(data_path, ios::in | ios::binary);
    cerr << num << " " << dim << endl;
    out.write((char *)&dim, 4);
    out.close();
  }

  template <typename T>
  static void ReadBinaryVecs(const string &data_path, T **vectors_ptr,
                             int *num_ptr, int *dim_ptr) {
    // T *&vectors = *vectors_ptr;
    // int &num = *num_ptr;
    // int &dim = *dim_ptr;
    // ifstream in(data_path, ios::binary);
    // if (!in.is_open()) {
    //   cerr << "Can't open " << data_path << endl;
    //   exit(-1);
    // }
    // in.read((char *)&dim, 4);
    // in.seekg(0, ios::end);
    // ios::pos_type file_tail_pos = in.tellg();
    // size_t file_size = (size_t)file_tail_pos;
    // num = file_size / (size_t)(4 + dim * sizeof(T));

    // in.seekg(0, ios::beg);
    // vectors = new T[(size_t)num * (size_t)dim];
    // int tmp = 0;
    // for (int i = 0; i < num; i++) {
    //   in.read((char *)&tmp, 4);
    //   in.read((char *)(vectors + (size_t)i * dim), dim * sizeof(T));
    // }
    // in.close();

    T *&vectors = *vectors_ptr;
    int &num = *num_ptr;
    int &dim = *dim_ptr;
    FILE *file_ptr = fopen(data_path.c_str(), "r");
    if (file_ptr != NULL) {
      fread((char *)&dim, sizeof(char), 4, file_ptr);
      fseek(file_ptr, 0, SEEK_END);
      size_t file_size = ftell(file_ptr);
      int total_num = file_size / (size_t)(4 + dim * sizeof(T));
      cerr << file_size << " " << total_num << " " << dim << endl;
      num = total_num;
      rewind(file_ptr);
      vectors = new T[(size_t)num * (size_t)dim];
      int tmp;
      for (int i = 0; i < num; i++) {
        fread((char *)&tmp, sizeof(char), 4, file_ptr);
        fread((char *)(vectors + (size_t)i * dim), sizeof(char),
               dim * sizeof(T), file_ptr);
      }
    } else {
      fclose(file_ptr);
      cerr << "Open " << data_path << " failed." << endl;
      exit(-1);
    }
    fclose(file_ptr);
  }

  template <typename T>
  static void ReadBinaryVecs(const string &data_path, T **vectors_ptr,
                             int *dim_ptr, int begin_pos, int read_num) {
    // T *&vectors = *vectors_ptr;
    // int &dim = *dim_ptr;
    // ifstream in(data_path, ios::binary);
    // if (!in.is_open()) {
    //   cerr << "Can't open " << data_path << endl;
    //   exit(-1);
    // }
    // in.read((char *)&dim, 4);
    // in.seekg(0, ios::end);
    // ios::pos_type file_tail_pos = in.tellg();
    // size_t file_size = (size_t)file_tail_pos;
    // int total_num = file_size / (size_t)(4 + dim * sizeof(T));
    // if (begin_pos + read_num > total_num) {
    //   cerr << data_path << " " << begin_pos << " " << read_num << " "
    //        << total_num << " " << file_size << " " << dim << " " << sizeof(T)
    //        << endl;
    //   assert(begin_pos + read_num <= total_num);
    // }

    // in.seekg((size_t)begin_pos * (dim * sizeof(T) + 4), ios::beg);
    // vectors = new T[(size_t)read_num * (size_t)dim];
    // int tmp = 0;
    // for (int i = 0; i < read_num; i++) {
    //   in.read((char *)&tmp, 4);
    //   in.read((char *)(vectors + (size_t)i * dim), dim * sizeof(T));
    // }
    // in.close();

    T *&vectors = *vectors_ptr;
    int &dim = *dim_ptr;
    FILE *file_ptr = fopen(data_path.c_str(), "r");
    if (file_ptr != NULL) {
      fread((char *)&dim, sizeof(char), 4, file_ptr);
      fseek(file_ptr, 0, SEEK_END);
      fpos_t file_tail_pos;
      fgetpos(file_ptr, &file_tail_pos);
      size_t file_size = (size_t)file_tail_pos.__pos;
      int total_num = file_size / (size_t)(4 + dim * sizeof(T));
      if (begin_pos + read_num > total_num) {
        cerr << data_path << " " << begin_pos << " " << read_num << " "
             << total_num << " " << file_size << " " << dim << " " << sizeof(T)
             << endl;
        assert(begin_pos + read_num <= total_num);
      }

      fseek(file_ptr, (size_t)begin_pos * (dim * sizeof(T) + 4), SEEK_SET);
      vectors = new T[(size_t)read_num * (size_t)dim];
      int tmp;
      for (int i = 0; i < read_num; i++) {
        fread((char *)&tmp, sizeof(char), 4, file_ptr);
        fread((char *)(vectors + (size_t)i * dim), sizeof(char),
              dim * sizeof(T), file_ptr);
      }
    } else {
      fclose(file_ptr);
      cerr << "Open " << data_path << " failed." << endl;
      exit(-1);
    }
    fclose(file_ptr);
  }

  template <typename T>
  static void WriteBinaryVecs(const string &data_path, const T *vectors,
                              const int num, const int dim) {
    FILE *file_ptr = fopen(data_path.c_str(), "w");
    if (file_ptr != NULL) {
      for (int i = 0; i < num; i++) {
        fwrite((char *)&dim, sizeof(char), 4, file_ptr);
        fwrite((char *)(vectors + (size_t)i * dim), sizeof(char),
               dim * sizeof(T), file_ptr);
      }
    } else {
      fclose(file_ptr);
      cerr << "Open " << data_path << " failed." << endl;
      exit(-1);
    }
    fclose(file_ptr);
  }

  template <typename T>
  static void WriteBinaryVecs(const string &data_path, const T *vectors,
                              const int begin_pos, const int write_num,
                              const int dim) {
    // ofstream out(data_path, ofstream::binary | ofstream::in);
    // if (!out.is_open()) {
    //   throw(std::string("Failed to open ") + data_path);
    // }
    // out.seekp(0, ios::end);
    // ios::pos_type file_tail_pos = out.tellp();
    // size_t file_size = (size_t)file_tail_pos;
    // int total_num = file_size / (size_t)(dim * sizeof(T) + 4);
    // assert(begin_pos + write_num <= total_num);

    // out.seekp(begin_pos * (dim * sizeof(T) + 4), ios::beg);
    // for (int i = 0; i < write_num; i++) {
    //   out.write((char *)&dim, 4);
    //   out.write((char *)(vectors + (size_t)i * dim), dim * sizeof(T));
    // }
    // out.close();
    FILE *file_ptr = fopen(data_path.c_str(), "r+");
    if (file_ptr != NULL) {
      fseek(file_ptr, begin_pos * (dim * sizeof(T) + 4), SEEK_SET);
      for (int i = 0; i < write_num; i++) {
        fwrite((char *)&dim, sizeof(char), 4, file_ptr);
        fwrite((char *)(vectors + (size_t)i * dim), sizeof(char),
               dim * sizeof(T), file_ptr);
      }
    } else {
      fclose(file_ptr);
      cerr << "Open " << data_path << " failed." << endl;
      exit(-1);
    }
    fclose(file_ptr);
  }

  static void ReadTxtVecs(const string &data_path, float **vectors_ptr,
                          int *num_ptr, int *dim_ptr,
                          const bool show_process = true) {
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

  static void ReadTxtVecs(float *&vecs, int &size, int &dim,
                          const std::string &data_path,
                          bool show_process = true) {
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

  template <typename T>
  static void WriteTxtVecs(const string &data_path, const T *vectors,
                           const int write_num, const int dim) {
    ofstream out(data_path);
    for (int i = 0; i < write_num; i++) {
      out << dim << '\t';
      for (int j = 0; j < dim; j++) {
        out << vectors[i * dim + j] << '\t';
      }
      out << endl;
    }
    out.close();
  }
};

#endif