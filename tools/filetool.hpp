#ifndef XMUKNN_FILETOOL_HPP
#define XMUKNN_FILETOOL_HPP
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace xmuknn {
class FileTool {
   public:
    static void GetFilePath(std::string &file_path) {
        //file_path = "D:/KNNDatasets/siftsmall_txt/siftsmall_base.txt";
        //file_path = "D:/KNNDatasets/sift100k/sift100k.txt";
        //file_path = "/mnt/d/Codes/KNN_Demos/datasets/sift1m/sift_base.txt";
        //file_path = "/mnt/d/Codes/KNN_Demos/datasets/rand100k/rand100k4d.txt";
        //file_path = "/mnt/d/Codes/KNN_Demos/datasets/glove100k/glove100k.txt";
        //file_path = "C:/KNNDatasets/gist100k/gist100k_base.txt";
        file_path = "D:/KNNDatasets/rand1m/rand1m100d.txt";
        //file_path = "D:/KNNDatasets/rand10k/rand10k.txt";
    }
    static std::string GetFilePath() {
        std::string file_path;
        //file_path = "D:/KNNDatasets/siftsmall_txt/siftsmall_base.txt";
        //file_path = "D:/KNNDatasets/sift100k/sift100k.txt";
        //file_path = "/mnt/d/Codes/KNN_Demos/datasets/sift1m/sift_base.txt";
        //file_path = "./datasets/sift1m/sift_base.txt";
        //file_path = "/mnt/d/Codes/KNN_Demos/datasets/rand100k/rand100k4d.txt";
        //file_path = "/mnt/d/Codes/KNN_Demos/datasets/glove100k/glove100k.txt";
        //file_path = "C:/KNNDatasets/gist100k/gist100k_base.txt";
        file_path = "D:/KNNDatasets/rand1m/rand1m100d.txt";
        //file_path = "D:/KNNDatasets/rand10k/rand10k.txt";
        return file_path;
    }

    static std::string GetOutPath() {
        return std::string("D:/KNNDatasets/out.txt");
    }
    static void GetOutPath(std::string &out_path) { out_path = "D:/KNNDatasets/out.txt"; }

    static void GetGraphPath(std::string &graph_path) {
        //graph_path =
        //    "D:/KNNDatasets/siftsmall_txt/"
        //    "siftsmall_knn_graph_r.txt";
        
        //graph_path =
        //    "D:/KNNDatasets/sift100k/"
        //    "sift100k_knn_graph_r.txt";

        // graph_path =
        //     "/mnt/d/Codes/KNN_Demos/datasets/sift100k/"
        //     "sift100k_dpg.txt";

        //graph_path = "/mnt/d/Codes/KNN_Demos/datasets/sift1m/sift1m_gold_knn40.txt";

        //graph_path = "/mnt/d/Codes/KNN_Demos/datasets/sift1m/sift1m_dpg2.txt";
        //graph_path = "/mnt/d/Codes/KNN_Demos/datasets/gist100k/gist100k_k40_graph.txt";
        //graph_path = "/mnt/d/Codes/KNN_Demos/datasets/gist100k/gist100k_dpg.txt";
        graph_path = "D:/KNNDatasets/rand1m/rand1m100d_knn_graph_r.txt";
        //graph_path = "/mnt/d/Codes/KNN_Demos/datasets/rand1m/rand1m100d_dpg0.txt";
        //graph_path = "/mnt/d/Codes/KNN_Demos/datasets/rand1m/rand1m100d_dpg2_k20.txt";
        //graph_path = "D:/KNNDatasets/rand10k/rand10k_knn_graph.txt";
    }
    static std::string GetGraphPath() {
        // std::string out_path = "./datasets/sift1m/sift1m_gold_knn40.txt";
        std::string graph_path;
        //graph_path =
        //    "D:/KNNDatasets/siftsmall_txt/"
        //    "siftsmall_knn_graph_r.txt";
        
        //graph_path =
        //    "D:/KNNDatasets/sift100k/"
        //    "sift100k_knn_graph_r.txt";

        // graph_path =
        //     "/mnt/d/Codes/KNN_Demos/datasets/sift100k/"
        //     "sift100k_dpg.txt";
        
        //graph_path = "/mnt/d/Codes/KNN_Demos/datasets/sift1m/sift1m_gold_knn40.txt";

        //graph_path = "/mnt/d/Codes/KNN_Demos/datasets/sift1m/sift1m_dpg2.txt";

        //graph_path = "/mnt/d/Codes/KNN_Demos/datasets/gist100k/gist100k_k40_graph.txt";
        //graph_path = "/mnt/d/Codes/KNN_Demos/datasets/gist100k/gist100k_dpg.txt";
        graph_path = "D:/KNNDatasets/rand1m/rand1m100d_knn_graph_r.txt";
        //graph_path = "/mnt/d/Codes/KNN_Demos/datasets/rand1m/rand1m100d_dpg0.txt";
        //graph_path = "/mnt/d/Codes/KNN_Demos/datasets/rand1m/rand1m100d_dpg2_k20.txt";
        //graph_path = "D:/KNNDatasets/rand10k/rand10k_knn_graph.txt";
        return graph_path;
    }

    static std::string GetQueryPath() {
        std::string query_path;
        //query_path = "D:/KNNDatasets/siftsmall_txt/siftsmall_base.txt";
        //query_path = "D:/KNNDatasets/sift100k/sift_query.txt";
        //query_path = "/mnt/d/Codes/KNN_Demos/datasets/sift1m/sift_query.txt";
        //query_path = "/mnt/d/Codes/KNN_Demos/datasets/rand100k/rand100k4d.txt";
        //query_path = "/mnt/d/Codes/KNN_Demos/datasets/glove100k/glove100k.txt";
        //query_path = "C:/KNNDatasets/gist100k_query.txt";
        query_path = "D:/KNNDatasets/rand1m/rand_qry.txt";
        //query_path = "D:/KNNDatasets/rand10k/rand_qry.txt";
        return query_path;
    }
    static void GetQueryPath(std::string &query_path) {
        //query_path = "D:/KNNDatasets/siftsmall_txt/siftsmall_base.txt";
        //query_path = "D:/KNNDatasets/sift100k/sift_query.txt";
        //query_path = "/mnt/d/Codes/KNN_Demos/datasets/sift1m/sift_query.txt";
        //query_path = "/mnt/d/Codes/KNN_Demos/datasets/rand100k/rand100k4d.txt";
        //query_path = "/mnt/d/Codes/KNN_Demos/datasets/glove100k/glove100k.txt";
        //query_path = "C:/KNNDatasets/gist100k_query.txt";
        query_path = "D:/KNNDatasets/rand1m/rand_qry.txt";
        //query_path = "D:/KNNDatasets/rand10k/rand_qry.txt";
    }

    static void Read2DVector(std::vector<std::vector<int>> &target,
                             const std::string &data_path) {
        std::ifstream in(data_path);
        if (!in.is_open()) {
            throw(std::string("Failed to open ") + data_path);
        }
        int size, dim;
        in >> size >> dim;
        target.resize(size);
        for (int i = 0; i < size; i++) {
            int id, nb_num;
            in >> id >> nb_num;
            target[i].resize(nb_num);
            for (int j = 0; j < nb_num; j++) {
                in >> target[i][j];
            }
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
                if (i % 12345 == 0) 
                    std::cerr << "\r" << i;
                if (i == size - 1)
                    std::cerr << "\r" << size;
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

    static int GetK() { return 100; }
};
}  // namespace xmuknn

#endif