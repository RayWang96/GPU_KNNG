import numpy as np


def read_data(path, line_start = 0, threshold=400000):
    """
    Read matrix from .txt files to nparray.\n
    The second arg is the threshold of the size of list,
    when the program reach it, list will be add to the row
    of final nparray to reduce the memory consumption.
    """
    tmp_list = []
    res = None
    opened_file = open(path, "r")
    num_elements, dim = list(map(int, opened_file.readline().split()))
    if num_elements < threshold:
        threshold = num_elements // 2
    for i in range(num_elements):
        if i % 10000 == 0:
            print(i)
        if i % threshold == 0 and i != 0:
            if i == threshold:
                res = np.array(tmp_list)
            else:
                res = np.row_stack((res, np.array(tmp_list)))
            tmp_list.clear()
        nums = list(map(float, opened_file.readline().split()))[line_start:]
        tmp_list.append(nums)
    if len(tmp_list) != 0:
        res = np.row_stack((res, np.array(tmp_list)))
    tmp_list.clear()
    opened_file.close()
    return res


def evaluate_result(result_path, ground_truth_path, recall_at=10, result_offset=0, grd_offset=0):
    """
    Return the recall of result.
    """
    opened_result_file = open(result_path, "r")
    opened_ground_truth_file = open(ground_truth_path, "r")
    num_1, dim_1 = list(map(int, opened_result_file.readline().split()))
    num_2, dim_2 = list(map(int, opened_ground_truth_file.readline().split()))
    true_positive, false_negative = 0, 0
    for i in range(num_1):
        line_1 = opened_result_file.readline()
        line_2 = opened_ground_truth_file.readline()
        set_1 = set(
            list(map(int, line_1.split()[:]))[result_offset:recall_at+result_offset])
            # list(map(int, line_1.split()[:]))[result_offset:])
        set_2 = set(
            list(map(int, line_2.split()[:]))[grd_offset:recall_at+grd_offset])
        tmp_cnt = len(set_1 & set_2)
        true_positive += tmp_cnt
        false_negative += recall_at - tmp_cnt
    recall = true_positive / (1.0 * true_positive + false_negative)
    return recall


def zhao2mine(input_path, output_path):
    """
    Format the result.
    """
    in_file = open(input_path, "r")
    out_file = open(output_path, "w")
    num, dim = list(map(int, in_file.readline().split()))
    out_file.write(str(num) + " " + str(dim) + "\n")
    for i in range(num):
        tmp_list = list(map(int, in_file.readline().split()))
        for j in range(len(tmp_list) - 1):
            if j == 1:
                continue
            out_file.write(str(tmp_list[j]) + " ")
        out_file.write("\n")

if __name__ == '__main__':
    zhao2mine("./datasets/sift1m/sift1m_gold_knn40.txt", "./datasets/sift1m/sift1m_gold_knn40_mine.txt")
    #print((525881620.0) / (100000.0 * 99999.0 / 2))