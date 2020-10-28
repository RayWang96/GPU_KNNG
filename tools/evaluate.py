import knntools
import os
import sys

if (len(sys.argv) != 1):
    recall1 = knntools.evaluate_result(sys.argv[1], sys.argv[2], recall_at=1, result_offset=2, grd_offset=2)
    recall10 = knntools.evaluate_result(sys.argv[1], sys.argv[2], recall_at=10, result_offset=2, grd_offset=2)

print("Top-1:", recall1)
print("Top-10:", recall10)