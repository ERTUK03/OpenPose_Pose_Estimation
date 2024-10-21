import numpy as np

def score_connection(point_a, point_b, paf_x, paf_y, num_intermediate_points=10):
    y1, x1 = map(float, point_a)
    y2, x2 = map(float, point_b)

    line = np.linspace([y1, x1], [y2, x2], num=num_intermediate_points)

    vec = np.array([y2 - y1, x2 - x1])
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        return 0
    vec /= vec_norm

    paf_scores = 0
    for point in line:
        py, px = int(round(point[0])), int(round(point[1]))
        paf_vector = np.array([paf_y[py, px].item(), paf_x[py, px].item()])
        paf_scores += np.dot(paf_vector, vec)

    paf_scores /= num_intermediate_points

    return paf_scores if paf_scores > 0 else 0
