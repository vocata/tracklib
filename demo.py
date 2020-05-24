import time
import numpy as np
import tracklib as tlb
from tracklib.tracker import JPDA_events, JPDA_clusters


def main():
    # valid_mat = np.array([[1, 1, 0], [1, 1, 1], [1, 0, 1]])
    # valid_mat = np.array([[1, 1, 1, 1, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 0, 1],
    #                       [1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0],
    #                       [1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 1, 1, 0, 1],
    #                       [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0],
    #                       [1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1],
    #                       [1, 1, 1, 1, 0, 1, 1, 1]])
    # valid_mat = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0],
    #                       [0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1]])
    valid_mat = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=bool)
    start = time.time()
    events_set = JPDA_events(valid_mat)
    end = time.time()
    print('time: %f' % (end - start))
    print('the number of events %d' % len(events_set))
    for e in events_set:
        print(e, end='\n\n')

    start = time.time()
    clusters_set = JPDA_clusters(valid_mat)
    end = time.time()
    print('time: %f' % (end - start))
    print('the number of clusters %d' % len(clusters_set[0]))
    print(clusters_set)


if __name__ == '__main__':
    main()