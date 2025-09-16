import numpy as np

class defect_belief:
    def __init__(self, pose, id, clss, states):
        """
        state: n x 3 np.array
        """
        self._id = id
        self._cls = clss
        self._pose = pose
        prob = np.full((states.shape[0], 1), 1/states.shape[0])
        self._belief = np.hstack((states, prob))
        self._visited = False

    def sample_belief(self, max=True):
        if max:
            max_row_index = np.argmax(self._belief[:, 3])
            return self._belief[max_row_index][:3]
        else:
            # Randomly sample a value based on the probability
            chosen_index = np.random.choice(len(self._belief), p=self._belief[:,3])
            return self._belief[chosen_index][:3]
    def sample_belief_b(self, max=True):
        if max:
            max_row_index = np.argmax(self._belief[:, 3])
            return self._belief[max_row_index]
        else:
            # Randomly sample a value based on the probability
            chosen_index = np.random.choice(len(self._belief), p=self._belief[:,3])
            return self._belief[chosen_index]
    def setPose(self, pose):
        self._pose = pose 
    def close(self, pose, dist=0.2):
        """ pose: (x,y,z) """
        if euclidean_distance(self._pose, pose) >= dist:
            return True
        return False
    def visited(self):
        self._visited = True




def boundingbox(loc):
    [x,y,w,h] = loc
    x0 = int(x - w/2)
    y0 = int(y - h/2)
    x1 = int(x - w/2)
    y1 = int(y + h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)
    x3 = int(x + w/2)
    y3 = int(y - h/2)
    # print((x0,y0), (x1,y1), (x2,y2), (x3,y3))
    return [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
