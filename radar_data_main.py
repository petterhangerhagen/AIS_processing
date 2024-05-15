import numpy as np
from AutoVerification import Vessel


test_msg = [(1,1), (2,2), (3,3), (4,4), (5,5)]
test_msg = np.array(test_msg)
test_name = 'test'
test_vessel = Vessel(test_name, test_msg)
print(test_vessel.name)