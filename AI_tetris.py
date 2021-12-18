
import matris
import numpy as np
matris.start_game()
best_weights = np.array([-3.6768946,  -2.42059294, -3.7124391,  -1.51863361, -2.44274385,
   -0.24400879, -1.92978995, -3.18480445,  0.71811423])
matris.start_round_GA(best_weights, [0,0,0,0])