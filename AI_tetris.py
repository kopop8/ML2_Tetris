
import matris
import numpy as np
matris.start_game()
best_weights = np.array([-1.41421812,  3.31925469,  1.75633084, -0.68687555, -2.51168949,
   -3.67272167, -2.48923022, -2.98793366, -3.92940647])
print('Score', matris.start_round_GA(best_weights, [0,0,0,0,0,0]))