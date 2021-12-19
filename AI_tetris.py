
import matris
import numpy as np
matris.start_game()
best_weights = np.array([-2.66708729, -0.50036617,  0.7458647,  -2.41497253,  1.56765042,
   -0.7979284,  -3.85126497, -5.62826254, -0.39802289])
matris.start_round_GA(best_weights, [0,0,0,0])