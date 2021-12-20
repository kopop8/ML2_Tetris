
import matris
import numpy as np
matris.start_game()
best_weights = np.array([-1.26648514, -3.45636673, -0.1197976,  -1.48284111, -1.6757665,
    1.9245446,  -2.64288392, -3.43604258,  1.73746571])
matris.start_round_GA(best_weights, [0,0,0,0])