import matris
import numpy as np
import csv
matris.start_game()
# with open('test.csv', "r") as f1:
# 	array = f1.readlines()[-2]
# 	x = array.split(",")
# 	array = x[2]
# 	x = array.split(" ")
# 	data1 = []
# 	for i in x:
# 		data1.append(i.replace(']',''))
# 	data = []
# 	for i in data1:
# 		try:
# 			i = float(i)
# 			if isinstance(float(i), float):
# 				data.append(i)
# 		except:
# 			print(i + "is not a float")
#
#
# best_weights = np.array(data)
best_weights = np.array([-1.41421812,  3.31925469,  1.75633084, -0.68687555, -2.51168949,
   -3.67272167, -2.48923022, -2.98793366, -3.92940647])
print('Score', matris.start_round_GA(best_weights, [0,0,0,0,0,0]))

