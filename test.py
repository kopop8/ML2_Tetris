import csv
with open('scores.csv', "r") as f1:
	array = []
	data = []
	csv_reader = csv.reader(f1)
	for line in csv_reader:
		data.append(line)
	array = data[-2][2]
	array = array.replace('  ', ',')
	array = array.replace(' -',',-')



