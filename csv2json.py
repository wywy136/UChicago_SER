import csv
import os
import numpy as np
import random
import json


csv_file = "/project/graziul/ra/team_ser/nosilence_slice_dictionary.csv"


data = []
with open(csv_file) as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    for row in csv_reader:
        json_data = {}
        for i in range(12):
            json_data[header[i+1]] = row[i+1]
        data.append(json_data)

with open("data.json", "w") as f:
    json.dump(data, f)
