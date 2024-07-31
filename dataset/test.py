import pickle

# Load the data from the pickle file
with open('2013.pickle', 'rb') as file:
    data = pickle.load(file)

for _ in data:
    print(_, "", data[_].size)