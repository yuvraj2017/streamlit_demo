import pickle
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [2, 3, 5, 8, 11]

model = LinearRegression()
model.fit(X, y)

# Save the trained model using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
