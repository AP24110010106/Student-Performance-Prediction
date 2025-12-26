import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



data = pd.read_csv("student_data.csv", encoding="cp1252")
print(data.head())


print("Dataset Preview:")
print(data.head())


X = data[['StudyHours', 'Attendance', 'InternalMarks']]
y = data['FinalScore']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
new_student = np.array([[6, 80, 70]])  
predicted_score = model.predict(new_student)
print("\nPredicted Final Score for New Student:", predicted_score[0])
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Final Scores")
plt.ylabel("Predicted Final Scores")
plt.title("Actual vs Predicted Student Performance")
plt.show()
