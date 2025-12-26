import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("dataset.csv")

data['result'] = data['result'].map({'FAIL': 0, 'PASS': 1})

# Features and target
X = data[['study_hours', 'attendance', 'internal_marks']]
y = data['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

def get_value(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"⚠️ Enter a value between {min_val} and {max_val}")
        except ValueError:
            print("⚠️ Please enter a valid number")

print("\n--- Student Performance Prediction ---")
print("Enter values within the given range:\n")

study_hours = get_value("Enter study hours (0 – 10): ", 0, 10)
attendance = get_value("Enter attendance percentage (0 – 100): ", 0, 100)
internal_marks = get_value("Enter internal marks (0 – 30): ", 0, 30)

input_data = pd.DataFrame(
    [[study_hours, attendance, internal_marks]],
    columns=['study_hours', 'attendance', 'internal_marks']
)

prediction = model.predict(input_data)

if prediction[0] == 1:
    print("\nPrediction: PASS ✅")
else:
    print("\nPrediction: FAIL ❌")



