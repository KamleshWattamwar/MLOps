import pandas as pd
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


VERSION      = "v1.0"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

df = pd.read_csv(r"C:\Users\Kamlesh\Downloads\iris.csv")


X = df[["sepal_length", "sepal_width", "petal_width"]]
y = df["petal_length"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

model = LinearRegression()
model.fit(X_train, y_train)



preds = model.predict(X_test)

#print(f"Version: {VERSION} | test_size={TEST_SIZE} | random_state={RANDOM_STATE}")
print(f"MSE: {mean_squared_error(y_test, preds):.4f} | R2: {r2_score(y_test, preds):.4f}")

