import pandas as pd

salary = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')


y = salary['Salary']

X = salary[['Experience Years']]

# Step 4 : train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

# check shape of train and test sample
X_train.shape, X_test.shape, y_train.shape, y_test.shape


from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Step 6 : train or fit model
model.fit(X_train,y_train)

print(model.intercept_)
print("\n")
print(model.coef_)
print("\n")
# Step 7 : predict model
y_pred = model.predict(X_test)

print(y_pred)

print("\n")
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score

print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))
print("\n")
print("mean_absolute_percentage_error:",mean_absolute_percentage_error(y_test,y_pred))
print("\n")
print("ean_squared_error:",mean_squared_error(y_test,y_pred))
print("r2_square:",r2_score(y_test,y_pred))



