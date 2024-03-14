import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


housing_df= pd.read_csv("./miami-housing.csv",sep=';')

#Графики
average_price_per_district = housing_df.groupby("Качество_строения")["Цена"].mean()
plt.barh(average_price_per_district.index,average_price_per_district)
plt.ylabel("Качество_строения")
plt.xlabel("Цена")
plt.title("Barh Plot of Цена vs Качество_строения")
#plt.show()

average_price_per_district = housing_df.groupby("Месяц_продажи")["Цена"].mean()
plt.barh(average_price_per_district.index,average_price_per_district)
plt.yticks(np.arange(1, 12+1, 1.0))
plt.ylabel("Месяц_продажи")
plt.xlabel("Цена")
plt.title("Barh Plot of Цена vs Месяц_продажи")
#plt.show()

average_price_per_district = housing_df.groupby("Шум_от_самолетов")["Цена"].mean()
plt.barh(average_price_per_district.index,average_price_per_district)
plt.yticks(np.arange(0, 1+1, 1.0))
plt.ylabel("Шум_от_самолетов")
plt.xlabel("Цена")
plt.title("Barh Plot of Цена vs Шум_от_самолетов")
#plt.show()

average_price_per_district = housing_df.groupby("Возраст_строения")["Цена"].mean()
plt.barh(average_price_per_district.index,average_price_per_district)
plt.ylabel("Возраст_строения")
plt.xlabel("Цена")
plt.title("Barh Plot of Цена vs Возраст_строения")
#plt.show()

plt.scatter(housing_df["Цена"], housing_df["Площадь_участка"])
plt.xlabel("Цена")
plt.ylabel("Площадь_участка")
plt.title("Scatter Plot of Цена vs Площадь_участка")
#plt.show()

plt.scatter(housing_df["Цена"], housing_df["Площадь_помещения"])
plt.xlabel("Цена")
plt.ylabel("Площадь_помещения")
plt.title("Scatter Plot of Цена vs Площадь_помещения")
#plt.show()

plt.scatter(housing_df["Цена"], housing_df["Стоимость_специальных_возможностей"])
plt.xlabel("Цена")
plt.ylabel("Стоимость_специальных_возможностей")
plt.title("Scatter Plot of Цена vs Стоимость_специальных_возможностей")
#plt.show()

#Удаление выбросов
z_threshold = 5
z_scores = stats.zscore(housing_df)
outlier_rows = (z_scores > z_threshold).any(axis=1)
cleaned_df = housing_df[~outlier_rows]

cleaned_df = cleaned_df[cleaned_df["Цена"] > 0]
X = cleaned_df.iloc[:,1:]
y = cleaned_df['Цена']

#Проверка наличия выбросов на графике
plt.scatter(cleaned_df["Цена"], cleaned_df["Площадь_участка"])
plt.xlabel("Цена")
plt.ylabel("Площадь_участка")
plt.title("Scatter Plot of Цена vs Площадь_участка")
#plt.show()

#Построение корреляционной матрицы
corr = cleaned_df.corr()
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.9, vmin=-.9, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
#plt.show()

#Тренировка модели RandomForest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
random_forest_model = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42)
random_forest_model.fit(X_train, y_train)

#Проверка модели через метрики
y_pred = random_forest_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'(MAPE): {mape}')
print(f'(R^2 Score): {r2}')

"""
(MAPE): 0.11608063678004026
(R^2 Score): 0.897781229949394
"""