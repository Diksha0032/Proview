import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,classification_report

data=pd.read_csv("reviews.csv")
#print(data.columns.tolist())

data["reviewText"] = data["reviewText"].fillna("")

x=data['reviewText']
y=(data['overall']>3).astype(int)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

vector=CountVectorizer()
x_train_vectorized=vector.fit_transform(x_train)
x_test_vectorized=vector.transform(x_test)
model=LogisticRegression(max_iter=1000)
model.fit(x_train_vectorized,y_train)
predictions=model.predict(x_test_vectorized)
accuracy=accuracy_score(y_test,predictions)

print("Accuracy:", accuracy)
print(classification_report(y_test,predictions))

model_data = {
    "vocabulary": vector.vocabulary_,
    "coefficients": model.coef_[0].tolist(),
    "intercept": float(model.intercept_[0]),
    "classes": model.classes_.tolist()
}

with open("model.json", "w", encoding="utf-8") as file:
    json.dump(model_data, file)

print("model.json created successfully!")