import json
import re

from scrapper import get_reviews


with open("model.json", "r", encoding="utf-8") as file:
    model = json.load(file)


def tokenize(text):
    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def predict(text):

    vocabulary = model["vocabulary"]
    coefficients = model["coefficients"]

    words = tokenize(text)

    score = model["intercept"]

    for word in words:

        if word in vocabulary:

            index = vocabulary[word]

            score += coefficients[index]

    if score >= 0:
        return "positive"

    return "negative"


url = input("Enter product URL: ")

print("\nFetching reviews...")

reviews = get_reviews(url)

print("\nReviews found:", len(reviews))

if len(reviews) == 0:

    print("No reviews found.")
    exit()


positive = 0
negative = 0

print("\nIndividual predictions:\n")


for i, review in enumerate(reviews, 1):

    sentiment = predict(review)

    if sentiment == "positive":
        positive += 1
    else:
        negative += 1

    print(f"{i}. {sentiment.upper()}")
    print(f"   {review[:150]}...")


total = len(reviews)


positive_percentage = (
    positive / total
) * 100


negative_percentage = (
    negative / total
) * 100


print(
    f"\nPositive: {positive_percentage:.1f}%"
)

print(
    f"Negative: {negative_percentage:.1f}%"
)


if positive_percentage >= 70:

    verdict = "WORTH BUYING"

elif positive_percentage >= 40:

    verdict = "MIXED REVIEWS"

else:

    verdict = "NOT RECOMMENDED"


print(
    "\nOverall:",
    verdict
)


result = {
    "total_reviews": total,
    "positive": positive,
    "negative": negative,
    "positive_percentage": round(
        positive_percentage,
        1
    ),
    "negative_percentage": round(
        negative_percentage,
        1
    ),
    "verdict": verdict
}


with open(
    "frontend/result.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        result,
        file,
        indent=4
    )


print(
    "\nResult saved to website/result.json"
)