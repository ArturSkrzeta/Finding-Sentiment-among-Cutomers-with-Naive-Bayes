import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def get_data():
    df = pd.read_json('data.json')
    review_txt = df['review_text'].str[3:]
    review_rate = df['review_rating'].str[0]
    review_rate.loc[(review_rate == '5') | (review_rate == '4')] = 'Positive'
    review_rate.loc[(review_rate == '3')] = 'Neutral'
    review_rate.loc[(review_rate == '2') | (review_rate == '1')] = 'Negative'
    return review_txt, review_rate

def main():

    # Getting and splitting data
    txts, rates = get_data()
    X_train, X_test, y_train, y_test = train_test_split(txts, rates, test_size=0.2, random_state=1234)

    # Bag of words
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Building Model
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)

    # Making prediction on new data
    new_txt = ['This phone sucks!','Nice phone :)']
    new_txt_vetorized = vectorizer.transform(new_txt)
    prediction = model.predict(new_txt_vetorized)

    print(prediction)

if __name__ == '__main__':
    main()
