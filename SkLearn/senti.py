import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("D:\\TUTS\\Annalytica\\train_tweets.csv") 
train_data.head()


def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", "",tweet.lower()).split(),x_train, x_test, y_train, y_test = train_test_split(train_data["processed_tweets"],train_data["label"], test_size = 0.2, random_state = 3)
)

x_train, x_test, y_train, y_test = train_test_split(train_data["processed_tweets"],train_data["label"], test_size = 0.2, random_state = 3)
count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)

## for transforming the 80% of the train data ##
X_train_counts = count_vect.fit_transform(X_train)
X_train_tfidf = transformer.fit_transform(X_train_counts)


## for transforming the 20% of the train data which is being used for testing ##
x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)

model = RandomForestClassifier(n_estimators=200)
model.fit(x_train_tfidf,y_train)