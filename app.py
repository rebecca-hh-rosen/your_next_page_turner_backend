import json
import pandas as pd
from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions
from flask_cors import CORS

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

app = FlaskAPI(__name__)
CORS(app)


with open('csvjson.json') as json_file:
    data = json.load(json_file)


df = pd.read_csv('df_with_keywords.csv')


@app.route("/books", methods=['GET'])
def notes_list():
    return [data]


@app.route('/books', methods=['POST'])
def returnTitle():
    text = request.json['text']
    text_json = {'data':  text}
    list_of_recs = recommendations(text, df, list_length=11)
    return json.dumps(list_of_recs)


# Final recommendation system WITHOUT filter

def recommendations(title, df, list_length=11, suppress=True):
    '''
    Return recommendations based on a count vectorized BoW comprised of book author, genres and description.
    Takes in title, list length, a dataframe, a similarity matrix and an option to suppress output.
    '''

    recommended_books = []

    # tfidf vectorize descriptions in dff
    tf = TfidfVectorizer(analyzer='word', ngram_range=(
        1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(df['description'])

    # get dot product of tfidf matrix on the transposition of itself to get the cosine similarity
    sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    idx = df[df.titles == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(sim[idx]).sort_values(ascending=False)
    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:list_length+1].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_books.append(list(df.index)[i])

    return df.loc[recommended_books].titles.tolist()
