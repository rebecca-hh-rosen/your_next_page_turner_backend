import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rake_nltk import Rake
import string


# Scraping / Cleaning Functions

def clean_row(row):
    '''cleans description'''
    # Cleaning: get rid of punctuations in descriptions
    row.description = row.description.replace('”','')
    row.description = row.description.replace('“','')
    
    for c in string.punctuation:
        row.description = row.description.replace(c,"")

    # Cleaning: get rid of digits in descriptions
    for s in string.digits:
        row.description = row.description.replace(s,"")

    return row['description']


def make_keywords(row):
    '''
    makes keywords
    '''
    plot = row['description']
    # instantiating Rake, by default is uses english stopwords from NLTK
    # and discard all puntuation characters
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(plot)

    # getting the dictionary with key words and their scores
    key_words_dict_scores = r.get_word_degrees()
    
    # return the key words
    return list(key_words_dict_scores.keys())


def make_BoD(row):
    '''makes bod'''
    words = ''
    colums = row.keys().tolist()
    for col in colums:
        words = words + str(row[col]) + ' '
    return words



def clean_create_BoG(row):
    ''' clean up row (can be a row or dictionary with key-value pairs) and add bag of description '''
    
    our_row = row.copy()
    
    # clean description
    our_row['description'] = clean_row(row)

    # assigning the key words to the new column
    our_row['Key_words'] = make_keywords(our_row)
    
    # make bag of description
    our_row['bag_of_description'] = make_BoD(our_row) 
    
    return our_row
    







# Most simple rec system

def simple_rec(genre, length, popularity, dict, df):
    ''' 
    use genre_id_dict and df_simple
    '''
    poss_books = dict[genre]
    df = df.loc[poss_books]
    
    return filter_df(length, popularity, df)
    


# Rec system using only description

def get_recommendations(title, dff):
    '''
    Takes in a title and dataframe (use dff), then makes an abbreviated df containing only the titles and index number. 
    Returns top 10 similar books based on cosine similarity of vectorized description ALONE.
    '''
    title = find_title(title, dff)

    # tfidf vectorize descriptions in dff
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(dff['bag_of_description'])

    # get dot product of tfidf matrix on the transposition of itself to get the cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # create a new dataframe with titles as index, and index as a feature
    indices = pd.Series(list(range(len(dff))), index=dff.index)
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]

    return dff.loc[indices[movie_indices].index][:11]




# helper functions for interfacing with the final recommendation system

def fail_to_find(df):
    final = input("That title did not match any of our books! Please try again, or enter 'quit!' to stop playing.")
    if final == 'quit!':
        return 0
    else:
        return find_title(final, df)
        
def find_title(guess, df):
    guess = guess.lower()
    final = []
    titles_list = {x.lower(): x for x in df.index}
    for possible in list(titles_list.keys()):
        if guess in possible:
               final.append(possible)
    if len(final) == 0:
        return fail_to_find(df)
    if len(final) == 1:
        print (f"\n Great! Looking for recomendations for the book: {titles_list[final[0]]}")
        return titles_list[final[0]]
    elif len(final) > 1:
        maybe = input(f"We found {len(final)} books that matched your search! Would you like to look thru them? If so enter'yes', otherwise enter 'no'.")
        if maybe == 'yes':
            print ("Is your book in this list? \n")
            maybe = input(f"{final}\n")
        for poss in final:
            end = input(f"Is your book {titles_list[poss]}? If so enter 'yes' and if not enter 'no'.")
            if end == 'yes':
                print (f"\n Great! Looking for recomendations for the book: {titles_list[poss]}")
                return titles_list[poss]
        return fail_to_find(df)
                     
                      
                      
                      
# Filter helper functions
                      
def return_pop_df(popularity, df):
    '''
    returns population filtered dataframe - options are:
        deep cut: < 27,000
        well known: between 80,000 and 27,000
        super popular: > 80,000
    '''
    if popularity == 'deep cut':
        return df[df['num_ratings'] < 27000]
    if popularity == 'well known':
        return df[(df['num_ratings'] < 80000) & (df['num_ratings'] > 27000)]
    if popularity == 'super popular':
        return df[df['num_ratings'] > 80000]
    
def filter_df(length, popularity, df):
    '''
    returns length and popularity filtered dataframe - options are:
        long: >= 350
        short: < 350
    '''
    if length != None:
        if length == 'long':
            df = df[(df['pages'] >= 350)]
        elif length == 'short':
            df = df[(df['pages'] < 350)]
        
    if popularity != None:
        df = return_pop_df(popularity, df)

    return df



# Final recommendation system WITHOUT filter                    
                      
def recommendations(title, df, filter_args=(None,None), list_length=11, suppress=False):
    '''
    Return recommendations based on a count vectorized BoW comprised of book author, genres and description.
    Takes in title, list length, a dataframe, a similarity matrix and an option to suppress output.
    filter_args is (length, popularity) : see filte
    '''
    
    recommended_books = []
    
    # tfidf vectorize descriptions in dff
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(df['description'])
    
    # get dot product of tfidf matrix on the transposition of itself to get the cosine similarity
    sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # creating a Series for the movie titles so they are associated to an ordered numerical list
    indices = pd.Series(list(range(len(df))), index=df.index)
    
    # getting the index of the book that matches the title
    idx = indices[title]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:list_length+1].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_books.append(list(df.index)[i])
    
    if suppress == False:
        print (f"\n We recommend \n {recommended_books}")
        
    if filter_args != (None,None):
        df = filter_df(filter_args[0],filter_args[1],df)

    return df.loc[recommended_books].titles.tolist()    
                      
