import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask


app = Flask(__name__)

class ContentBasedRecommender:
    def __init__(self, db_connection):
        self.conn = db_connection
        self.vectorizer = TfidfVectorizer()
        self._prepare_data()
    def _prepare_data(self):
        self.user_profiles, self.tfidf_matrix, self.user_post_set, self.unique_posts, self.user_own_posts = self._load_data()
    def _load_data(self):
        query = """
            SELECT l.USERID, l.POSTID, p.BATIKID
            FROM Likes l
            INNER JOIN Post p ON l.POSTID = p.POSTID
            """
        data = pd.read_sql(query, self.conn)
        
        # Create user profiles based on the labels of the posts they have liked
        user_profiles = data.groupby('USERID')['BATIKID'].apply(lambda x: ' '.join(x)).reset_index()
        user_profiles.columns = ['USERID', 'Profile']
        
        # Combine all post labels into a single string for vectorization
        all_labels = data['BATIKID'].unique()
        
        # Fit and transform the post labels to create a TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(all_labels)
        
        # Get a list of all unique post IDs and labels
        unique_posts = data[['POSTID', 'BATIKID']].drop_duplicates()
        
        # Create a set of (user, post) pairs to check which posts a user has already liked
        user_post_set = set(zip(data['USERID'], data['POSTID']))
        
        user_own_posts = set(zip(data['USERID'], data['POSTID']))

        return user_profiles, tfidf_matrix, user_post_set, unique_posts, user_own_posts
    def update_data(self):
        self.user_profiles, self.tfidf_matrix, self.user_post_set, self.unique_posts, self.user_own_posts = self._load_data()
  
    def get_recommendations(self, user_id, top_n=10):
        user_row = self.user_profiles[self.user_profiles['USERID'] == user_id]
        if user_row.empty:
            return f"No data available for USERID {user_id}"
        
        user_profile = user_row['Profile'].values[0]

        # Transform the user's profile using the same TF-IDF vectorizer
        user_tfidf = self.vectorizer.transform([user_profile])

        # Initialize an empty list to store similarity scores for posts the user hasn't liked yet
        similarity_scores = []

        # Iterate over all unique posts
        for post_id, post_label in self.unique_posts.itertuples(index=False):
            if (user_id, post_id) not in self.user_post_set and (user_id, post_id) not in self.user_own_posts:
                # Transform the post label using the same TF-IDF vectorizer
                post_tfidf = self.vectorizer.transform([post_label])

                # Calculate cosine similarity between user profile and post label
                similarity = cosine_similarity(user_tfidf, post_tfidf).flatten()[0]

                # Append the post ID and its similarity score
                similarity_scores.append((post_id, similarity))
            
        # Sort the similarity scores in descending order and get the top N posts
        top_posts = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:top_n]
        top_post_ids = [post_id for post_id, score in top_posts]

        return top_post_ids


