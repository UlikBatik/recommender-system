from flask import jsonify
from services.cbf_sql import *
from db import *
from collections import OrderedDict

def configure_routes(app):
    @app.route('/')
    def home():
        message = "Welcome to our machine learning model endpoints."
        return jsonify(message)
    
    @app.route('/recommend/<user_id>', methods=['POST'])
    def recommend_post(user_id):
        try:
            database = get_mysql_engine()
            recommender = ContentBasedRecommender(database)
            recommendations = recommender.get_recommendations(user_id)
        
            response = OrderedDict()
            response["status"] = True
            response["message"] = "This is your post recommendations"
            response["result"] = recommendations
        
            return jsonify(response)
        except Exception as err:
            error_response = OrderedDict()
            error_response["status"] = False
            error_response["message"] = "An error occurred"
            error_response["err"] = str(err)
        
            return jsonify(error_response), 500
    