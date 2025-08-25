from flask import Flask, render_template, request
from recommender import recommend
import pickle

app = Flask(__name__)

movies = pickle.load(open('movies.pkl', 'rb'))

@app.route('/')
def index():
    movie_list = movies['title'].values
    return render_template('index.html', movies=movie_list)

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    movie = request.form['movie']
    recommendations = recommend(movie)
    return render_template('recommend.html', movie=movie, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
