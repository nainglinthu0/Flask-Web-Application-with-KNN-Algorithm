from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

knn = joblib.load('knn_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        review_text = request.form['review']

        review_text = review_text.lower() 
        review_text = ''.join([char for char in review_text if char.isalnum() or char == ' '])

        review_vectorized = vectorizer.transform([review_text]).toarray()

        prediction = knn.predict(review_vectorized)[0]


        sentiment = "Positive" if prediction == 1 else "Negative"

        return render_template('output.html', 
                              sentiment=sentiment,
                              name="Naing Lin Thu", 
                              student_id="PIUS20220032")
    except Exception as e:
        return str(e)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        review_text = data['review']

        review_text = review_text.lower() 
        review_text = ''.join([char for char in review_text if char.isalnum() or char == ' ']) 

        review_vectorized = vectorizer.transform([review_text]).toarray()

        prediction = knn.predict(review_vectorized)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"

        return jsonify({
            'sentiment': sentiment,
            'submitted_by': 'Naing Lin Thu (PIUS20220032)'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)