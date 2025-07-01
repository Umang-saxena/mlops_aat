from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

app = Flask(__name__)
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

PORT_ID = "5000"  # Change this manually per instance

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([data])
    return jsonify({
        'prediction': int(prediction[0]),
        'served_by': f'Instance running on port {PORT_ID}'
    })

if __name__ == '__main__':
    app.run(debug=True, port=int(PORT_ID))
