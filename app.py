## App Utilities
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from keras.models import load_model

from resources.utils import allowed_file, image_classification

global graph,model,sess
graph = tf.get_default_graph()

## App Settings

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = False


@app.route('/', methods=['GET', 'POST'])
def dashboard():
    """Main Dashboard"""
    return render_template("dashboard.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Image Classification"""
    if request.method == 'POST' and 'file' in request.files:
        image = request.files['file']
        # .jpg file extension check
        if allowed_file(image.filename):
            # Apply neural network
            guess = image_classification(image,sess,graph,model)
            print(guess)
            return jsonify({'guess': guess})

        else:
            return jsonify({'error': "Only .jpg files allowed"})
    else:
        return jsonify({'error': "Please upload a .jpg file"})


### ERROR HANDLING


@app.errorhandler(404)
def error404(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def error500(error):
    return render_template('500.html'), 500




## APP INITIATION
if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        model = load_model('model100.h5')

    app.run(host='0.0.0.0', port=5000)
