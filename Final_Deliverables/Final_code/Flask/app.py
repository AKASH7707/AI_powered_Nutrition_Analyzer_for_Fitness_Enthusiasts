from flask import Flask, render_template, request
import requests

# Flask-It is our framework which we are going to use to run/serve our application.
# request-for accessing file which was uploaded by the user on our application.
import os
import numpy as np  # used for numerical analysis
from tensorflow.keras.models import load_model  # to load our trained model
from tensorflow.keras.preprocessing import image
import requests

app = Flask(__name__, template_folder="template")  # initializing a flask app
# Loading the model
model = load_model('nutrition.h5')


@app.route('/')  # route to display the home page
def home():
    return render_template('home.html')  # rendering the home page


@app.route('/image')
def image1():
    return render_template("image.html")


@app.route('/predict', methods=['GET', 'POST']) # route to show the predictions in a web UI
def launch():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname('__file__')
        filepath = os.path.join(basepath, "uploads", f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(64, 64))  # Loading of the image
        x = image.img_to_array(img)  # image to array
        x = np.expand_dims(x, axis=0)  # changing the shape
        pred = np.argmax(model.predict(x))
        print(pred, model.predict(x))

        op = ['APPLES', 'BANANA', 'ORANGE', 'PINEAPPLE', 'WATERMELON']  # Creating list of output categories
        result = op[pred]
        print(result)
        x = result
        result = nutrition(result)
        print(result)
        return render_template("imageprediction.html", y=(result), x=(x))




def nutrition(index):
    query = index
    api_url = 'https://api.api-ninjas.com/v1/nutrition?query={}'.format(query)
    response = requests.get(api_url, headers={'X-Api-Key': 'yourkey'})
    if response.status_code == requests.codes.ok:
        # Parse the JSON response
        data = response.json()
        if data:
            item = data[0]  # Access the first item in the list (e.g., "banana")
            
            # Extract individual nutritional values
            name = item.get("name", "N/A")
            fat_total = item.get("fat_total_g", "N/A")
            fat_saturated = item.get("fat_saturated_g", "N/A")
            sodium = item.get("sodium_mg", "N/A")
            potassium = item.get("potassium_mg", "N/A")
            cholesterol = item.get("cholesterol_mg", "N/A")
            carbohydrates = item.get("carbohydrates_total_g", "N/A")
            fiber = item.get("fiber_g", "N/A")
            sugar = item.get("sugar_g", "N/A")
            
            # Format the output as a text string
            output = f"""
            Nutritional Information for {name}:
            Total Fat: {fat_total}g, 
            Saturated Fat: {fat_saturated}g, 
            Sodium: {sodium}mg, 
            Potassium: {potassium}mg, 
            Cholesterol: {cholesterol}mg, 
            Carbohydrates: {carbohydrates}g, 
            Fiber: {fiber}g, 
            Sugar: {sugar}g
            """
            return output.strip()  # Strip unnecessary leading/trailing spaces
        else:
            return "No data found for the given query."
    else:
        # If the API call failed, return the error status
        return f"Error: {response.status_code} - {response.text}"
    
if __name__=='__main__':
    app.run()