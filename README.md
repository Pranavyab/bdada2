# Agricultural Yeild Prediction System:

ABOUT THE DATASET:
The crop recommendation dataset, developed by Atharva Ingle, includes crucial agricultural insights that help optimize crop yield and sustainability. This dataset contains 2,200 entries with multiple features relevant to crop growth, including:
Soil Composition: Levels of Nitrogen, Phosphorus, and Potassium.
Environmental Factors: Temperature (in Celsius), Humidity (percentage), pH Value of the soil, and Rainfall (in mm).
Target Variable: The type of crop that can be cultivated based on the above parameters.
By analyzing these features, the dataset enables data-driven decision-making for crop selection, enhancing agricultural productivity and resource management​.

HARDWARE REQUIREMENTS
Computer System:
Processor: At least a dual-core CPU. A quad-core or higher is recommended for faster computations.
RAM: Minimum of 8 GB. 16 GB or more is preferable for handling large datasets.
Storage: At least 50 GB of free disk space to store datasets, models, and dependencies.

SOFTWARE REQUIREMENTS
Operating System:
Windows, macOS, or a Linux distribution (like Ubuntu) can be used.
Python:
Install Python 3.6 or higher. You can download it from the official Python website.
Python Packages:
Use pip to install the necessary packages. ! 
Install required libraries:(code)     
pip install kaggle somoclu pandas numpy scikit-learn matplotlib

IMPLEMENTATION:

Step 1: Set Up Google Colab Environment
Begin by opening Google Colab in the web browser at colab.research.google.com. 
Once the page loads, create a new notebook and sign in using Google account. This cloud-based platform will allow to execute Python code, making it ideal for your agricultural yield prediction project.

Step 2: Install Required Libraries
To work with the dataset and implement the Self-Organizing Map (SOM) and neural networks, install several essential Python libraries. This includes somoclu for creating SOMs, as well as pandas, numpy, scikit-learn, and matplotlib for data manipulation and visualization. Install these libraries by running a specific command in a code cell.

Step 3: Set Up Kaggle API
To access datasets from Kaggle, obtain Kaggle API credentials. Start by signing in to Kaggle and navigating to account settings. Create an API token, which will download a file named kaggle.json. In Google Colab notebook, upload this file to authenticate your Kaggle API access. After uploading, set up the Kaggle API by creating a directory for it, moving the kaggle.json file to the correct location, and ensuring that the file permissions are set appropriately.

Step 4: Download Agricultural Dataset from Kaggle
Next, search for a suitable agricultural dataset, such as the "Crop Yield Prediction Dataset," on Kaggle. Once identified a relevant dataset, download it directly in your Colab environment using the Kaggle API. After the download, receive a zip file containing the dataset. Unzip this file to access the CSV file containing the crop recommendation data.

Step 5: Load and Preprocess the Data
Once the dataset is available, use the pandas library to load it into notebook. You can explore the dataset by inspecting the first few rows to understand its structure. It's essential to check for missing values in the dataset; if any are found, decide how to handle them, either by filling or dropping those values. To prepare the data for modeling, normalize the numerical features using the StandardScaler from scikit-learn. This step ensures that all features are on a similar scale, which is critical for the performance of the models.

Step 6: Implement Self-Organizing Maps (SOM)
After preprocessing the data, use the somoclu library to set up and train a Self-Organizing Map. Configure the SOM with an appropriate grid size, adjusting the number of rows and columns based on the complexity of your dataset. Once the SOM is initialized, train it on the scaled data. After training, visualize the SOM map using the U-matrix to gain insights into how the data points are organized and clustered in the map.

Step 7: Extract SOM Clusters for Yield Prediction
With the SOM successfully trained, extract the best matching units (BMUs) for each data point. These BMUs indicate which grid unit each data point belongs to and help identify clusters in the data that reveal relationships between environmental conditions and crop yields. Understanding these clusters can provide valuable insights into how various factors affect agricultural outcomes.

Step 8: Prepare Data for Neural Network
Finally, prepare the data for training a neural network model for yield prediction. Combine the BMUs extracted from the SOM with the original features to create a new feature set. The target variable will be the crop type or yield. Split this combined dataset into training and testing sets, ensuring that sufficient amount of data for both training the model and evaluating its performance. This step is crucial for developing a robust neural network that can make accurate predictions on unseen data.
Followed by,Encoding steps transforms categorical data into a numerical format, allowing machine learning algorithms to interpret the information correctly. By converting unique string categories into distinct integers, encoding prevents the model from misinterpreting the order of categories. This step enhances the model’s ability to learn from the data, facilitates accurate predictions, and enables efficient calculations during training and evaluation.

Step 9: Build and Train the Neural Network
Import the Necessary Class: Start by importing the MLPRegressor from the sklearn.neural_network module.
Initialize the Neural Network: Create an instance of the MLPRegressor, specifying the structure of the hidden layers, setting the activation function to 'relu', and defining the maximum number of training iterations.
Train the Model: Fit the neural network to the training data by passing the training features and target labels into the fit method.
Make Predictions: Use the trained model to generate predictions on the test dataset.

Step 10: Evaluate the Model
Import Evaluation Metrics: Import the functions needed to evaluate the model's performance.
Calculate Mean Squared Error: Compute the Mean Squared Error (MSE) to quantify the average squared difference between the predicted and actual values.
Calculate R² Score: Compute the R² score to determine the proportion of variance in the target variable explained by the model.
Display Metrics: Print the MSE and R² score to gain insights into the model's performance.

Step 11: Visualize the Results
To gain a better understanding of the model's performance, it's essential to visualize the predicted versus actual yield or crop type. This can be achieved using a plotting library like Matplotlib. Begin by setting up a figure with a specified size to enhance visibility. Then, plot the actual values from the test dataset alongside the predicted values generated by the model, using a distinct line style for differentiation. Include a legend to identify the actual and predicted values, and add a title to the plot to clarify what is being displayed. Finally, display the plot to visually assess how closely the predictions align with the actual values, allowing for an intuitive evaluation of the model's accuracy.

Step 12: Save and Export the Model
After visualizing the results, consider saving the trained models for future use or deployment. This can be accomplished using the joblib library, which allows for efficient serialization of Python objects. First, import the joblib module, then save the Self-Organizing Map (SOM) model to a file, providing a name such as 'som_agriculture_model.pkl'. Next, save the trained neural network model in a similar manner, using a filename like 'nn_yield_prediction_model.pkl'. This step ensures that you can easily retrieve and utilize the models later without needing to retrain them.
