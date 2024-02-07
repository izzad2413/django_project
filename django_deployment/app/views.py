from django.shortcuts import render
import pickle
import pandas as pd

def home(request):
    return render(request, 'index.html')

def cats_transformer(df):
    # get the categorical features from the saved model
    model_cat_cols = pickle.load(open('./../models/all_cats_ohe.pkl', 'rb'))

    # to only check these define categorical features
    cat_cols = ['airline', 'classes', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city']

    # transform the input data
    df_processed = pd.get_dummies(df, columns=cat_cols)
    new_dict = {}
    for i in model_cat_cols:
        if i in df_processed.columns:
            new_dict[i] = df_processed[i].values
        else:
            new_dict[i] = 0
    new_df = pd.DataFrame(new_dict)
    return new_df

def get_predictions(process_input):
    # load the saved models
    model = pickle.load(open('./../models/lr_model.pkl', 'rb'))
    scaler = pickle.load(open('./../models/scaler.pkl', 'rb'))

    # normalized the input
    scaled_input = scaler.transform(process_input)

    # predict the input
    prediction = model.predict(scaled_input)

    # return the predicted value
    predicted = int(prediction)

    # to check result appear  at the terminal
    print(f'The predicted price is {predicted}')
    return predicted

# getting new input from app
def result(request):
    # convert input values to the correct types
    airline = request.GET['airline']
    classes = request.GET['classes']
    source_city = request.GET['source_city']
    departure_time = request.GET['departure_time']
    stops = request.GET['stops']
    arrival_time = request.GET['arrival_time']
    destination_city = request.GET['destination_city']
    duration = float(request.GET['duration'])
    days_left = int(request.GET['days_left'])

    # store input data as dictionary
    input_dict = {
        'airline': [airline],
        'classes': [classes],
        'source_city': [source_city],
        'departure_time': [departure_time],
        'stops': [stops],
        'arrival_time': [arrival_time],
        'destination_city': [destination_city],
        'duration': [duration],
        'days_left': [days_left],
    }

    # input data store as dataframe
    df = pd.DataFrame(input_dict, index=[0])
    # transform data
    process_input = cats_transformer(df)
    # predict data
    result = get_predictions(process_input)
    # return the predicted value
    return render(request, 'result.html', {'result': result})