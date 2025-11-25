from flask import Flask, render_template, request
import smtplib
import ssl
from flask import redirect, url_for
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load and preprocess the fighter data
df = pd.read_csv('ufc-fighters-statistics.csv')

def data_preprocess(df):
    try:
        df['total_fights'] = df['wins'] + df['losses'] + df['draws']
        df['win_percentage'] = df.apply(lambda row: (row['wins']) / row['total_fights'] * 100 if row['total_fights'] > 0 else 0, axis=1)

        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], format='%d/%m/%Y').dt.year
        current_year = datetime.now().year

        df['age'] = (current_year - df['date_of_birth'])
        df = df[df['age'] < 45]

        df['losses'] = - df['losses']

        df = df.drop(columns=['date_of_birth','draws','reach_in_cm'])
        return df
    except (KeyError, TypeError) as e:
        print('error')
        return None

fighters_df = data_preprocess(df.copy())
if fighters_df is not None:
    fighters_df = fighters_df.drop(columns=['stance','nickname'])
    fighter_names = sorted(fighters_df['name'].unique())
else:
    fighter_names = []


def fighter_features(fighter_name, fighters_df):
    fighter_features = fighters_df[fighters_df['name'] == fighter_name].drop(columns=['name'])
    if fighter_features.empty:
        return None
    else:
        return fighter_features.iloc[0]

def combine_features(fighter_A, fighter_B, fighters_df):
    features_A = fighter_features(fighter_A, fighters_df)
    features_B = fighter_features(fighter_B, fighters_df)

    if features_A is None or features_B is None:
        return None

    base_features = features_A.index
    
    ordered_columns = [f + '_A' for f in base_features] + [f + '_B' for f in base_features]
    
    combined = pd.concat([features_A.add_suffix('_A'), features_B.add_suffix('_B')]).to_frame().T
    
    return combined[ordered_columns]


@app.route('/')
def index():
    return render_template('index.html')

def is_mobile():
    user_agent = request.headers.get('User-Agent')
    mobile_keywords = ["iphone", "android", "ipad", "mobile", "opera mini", "blackberry"]
    return any(keyword in user_agent.lower() for keyword in mobile_keywords)

@app.route('/about')
def about_me():
    if is_mobile():
        return render_template('aboutme-mobile.html')
    else:
        return render_template('aboutme.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/aspirations')
def aspirations():
    return render_template('aspirations.html')



@app.route('/ufc-predictor', methods=['GET', 'POST'])
def ufc_predictor():
    prediction_result = None
    if request.method == 'POST':
        fighter_A = request.form['fighter_A']
        fighter_B = request.form['fighter_B']

        if fighter_A == fighter_B:
            prediction_result = {'error': 'Please select two different fighters.'}
        else:
            input_df = combine_features(fighter_A, fighter_B, fighters_df)

            if input_df is not None:
                try:
                    input_scaled = scaler.transform(input_df)
                    prediction = model.predict(input_scaled)
                    probabilities = model.predict_proba(input_scaled)
                    confidence = probabilities[0][prediction[0]]
                    winner = fighter_A if prediction == 1 else fighter_B
                    prediction_result = {'winner': winner, 'confidence': f'{confidence * 100:.2f}%'}
                except Exception as e:
                    prediction_result = {'error': f'Error during prediction: {e}'}
            else:
                prediction_result = {'error': 'Could not find one of the fighters.'}

    return render_template('ufc-predictor.html', fighters=fighter_names, prediction=prediction_result)


@app.route('/contactme', methods=['GET', 'POST'])
def contactme():
    if request.method == 'POST':
        user_email = request.form['email']
        user_message = request.form['message']

        # Your email info - REMOVED HARDCODED PASSWORD for security
        your_email = "joshuacoylework@gmail.com"
        your_app_password = "YOUR_APP_PASSWORD" # IMPORTANT: Use environment variables for this

        # Proper email format
        email_body = (
            f"From: {user_email}\r\n"
            f"To: {your_email}\r\n"
            f"Subject: New Contact Form Message\r\n\r\n"
            f"{user_message}"
        )

        # Send email to yourself
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(your_email, your_app_password)
                server.sendmail(your_email, your_email, email_body)
            return redirect(url_for('contactme', sent='true'))
        except Exception as e:
            # You might want to log the error here
            return redirect(url_for('contactme', sent='false', error=str(e)))

    return render_template('contactme.html')

if __name__ == '__main__':
    app.run(debug=True, port=50001)
