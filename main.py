from flask import Flask, render_template, request
import smtplib
import ssl
from flask import redirect, url_for

app = Flask(__name__)

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

@app.route('/contactme', methods=['GET', 'POST'])
def contactme():
    if request.method == 'POST':
        user_email = request.form['email']
        user_message = request.form['message']

        # Your email info
        your_email = "joshuacoylework@gmail.com"
        your_app_password = "laar hpbr cccb cwgy"

        # Proper email format
        email_body = (
            f"From: {user_email}\r\n"
            f"To: {your_email}\r\n"
            f"Subject: New Contact Form Message\r\n\r\n"
            f"{user_message}"
        )

        # Send email to yourself
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(your_email, your_app_password)
            server.sendmail(your_email, your_email, email_body)

        return redirect(url_for('contactme', sent='true'))
    return render_template('contactme.html')

if __name__ == '__main__':
    app.run(debug=True)
