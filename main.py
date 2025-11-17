from flask import Flask, render_template, request
# from mistune.plugins.table import render_table

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

@app.route('/contactme')
def contactme():
    return render_template('contactme.html')


if __name__ == '__main__':
    app.run(debug=True)

