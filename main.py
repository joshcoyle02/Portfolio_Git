from flask import Flask, render_template
# from mistune.plugins.table import render_table

app = Flask(__name__)
@app.route('/')
def index():
    user_agent = request.headers.get('User-Agent').lower()

    if 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent:
        return render_template('index-mobile.html')
    else:
        return render_template('index.html')
@app.route('/about')
def about_me():
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

