from flask import Flask

app = Flask(__name__)

@app.route('/')
def greeting():
    return "Skin detective API."


@app.route('/healthcheck')
def healthcheck():
    return "API is alive."

if __name__ == "__main__":
    app.run()
