from flask import Flask, render_template, request
import pandas as pd
import json
import plotly
import plotly.express as px


app = Flask(__name__)




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chart1')
def chart1():
    df = pd.DataFrame({
        "Fruit": ["Angry", "Disgust", "Fear","Happy","Neutral","Sad","Surprise"],
        "Amount": [4, 100, 200,300,200,500,1],
        "City": ["SF", "SF", "SF","SF", "SF", "SF","SF"]
    })

    fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header = "Fruit in North America"
    description = """
    A academic study of the number of apples, oranges and bananas in the cities of
    San Francisco and Montreal would probably not come up with this chart.
    """
    return render_template('analyze.html', graphJSON=graphJSON, header=header, description=description)


@app.route("/chart2", methods=['GET', 'POST'])
def chart2():
    if request.method == 'POST':
        if request.form.get('action1') == 'VALUE1':
            pass  # do something
        elif request.form.get('action2') == 'VALUE2':
            pass  # do something else
        else:
            pass  # unknown
    elif request.method == 'GET':
        return render_template('dataset.html')

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)