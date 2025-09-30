from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_input = request.form["job_text"]
        # PLACE HOLDER FOR NOW
        result = f"Received input of length {len(user_input)} characters."
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)