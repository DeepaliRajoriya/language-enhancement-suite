from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

from nlp_engine import process_text, summarize_text


# -------------------------------------------------------
#  APP + DB CONFIG
# -------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = "supersecretkey"      # change to a stronger secret for production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login_page"


# -------------------------------------------------------
#  DATABASE MODELS
# -------------------------------------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(300))


class UserRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    input_text = db.Column(db.Text)
    output_text = db.Column(db.Text)
    task_type = db.Column(db.String(100))
    timestamp = db.Column(db.String(100))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# -------------------------------------------------------
# ROUTES: HOME + AUTH PAGES
# -------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login-page')
def login_page():
    return render_template("login.html")

@app.route('/signup-page')
def signup_page():
    return render_template("signup.html")


# -------------------------------------------------------
# AUTH ROUTES (SIGNUP / LOGIN / LOGOUT)
# -------------------------------------------------------
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not (name and email and password):
        return jsonify({"error": "All fields required"}), 400

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"error": "Email already exists"}), 400

    hashed_pw = generate_password_hash(password, method='sha256')
    new_user = User(name=name, email=email, password=hashed_pw)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "Signup successful"})


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password', '')

    if not (email and password):
        return jsonify({"error": "Both fields required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({"error": "Invalid credentials"}), 401

    login_user(user)
    return jsonify({"message": "Login successful", "user_id": user.id})


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out"})


# -------------------------------------------------------
# NLP ANALYSIS + SAVE HISTORY
# -------------------------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json() or {}

    text = data.get('text', '').strip()
    action = data.get('action', 'check')   # 'check' or 'summarize'
    mode = data.get('mode', 'general')     # 'general' or 'academic'
    tone = data.get('tone', 'neutral')     # 'neutral','academic','formal','direct'

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Base grammar + correction & other stats
    results = process_text(text)

    # 1) Summarization action
    if action == "summarize":
        summary = summarize_text(text, num_sentences=2)
        # keep simple summary in corrected_text
        results["corrected_text"] = summary

    # 2) Grammar check / default action: we already have results['corrected_text'] from process_text
    # (no extra action needed for 'check')

    # 3) Tone conversions (apply if user selected non-neutral tone)
    if tone and tone != "neutral":
        improved = results.get('corrected_text', '')
        # Basic replacements (you can expand or use an LLM later)
        replacements = {
            " can't ": " cannot ",
            " don't ": " do not ",
            " won't ": " will not ",
            " I'm ": " I am ",
            " it's ": " it is ",
            " gonna ": " going to ",
            " wanna ": " want to ",
            " kids ": " children ",
            " guys ": " individuals "
        }
        # do case-insensitive safe replacements by lowercasing a working copy if needed
        # simple approach:
        for old, new in replacements.items():
            improved = improved.replace(old, new)
            improved = improved.replace(old.capitalize(), new.capitalize())
        results["corrected_text"] = improved

    # Build a readable task type for history
    task_text = f"{action}"
    if tone and tone != "neutral":
        task_text += f" / tone={tone}"

    # 4) Save to history if authenticated
    if current_user.is_authenticated:
        entry = UserRequest(
            user_id=current_user.id,
            input_text=text,
            output_text=results.get("corrected_text", ""),
            task_type=task_text,
            timestamp=str(datetime.now())
        )
        db.session.add(entry)
        db.session.commit()

    return jsonify(results)


# -------------------------------------------------------
# USER HISTORY FOR SIDEBAR
# -------------------------------------------------------
@app.route('/history')
@login_required
def history():
    records = UserRequest.query.filter_by(user_id=current_user.id).order_by(UserRequest.id.desc()).limit(200).all()
    output = []
    for r in records:
        output.append({
            "input": r.input_text,
            "output": r.output_text,
            "task": r.task_type,
            "timestamp": r.timestamp
        })
    return jsonify(output)


# -------------------------------------------------------
# RUN APP
# -------------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
