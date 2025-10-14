from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from dotenv import load_dotenv
import hashlib
import uuid
from gmail_service import send_email


# Try importing Supabase
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
    print("✓ Supabase library imported successfully")
except ImportError as e:
    SUPABASE_AVAILABLE = False
    print(f"✗ Failed to import Supabase: {e}")
    print("Please run: python -m pip install supabase")

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-12345')

# Initialize Supabase client
supabase = None
if SUPABASE_AVAILABLE:
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if supabase_url and supabase_key:
        try:
            supabase = create_client(supabase_url, supabase_key)
            print("✓ Supabase client initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize Supabase: {e}")
            supabase = None
    else:
        print("✗ Please set SUPABASE_URL and SUPABASE_KEY in the .env file")
else:
    print("✗ Supabase unavailable, using in-memory storage")

# Simple password hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Fallback in-memory storage (when Supabase is not available)
if not supabase:
    users = {
        'admin': {
            'id': '1', 
            'Name': 'admin', 
            'Password': hash_password('admin123'), 
            'Email': 'admin@example.com',
            'Permission level': 1  # admin privilege
        },
        'test': {
            'id': '2', 
            'Name': 'test', 
            'Password': hash_password('test123'), 
            'Email': 'test@example.com',
            'Permission level': 0  # normal user privilege
        }
    }
    print("✓ Using in-memory user storage")

@app.route("/")
def home():
    if 'user_id' in session:
        return redirect(url_for('detection'))
    return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        if not username or not password:
            flash('Please enter both username and password!', 'error')
            return render_template("login.html")
        
        try:
            if supabase:
                # Query user from Supabase (login table)
                response = supabase.table("login")\
                    .select("id, Name, Password, Email, Permission_level")\
                    .eq("Name", username)\
                    .execute()
                
                if response.data and len(response.data) > 0:
                    user = response.data[0]
                    if user['Password'] == hash_password(password):
                        session['user_id'] = user['id']
                        session['username'] = user['Name']
                        session['permission_level'] = user.get('Permission_level', 0)
                        flash('Login successful!', 'success')
                        print(f"✓ User {username} logged in successfully via Supabase login table")
                        return redirect(url_for('detection'))
                    else:
                        flash('Incorrect password!', 'error')
                else:
                    flash('User does not exist!', 'error')
            else:
                # Use in-memory storage
                if username in users and users[username]['Password'] == hash_password(password):
                    session['user_id'] = users[username]['id']
                    session['username'] = users[username]['Name']
                    session['permission_level'] = users[username].get('Permission level', 0)
                    flash('Login successful!', 'success')
                    print(f"✓ User {username} logged in successfully via in-memory storage")
                    return redirect(url_for('detection'))
                else:
                    flash('Invalid username or password!', 'error')
                    
        except Exception as e:
            print(f"Login error: {str(e)}")
            flash('An error occurred during login, please try again later', 'error')
    
    return render_template("login.html")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        
        if not email:
            flash('Please enter your email address!', 'error')
        else:
            try:
                # Here you would verify the email exists in Supabase
                user_check = supabase.table("login").select("id").eq("Email", email).execute()
                if not user_check.data:
                    flash('If this email is registered, a reset link has been sent.', 'success')
                    return render_template("forgot_password.html")

                # Create reset token (simple example, use a secure token in production)
                reset_token = str(uuid.uuid4())
                reset_link = f"http://localhost:5000/reset-password/{reset_token}"

                # Optionally, save token to Supabase with expiry time
                # supabase.table("password_resets").insert({"email": email, "token": reset_token, "expires": datetime.now() + timedelta(hours=1)}).execute()

                # Send email via Gmail API
                email_body = f"""
                <p>Hello,</p>
                <p>Click the link below to reset your ScamGuard password:</p>
                <p><a href="{reset_link}">{reset_link}</a></p>
                <p>If you did not request this, please ignore this email.</p>
                """
                send_email(email, "ScamGuard Password Reset", email_body)
                flash('If this email is registered, a reset link has been sent.', 'success')

            except Exception as e:
                print(f"Password reset error: {str(e)}")
                flash('An error occurred while sending the reset email, please try again later.', 'error')
    
    return render_template("forgot_password.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        email = request.form.get("email", "").strip()
        
        # Validate input
        if not username or not password or not email:
            flash('All fields are required!', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters long!', 'error')
        elif password != confirm_password:
            flash('Passwords do not match!', 'error')
        else:
            try:
                if supabase:
                    # Check if username already exists
                    user_check = supabase.table("login")\
                        .select("Name")\
                        .eq("Name", username)\
                        .execute()
                    
                    if user_check.data:
                        flash('Username already exists!', 'error')
                        return render_template("register.html")
                    
                    # Check if email already exists
                    email_check = supabase.table("login")\
                        .select("Email")\
                        .eq("Email", email)\
                        .execute()
                    
                    if email_check.data:
                        flash('Email already registered!', 'error')
                        return render_template("register.html")
                    
                    # Create new user (insert into login table)
                    new_user = {
                        "Name": username,
                        "Email": email,
                        "Password": hash_password(password),
                        "Permission_level": 0  # default permission level = normal user
                    }
                    
                    response = supabase.table("login").insert(new_user).execute()
                    
                    if response.data:
                        flash('Registration successful! Please log in.', 'success')
                        print(f"✓ New user {username} registered in Supabase login table")
                        return redirect(url_for('login'))
                    else:
                        flash('Registration failed, please try again later', 'error')
                else:
                    # In-memory storage
                    if username in users:
                        flash('Username already exists!', 'error')
                    else:
                        users[username] = {
                            'id': str(len(users) + 1),
                            'Name': username,
                            'Password': hash_password(password),
                            'Email': email,
                            'Permission level': 0  # default permission level
                        }
                        flash('Registration successful! Please log in.', 'success')
                        print(f"✓ New user {username} registered in in-memory storage")
                        return redirect(url_for('login'))
                        
            except Exception as e:
                print(f"Registration error: {str(e)}")
                flash('An error occurred during registration, please try again later', 'error')
    
    return render_template("register.html")

@app.route("/detection", methods=["GET", "POST"])
def detection():
    # Check if user is logged in
    if 'user_id' not in session:
        flash('Please log in first!', 'error')
        return redirect(url_for('login'))
    
    result = None
    if request.method == "POST":
        user_input = request.form.get("job_text", "")
        # Placeholder detection logic
        result = f"Detection complete! Input text length: {len(user_input)} characters. Estimated scam probability: 25%"
    
    return render_template("detection.html", 
                         username=session['username'], 
                         result=result)

@app.route("/logout")
def logout():
    username = session.get('username', 'Unknown user')
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('permission_level', None)
    flash('You have successfully logged out!', 'success')
    print(f"✓ User {username} logged out")
    return redirect(url_for('login'))

# Test route to verify Supabase connection and login table data
@app.route("/test-supabase")
def test_supabase():
    """Test page showing Supabase connection status and login table data"""
    if not supabase:
        return "❌ Supabase unavailable"
    
    try:
        # Get all users from login table
        response = supabase.table("login").select("*").execute()
        users = response.data
        
        # Build simple HTML output
        html = f"""
        <h1>Supabase Connection Test</h1>
        <p>✅ Supabase connection successful!</p>
        <p>📊 Number of users in login table: {len(users)}</p>
        <h3>User list:</h3>
        <table border="1" style="border-collapse: collapse;">
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Email</th>
                <th>Password Hash</th>
                <th>Permission Level</th>
            </tr>
        """
        
        for user in users:
            html += f"""
            <tr>
                <td>{user.get('id', 'N/A')}</td>
                <td>{user.get('Name', 'N/A')}</td>
                <td>{user.get('Email', 'N/A')}</td>
                <td>{user.get('Password', 'N/A')[:20]}...</td>
                <td>{user.get('Permission_level', 'N/A')}</td>
            </tr>
            """
        
        html += "</table>"
        html += '<p><a href="/">Return to Home</a></p>'
        
        return html
        
    except Exception as e:
        return f"❌ Failed to query Supabase login table: {str(e)}"

if __name__ == "__main__":
    print("=" * 50)
    print("ScamGuard Flask App starting...")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
