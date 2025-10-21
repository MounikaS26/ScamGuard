# auth.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from dateutil import parser
import hashlib, uuid, datetime
from gmail_service import send_email

# Blueprint 注册
auth = Blueprint("auth", __name__, template_folder="templates")

# Supabase 客户端将在 app.py 中注入
supabase = None

def hash_password(password: str) -> str:
    """简单哈希函数"""
    return hashlib.sha256(password.encode()).hexdigest()

# ------------------ login ------------------
@auth.route("/login", methods=["GET", "POST"])
def login():
    global supabase
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Please enter both username and password!", "error")
            return render_template("login.html")

        if not supabase:
            flash("Supabase not initialized!", "error")
            return render_template("login.html")

        try:
            # 查询用户
            response = supabase.table("login").select("*").eq("Name", username).execute()
            print("DEBUG: Supabase response data:", response.data)

            if not response.data:
                flash("User does not exist!", "error")
                print(f"DEBUG: User '{username}' not found in Supabase.")
                return render_template("login.html")

            user = response.data[0]
            entered_hash = hash_password(password)
            print(f"DEBUG: Entered password hash: {entered_hash}")
            print(f"DEBUG: Stored password hash: {user.get('Password')}")

            if entered_hash == user.get("Password"):
                session["user_id"] = user["id"]
                session["username"] = user["Name"]
                session["permission_level"] = int(user.get("Permission_level", 0))  
                flash(f"Welcome back, {user['Name']}!", "success")
                print(f"DEBUG: Login successful for user '{username}'")
                return redirect(url_for("index"))  
            else:
                flash("Incorrect password!", "error")
                print(f"DEBUG: Password mismatch for user '{username}'")

        except Exception as e:
            flash(f"Login error: {e}", "error")
            print(f"DEBUG: Exception during login: {e}")

    return render_template("login.html")


# ------------------ register ------------------
@auth.route("/register", methods=["GET", "POST"])
def register():
    global supabase
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm = request.form.get("confirm_password", "").strip()
        email = request.form.get("email", "").strip()

        if not username or not password or not email:
            flash("All fields required!", "error")
        elif password != confirm:
            flash("Passwords do not match!", "error")
        else:
            if not supabase:
                flash("Supabase not initialized!", "error")
                return render_template("register.html")
            try:
                existing = supabase.table("login").select("*").eq("Name", username).execute()
                print("DEBUG: Supabase existing user check:", existing.data)

                if existing.data:
                    flash("Username already exists!", "error")
                    return render_template("register.html")

                supabase.table("login").insert({
                    "Name": username,
                    "Email": email,
                    "Password": hash_password(password),
                    "Permission_level": 0
                }).execute()
                flash("Registration successful! Please log in.", "success")
                print(f"DEBUG: New user '{username}' registered successfully.")
                return redirect(url_for("auth.login"))

            except Exception as e:
                flash(f"Registration error: {e}", "error")
                print(f"DEBUG: Exception during registration: {e}")

    return render_template("register.html")


# ------------------ Forgot Password ------------------
@auth.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    global supabase
    if request.method == "POST":
        email = request.form.get("email", "").strip()

        if not email:
            flash("Please enter your email address!", "error")
            return render_template("forgot_password.html")

        if not supabase:
            flash("Supabase not initialized!", "error")
            return render_template("forgot_password.html")

        try:
            # 查询用户是否存在
            user_check = supabase.table("login").select("id, Name").eq("Email", email).execute()
            if not user_check.data:
                flash("This email is not registered with any account.", "error")
                return render_template("forgot_password.html")

            user = user_check.data[0]
            reset_token = str(uuid.uuid4())
            reset_expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)

            # 删除旧 token 并插入新 token
            supabase.table("password_resets").delete().eq("email", email).execute()
            supabase.table("password_resets").insert({
                "email": email,
                "token": reset_token,
                "expires_at": reset_expiry.isoformat()
            }).execute()

            # 构造重置链接
            reset_link = url_for('auth.reset_password', token=reset_token, _external=True)
            email_body = f"""
            <h3>ScamGuard Password Reset</h3>
            <p>Hello {user.get('Name', 'User')},</p>
            <p>Click the link below to reset your password:</p>
            <a href="{reset_link}">{reset_link}</a>
            <p>This link will expire in 24 hours.</p>
            """
            try:
                send_email(email, "ScamGuard Password Reset", email_body)
                flash("Password reset link has been sent to your email!", "success")
            except Exception as e:
                flash(f"Failed to send email: {e}", "error")

        except Exception as e:
            flash(f"Error: {e}", "error")

    return render_template("forgot_password.html")


# ------------------ Reset Password ------------------
@auth.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    global supabase
    if not supabase:
        flash("Password reset not available in demo mode", "error")
        return redirect(url_for("auth.forgot_password"))

    try:
        token_check = supabase.table("password_resets").select("*").eq("token", token).execute()
        if not token_check.data:
            flash("Invalid or expired reset token!", "error")
            return redirect(url_for("auth.forgot_password"))

        reset_record = token_check.data[0]
        email = reset_record['email']
        expiry = parser.isoparse(reset_record['expires_at'])
        now = datetime.datetime.now(datetime.timezone.utc)

        if now > expiry:
            supabase.table("password_resets").delete().eq("token", token).execute()
            flash("Reset token expired!", "error")
            return redirect(url_for("auth.forgot_password"))

        if request.method == "POST":
            new_password = request.form.get("new_password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()

            if not new_password or not confirm_password:
                flash("Please fill in all fields!", "error")
            elif new_password != confirm_password:
                flash("Passwords do not match!", "error")
            else:
                user_data = supabase.table("login").select("id").eq("Email", email).execute()
                if not user_data.data:
                    flash("User not found!", "error")
                    return render_template("reset_password.html", token=token)

                user_id = user_data.data[0]['id']
                supabase.table("login").update({"Password": hash_password(new_password)}).eq("id", user_id).execute()
                supabase.table("password_resets").delete().eq("token", token).execute()
                flash("Password reset successfully!", "success")
                return redirect(url_for("auth.login"))

        return render_template("reset_password.html", token=token)

    except Exception as e:
        flash(f"Error: {e}", "error")
        return redirect(url_for("auth.forgot_password"))


# ------------------ logout ------------------
@auth.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully!", "info")
    return redirect(url_for("auth.login"))  
