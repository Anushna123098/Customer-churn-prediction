import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

st.set_page_config(page_title="Churn Intelligence", layout="wide")

# ---------- UI ----------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.block-container {padding-top: 0rem !important;}
header, footer {visibility: hidden;}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#eef2ff,#dbeafe);
}

section[data-testid="stSidebar"] {
    background: #e0f2fe;
}
</style>
""", unsafe_allow_html=True)

# ---------- USERS ----------
def load_users():
    if not os.path.exists("users.json"):
        return {}
    return json.load(open("users.json"))

def save_users(u):
    json.dump(u, open("users.json","w"))

users = load_users()

if "page" not in st.session_state:
    st.session_state.page = "login"

# ================= LOGIN =================
if st.session_state.page == "login":

    col1, col2 = st.columns([1,2])

    with col1:
        st.title("🔐 Login")

        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")

        if st.button("Login"):
            if email in users:
                if users[email]["password"] == pw:
                    st.session_state.user = email
                    st.session_state.name = users[email]["name"]
                    st.session_state.page = "upload"
                    st.rerun()
                else:
                    st.error("Incorrect password ❌")
            else:
                st.error("User not found ❌")

        if st.button("Go to Sign Up"):
            st.session_state.page = "signup"
            st.rerun()

# ================= SIGNUP =================
elif st.session_state.page == "signup":

    st.title("📝 Create Account")

    name = st.text_input("Name")
    email = st.text_input("Email")
    pw = st.text_input("Password", type="password")
    confirm_pw = st.text_input("Confirm Password", type="password")

    if st.button("Create Account"):
        if pw != confirm_pw:
            st.error("Passwords do not match ❌")
        elif email in users:
            st.warning("User already exists ⚠️")
        else:
            users[email] = {"password": pw, "name": name}
            save_users(users)
            st.success("Account created ✅")

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# ================= UPDATE PASSWORD =================
elif st.session_state.page == "update_pw":

    st.title("🔑 Update Password")

    new_pw = st.text_input("New Password", type="password")
    confirm_pw = st.text_input("Confirm Password", type="password")

    if st.button("Update Password"):
        if new_pw != confirm_pw:
            st.error("Passwords do not match ❌")
        else:
            users[st.session_state.user]["password"] = new_pw
            save_users(users)
            st.success("Password updated ✅")
            time.sleep(1)
            st.session_state.clear()
            st.session_state.page = "login"
            st.rerun()

# ================= UPLOAD =================
elif st.session_state.page == "upload":

    st.title("📁 Upload Dataset")

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)

        if "Churn" not in df.columns:
            st.error("Dataset must contain 'Churn'")
        else:
            df = df.drop_duplicates()
            df = df.fillna(method="ffill")

            if df["Churn"].dtype == "object":
                df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

            st.session_state.df = df
            st.session_state.page = "dashboard"
            st.rerun()

# ================= DASHBOARD =================
elif st.session_state.page == "dashboard":

    df = st.session_state.df.copy()

    st.sidebar.title("🚀 Churn Intelligence")
    st.sidebar.write(f"👤 {st.session_state.name}")
    st.sidebar.write(f"📧 {st.session_state.user}")

    st.sidebar.markdown("### Filters")

    if "Contract" in df.columns:
        c = st.sidebar.selectbox("Contract", ["All"] + list(df["Contract"].unique()))
        if c != "All":
            df = df[df["Contract"] == c]

    if "PaymentMethod" in df.columns:
        p = st.sidebar.selectbox("Payment", ["All"] + list(df["PaymentMethod"].unique()))
        if p != "All":
            df = df[df["PaymentMethod"] == p]

    st.sidebar.markdown("---")

    nav = st.sidebar.radio("Navigation", ["Dashboard","Prediction","Model Comparison"])

    if st.sidebar.button("Upload New"):
        st.session_state.page = "upload"
        st.rerun()

    if st.sidebar.button("Change Password"):
        st.session_state.page = "update_pw"
        st.rerun()

    if st.sidebar.button("Logout"):
        st.session_state.page = "login"
        st.rerun()

    # ---------- MODEL ----------
    df_model = df.copy()
    le = LabelEncoder()

    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

    X = df_model.drop("Churn", axis=1)
    y = df_model["Churn"]

    model = GradientBoostingClassifier().fit(X,y)
    y_pred = model.predict(X)

    # ================= DASHBOARD =================
    if nav == "Dashboard":

        st.title("📊 Customer Churn Dashboard")

        c1,c2,c3 = st.columns(3)
        c1.metric("Customers", len(df))
        c2.metric("Churn Rate", f"{df['Churn'].mean()*100:.2f}%")
        c3.metric("Accuracy", f"{accuracy_score(y,y_pred)*100:.2f}%")

        st.divider()

        col1,col2 = st.columns(2)

        with col1:
            st.subheader("Churn Distribution")
            st.plotly_chart(px.pie(df, names="Churn", hole=0.6,
                                  color_discrete_sequence=["#3b82f6","#ec4899"]))

        with col2:
            st.subheader("Contract Analysis")
            st.plotly_chart(px.histogram(df, x="Contract", color="Churn",
                                         color_discrete_sequence=["#6366f1","#f472b6"]))

        col3,col4 = st.columns(2)

        with col3:
            st.subheader("Payment Method Analysis")
            st.plotly_chart(px.histogram(df, x="PaymentMethod", color="Churn",
                                         color_discrete_sequence=["#60a5fa","#fbbf24"]))

        with col4:
            st.subheader("Charges vs Churn")
            st.plotly_chart(px.box(df, x="Churn", y="MonthlyCharges"))

        st.subheader("🔥 Feature Importance")

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.plotly_chart(px.bar(importance.head(10),
                               x="Importance", y="Feature",
                               orientation="h",
                               color="Importance",
                               color_continuous_scale="turbo"))

    # ================= PREDICTION =================
    elif nav == "Prediction":

        st.title("🔮 Churn Prediction")

        col1,col2,col3 = st.columns(3)

        tenure = col1.slider("Tenure",1,72,12)
        monthly = col2.number_input("Monthly Charges",50.0)
        total = col3.number_input("Total Charges",500.0)

        if st.button("Predict"):

            sample = X.iloc[0:1].copy()
            sample["tenure"] = tenure
            sample["MonthlyCharges"] = monthly
            sample["TotalCharges"] = total

            prob = model.predict_proba(sample)[0][1]

            colA,colB = st.columns([2,1])

            with colA:
                st.plotly_chart(go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob*100,
                    title={'text': "Churn Risk (%)"},
                    gauge={
                        'axis': {'range':[0,100]},
                        'steps':[
                            {'range':[0,30],'color':"#22c55e"},
                            {'range':[30,70],'color':"#facc15"},
                            {'range':[70,100],'color':"#ef4444"},
                        ]
                    }
                )))

            with colB:
                st.subheader("Risk Level")

                if prob < 0.3:
                    st.success("🟢 Low Risk")
                elif prob < 0.7:
                    st.warning("🟡 Medium Risk")
                else:
                    st.error("🔴 High Risk")

            st.subheader("Prediction Breakdown")

            st.plotly_chart(px.bar(
                pd.DataFrame({"Outcome":["No Churn","Churn"],
                              "Probability":[1-prob,prob]}),
                x="Outcome", y="Probability",
                color="Outcome",
                color_discrete_sequence=["#3b82f6","#ec4899"]
            ))

            st.subheader("💡 Retention Strategy")

            if prob < 0.3:
                
                st.success("✔ Maintain engagement\n"
    "✔ Loyalty rewards\n"
    "✔ Upsell premium features\n"
    "✔ Encourage referrals and reviews")
            elif prob < 0.7:
                st.warning("✔ Offer discounts\n"
        "✔ Improve support\n"
        "✔ Personalized communication\n"
        "✔ Address customer pain points\n"
        "✔ Provide limited-time offers")
            else:
                st.error("✔ Immediate retention\n"
        "✔ Reduce pricing\n"
        "✔ Dedicated support\n"
        "✔ Assign account manager\n"
        "✔ Conduct feedback survey\n"
        "✔ Provide exclusive retention deals")

    # ================= MODEL COMPARISON =================
    else:

        st.title("⚙️ Model Comparison")

        gb = GradientBoostingClassifier().fit(X,y)
        rf = RandomForestClassifier().fit(X,y)
        lr = LogisticRegression(max_iter=1000).fit(X,y)

        st.subheader("Model Accuracy")
        st.write("Gradient Boosting:", accuracy_score(y,gb.predict(X)))
        st.write("Random Forest:", accuracy_score(y,rf.predict(X)))
        st.write("Logistic Regression:", accuracy_score(y,lr.predict(X)))

        st.subheader("Confusion Matrix")
        st.plotly_chart(px.imshow(confusion_matrix(y,gb.predict(X)),
                                  color_continuous_scale="Blues"))

        st.subheader("ROC Curve")

        prob = gb.predict_proba(X)[:,1]
        fpr,tpr,_ = roc_curve(y,prob)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr,y=tpr,name="ROC Curve"))
        st.plotly_chart(fig)