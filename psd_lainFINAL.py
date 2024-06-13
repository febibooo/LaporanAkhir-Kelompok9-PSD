import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from PIL import Image
import io
# # Custom CSS for playful design
# st.markdown("""
#     <style>
#     body {
#         background-color: #fff;
#     }
#     .stApp {
#         background-color: #fff;
#         padding: 2rem;
#         border-radius: 10px;
#         box-shadow: 0 4px 8px rgba(0,0,0,0.1);
#     }
#     .css-18e3th9 {
#         padding: 1rem;
#     }
#     .st-c4 {
#         background-color: #ffd1dc !important;
#         color: #000000 !important;
#     }
#     .stButton button {
#         background-color: #f4a7b9;
#         color: #ffffff;
#         border-radius: 5px;
#     }
#     .stButton button:hover {
#         background-color: #e667af;
#         color: #ffffff;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Title and image
# st.title('Mental Health Analysis & Prediction')
# # Add a fun image at the top
# st.image('https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcShudAm0lUkSpWg_vZV4jeCZdCZZfqlhlpDSguNkQMQo5h6lhu_', width=250)

img = Image.open("mental_health.png")
col1, col2 = st.columns(2)
with col1:
    st.image(img)
with col2:
    st.title('Mental Health Analysis & Prediction')

# Load data
@st.cache_data
def load_data(data_path):
    df = pd.read_csv(data_path)
    df.columns = ['Date_Time', 'Gender', 'Age', 'Course', 'Year', 'CGPA', 'Marital_Status', 'Depression', 'Anxiety', 'Panic_Attack', 'Treatment']
    df['Year'] = df['Year'].apply(lambda x: int(x.split(' ')[-1]))
    return df

data_path = "https://raw.githubusercontent.com/jinniekyo/Data-Mental-Health-Analysis/main/data%20mental%20health.csv"
df_raw = load_data(data_path)
df = load_data(data_path)

# Preprocess data
@st.cache_data
def preprocess_data(df):
    def process_cgpa(cgpa):
        if '-' in cgpa:
            low, high = map(float, cgpa.split('-'))
            return (low + high) / 2
        else:
            return float(cgpa)
    
    df['CGPA'] = df['CGPA'].apply(process_cgpa)
    bins = [0, 2, 2.5, 3, 3.5, 4]
    labels = ['0 - 1.99', '2.00 - 2.49', '2.50 - 2.99', '3.00 - 3.49', '3.50 - 4.00']
    df['CGPA_bin'] = pd.cut(df['CGPA'], bins=bins, labels=labels, include_lowest=True)
    
    course_list = {'engin': 'Engineering', 'Engine': 'Engineering', 'Islamic education': 'Islamic Education',
                   'Pendidikan islam': 'Pendidikan Islam', 'BIT': 'IT', 'psychology': 'Psychology', 'koe': 'Koe',
                   'Kirkhs': 'Irkhs', 'KIRKHS': 'Irkhs', 'Benl': 'BENL', 'Fiqh fatwa ': 'Fiqh', 'Laws': 'Law'}
    df['Course'].replace(course_list, inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    df['Treatment'] = df['Treatment'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Panic_Attack'] = df['Panic_Attack'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Depression'] = df['Depression'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Anxiety'] = df['Anxiety'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return df

df = preprocess_data(df)

# Encode categorical variables
def encode_data(df):
    labelencoder = LabelEncoder()
    df['Gender'] = labelencoder.fit_transform(df['Gender'])
    df['Marital_Status'] = labelencoder.fit_transform(df['Marital_Status'])
    df['Depression'] = labelencoder.fit_transform(df['Depression'])
    df['Anxiety'] = labelencoder.fit_transform(df['Anxiety'])
    df['Panic_Attack'] = labelencoder.fit_transform(df['Panic_Attack'])
    df['Treatment'] = labelencoder.fit_transform(df['Treatment'])
    return df

df_encoded = encode_data(df)

# Train models and store in session state
@st.cache_data
def train_model(df, target):
    labelencoder = LabelEncoder()
    df['Gender'] = labelencoder.fit_transform(df['Gender'])
    df['Course'] = labelencoder.fit_transform(df['Course'])
    df['Marital_Status'] = labelencoder.fit_transform(df['Marital_Status'])
    
    features = ['Gender', 'Age', 'Course', 'Year', 'CGPA', 'Marital_Status']
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # RandomForest
    param_dist_rf = {
        'n_estimators': [100],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'bootstrap': [True]
    }
    
    rf = RandomForestClassifier(random_state=42)
    random_search_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_dist_rf, n_iter=1, cv=3, scoring='accuracy', n_jobs=-1)
    random_search_rf.fit(X_train, y_train)
    
    best_rf = random_search_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)
    
    # XGBoost
    xgboost = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8)
    xgboost.fit(X_train, y_train)
    
    y_pred_xgb = xgboost.predict(X_test)
    
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    report_xgb = classification_report(y_test, y_pred_xgb)
    
    return (best_rf, accuracy_rf, report_rf), (xgboost, accuracy_xgb, report_xgb)

# Train models for the Treatment condition
if 'model_rf_treat' not in st.session_state or 'model_xgb_treat' not in st.session_state:
    (model_rf_treat, accuracy_rf_treat, report_rf_treat), (model_xgb_treat, accuracy_xgb_treat, report_xgb_treat) = train_model(df, 'Treatment')
    st.session_state.model_rf_treat = model_rf_treat
    st.session_state.model_xgb_treat = model_xgb_treat
    st.session_state.accuracy_rf_treat = accuracy_rf_treat
    st.session_state.report_rf_treat = report_rf_treat
    st.session_state.accuracy_xgb_treat = accuracy_xgb_treat
    st.session_state.report_xgb_treat = report_xgb_treat

# Sidebar navigation
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Menu", ["Data Overview", "Data Visualization", "Predictive Modeling", "User Input"])

# Data Overview
if menu == "Data Overview":
    st.title("Data Overview")
    st.write(df_raw.head(10))

# Data Visualization
elif menu == "Data Visualization":
    st.title("🪷Data Visualization")
    
    st.subheader("💫 Gender Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Gender', data=df.replace({'Gender': {0: 'Female', 1: 'Male'}}), ax=ax, palette=['#a67675', '#dbd5a4'])
    ax.set_title('Gender Distribution')
    st.pyplot(fig)

    st.subheader("💫 Age Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['Age'], kde=True, ax=ax, color='red', edgecolor='black', alpha=0.5)
    ax.set_title('Age Distribution')
    st.pyplot(fig)
    
    st.subheader("💫 CGPA Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='CGPA_bin', data=df, ax=ax, palette=['#5b0e6c'])
    ax.set_title('CGPA Distribution')
    st.pyplot(fig)

    st.subheader("💫 Course Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(y='Course', data=df, ax=ax, order=df['Course'].value_counts().index)
    ax.set_title('Course Distribution')
    st.pyplot(fig)

    st.subheader("💫 Year Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(y='Year', data=df, ax=ax, order=df['Year'].value_counts().index, edgecolor='pink')
    ax.set_title('Year Distribution')
    st.pyplot(fig)
    st.subheader("💫 Marital_Status")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Marital_Status', data=df_raw, ax=ax, palette=['#5b0e6c', '#a67675'])
    ax.set_title('Marital_Status Distribution')
    st.pyplot(fig)

    st.subheader("💫 Treatment Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Treatment', data=df.replace({'Treatment': {0: 'No', 1: 'Yes'}}), ax=ax, palette=['#a67675', '#dbd5a4'])
    ax.set_title('Treatment Distribution')
    st.pyplot(fig)

    st.subheader("💫 Depression Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Depression', data=df.replace({'Depression': {0: 'No', 1: 'Yes'}}), ax=ax, palette=['#a67675', '#dbd5a4'])
    ax.set_title('Depression Distribution')
    st.pyplot(fig)
    
    st.subheader("💫 Anxiety Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Anxiety', data=df.replace({'Anxiety': {0: 'No', 1: 'Yes'}}), ax=ax, palette=['#a67675', '#dbd5a4'])
    ax.set_title('Anxiety Distribution')
    st.pyplot(fig)

    st.subheader("💫 Panic Attack Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Panic_Attack', data=df.replace({'Panic_Attack': {0: 'No', 1: 'Yes'}}), ax=ax, palette=['#a67675', '#dbd5a4'])
    ax.set_title('Panic Attack Distribution')
    st.pyplot(fig)

    st.subheader("💫 Student Count by Course for Year 1")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='Course', data=df[df['Year'] == 1], ax=ax)
    ax.set_title('Student Count for Year=1')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("💫 Student Count by Course for Year 2")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.countplot(x='Course', data=df[df['Year'] == 2], ax=ax)
    ax.set_title('Student Count for Year=2')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("💫 Student Count by Course for Year 3")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='Course', data=df[df['Year'] == 3], ax=ax)
    ax.set_title('Student Count for Year=3')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("💫 Student Count by Course for Year 4")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='Course', data=df[df['Year'] == 4], ax=ax)
    ax.set_title('Student Count for Year=4')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("💫 Treatment Distribution by Gender")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Gender', hue='Treatment', data=df_raw, ax=ax, palette=['#a67675', '#dbd5a4'])
    ax.set_title('Treatment Distribution by Gender')
    st.pyplot(fig)
    
    st.subheader("💫 Panic Attack Distribution by Gender")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Gender', hue='Panic_Attack', data=df_raw, ax=ax, palette=['#a67675', '#dbd5a4'])
    ax.set_title('Panic Attack Distribution by Gender')
    st.pyplot(fig)

    st.subheader("💫 Anxiety Distribution by Gender")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Gender', hue='Anxiety', data=df_raw, ax=ax, palette=['#a67675', '#dbd5a4'])
    ax.set_title('Anxiety Distribution by Gender')
    st.pyplot(fig)

    st.subheader("💫 Depression Distribution by Gender")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Gender', hue='Depression', data=df_raw, ax=ax, palette=['#a67675', '#dbd5a4'])
    ax.set_title('Depression Distribution by Gender')
    st.pyplot(fig)

    st.subheader("💫 Course vs Treatment")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.stripplot(x="Treatment", y="Course", hue="Gender", data=df.replace({0: "No", 1: "Yes"}), jitter=True, palette=['#a67675', '#dbd5a4'], ax=ax)
    ax.set_title("Course vs Treatment")
    st.pyplot(fig)

    st.subheader("💫 Course vs Depression")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.stripplot(x="Depression", y="Course", hue="Gender", data=df.replace({0: "No", 1: "Yes"}), jitter=True, palette=['#a67675', '#dbd5a4'], ax=ax)
    ax.set_title("Course vs Depression")
    st.pyplot(fig)

    st.subheader("💫 Course vs Anxiety")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.stripplot(x="Anxiety", y="Course", hue="Gender", data=df.replace({0: "No", 1: "Yes"}), jitter=True, palette=['#a67675', '#dbd5a4'], ax=ax)
    ax.set_title("Course vs Anxiety")
    st.pyplot(fig)

    st.subheader("💫 Course vs Panic Attack")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.stripplot(x="Panic_Attack", y="Course", hue="Gender", data=df.replace({0: "No", 1: "Yes"}), jitter=True, palette=['#a67675', '#dbd5a4'], ax=ax)
    ax.set_title("Course vs Panic Attack")
    st.pyplot(fig)

    st.subheader("💫 Count of CGPA by Treatment")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='CGPA', hue='Treatment', data=df_raw, palette=['#a67675', '#dbd5a4'], ax=ax)
    ax.set_title("Count of CGPA by Treatment")
    st.pyplot(fig)

    st.subheader("💫 Count of CGPA by Depression")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='CGPA_bin', hue='Depression', data=df.replace({0: "No", 1: "Yes"}), palette=['#a67675', '#dbd5a4'], ax=ax)
    ax.set_title("Count of CGPA by Depression")
    st.pyplot(fig)

    st.subheader("💫 Count of CGPA by Anxiety")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='CGPA_bin', hue='Anxiety', data=df.replace({0: "No", 1: "Yes"}), palette=['#a67675', '#dbd5a4'], ax=ax)
    ax.set_title("Count of CGPA by Anxiety")
    st.pyplot(fig)

    st.subheader("💫 Count of CGPA by Panic Attack")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='CGPA', hue='Panic_Attack', data=df_raw, palette=['#a67675', '#dbd5a4'], ax=ax)
    ax.set_title("Count of CGPA by Panic Attack")
    st.pyplot(fig)

    st.subheader("💫 Count of Year by Treatment")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Year', hue='Treatment', data=df_raw, palette=['#a67675', '#dbd5a4'], ax=ax)
    ax.set_title("Count of Year by Treatment")
    st.pyplot(fig)

    st.subheader("💫 Count of Year by Anxiety")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Year', hue='Anxiety', data=df_raw, palette=['#a67675', '#dbd5a4'], ax=ax)
    ax.set_title("Count of Year by Anxiety")
    st.pyplot(fig)

    st.subheader("💫 Count of Year by Depression")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Year', hue='Depression', data=df_raw, palette=['#a67675', '#dbd5a4'], ax=ax)
    ax.set_title("Count of Year by Depression")
    st.pyplot(fig)

    st.subheader("💫 Count of Year by Panic Attack")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Year', hue='Panic_Attack', data=df_raw, palette=['#a67675', '#dbd5a4'], ax=ax)
    ax.set_title("Count of Year by Panic Attack")
    st.pyplot(fig)

    st.subheader("💫 CGPA Distribution by Year")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Year', hue='CGPA_bin', data=df, ax=ax)
    ax.set_title('CGPA Distribution by Year')
    st.pyplot(fig)

    st.subheader("💫 Age Distribution by Year")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(df['Age'], label='Age', ax=ax)
    sns.kdeplot(df['Year'], label='Year of Study', ax=ax)
    ax.set_title('Age Distribution by Year')
    ax.legend()
    st.pyplot(fig)

    st.subheader("💫 Gender vs Age")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Gender', hue='Age', data=df_raw, ax=ax)
    ax.set_title('Gender Vs Age')
    st.pyplot(fig)

    st.subheader("💫 Gender vs Course")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(y="Course", hue="Gender", data=df_raw, palette=['#a67675', '#dbd5a4'], order=df_raw['Course'].value_counts().index)
    ax.set_title("Gender vs Course")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("💫 Gender vs Year")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Gender', hue='Year', data=df.replace({'Gender': {0: 'Female', 1: 'Male'}}), ax=ax, palette='pastel')
    ax.set_title('Gender vs Year')
    st.pyplot(fig)

    st.subheader("💫 Gender vs CGPA")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='Gender', y='CGPA', data=df.replace({'Gender': {0: 'Female', 1: 'Male'}}), ax=ax, palette=['#a67675', '#dbd5a4'])
    ax.set_title('Gender vs CGPA')
    st.pyplot(fig)

    st.subheader("💫 Gender vs Marital Status")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Gender', hue='Marital_Status', data=df_raw, ax=ax, palette = 'pastel')
    ax.set_title('Gender Vs Marital_Status')
    st.pyplot(fig)

    st.subheader("💫 Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    selected_columns = ['Gender', 'Age', 'Course', 'Year', 'CGPA', 'Marital_Status', 'Depression', 'Anxiety', 'Panic_Attack', 'Treatment']
    corr = df_encoded[selected_columns].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

    st.subheader("💫 Marital Status Distribution by Course")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(y='Course', hue='Marital_Status', data=df, palette='pastel', order=df['Course'].value_counts().index, ax=ax)
    ax.set_title('Marital Status Distribution by Course')
    st.pyplot(fig)

    st.subheader("💫 Correlation Heatmap of Mental Health Conditions")
    fig, ax = plt.subplots(figsize=(10, 8))
    mental_health_conditions = ['Depression', 'Anxiety', 'Panic_Attack', 'Treatment']
    corr_matrix = df[mental_health_conditions].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap of Mental Health Conditions')
    st.pyplot(fig)

    st.subheader("💫 Pairplot of Key Features")
    key_features = ['Age', 'CGPA', 'Depression', 'Anxiety', 'Panic_Attack', 'Treatment']
    sns.pairplot(df[key_features], diag_kind='kde')
    st.pyplot()

    st.subheader("💫 Barplot of Age Group vs Treatment")
    df['Age_Group'] = pd.cut(df['Age'], bins=[17, 20, 23, 26, 29, 32, 35, 100], labels=['18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36+'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Age_Group', hue='Treatment', data=df, palette='pastel', ax=ax)
    ax.set_title('Age Group vs Treatment')
    st.pyplot(fig)

    st.subheader("💫 Stacked Bar Plot for Depression, Anxiety, Panic Attack by Gender")
    fig, ax = plt.subplots(figsize=(12, 8))
    mental_health_conditions = ['Depression', 'Anxiety', 'Panic_Attack']
    df_melted = df.melt(id_vars='Gender', value_vars=mental_health_conditions, var_name='Condition', value_name='Presence')
    sns.histplot(data=df_melted, x='Condition', hue='Gender', multiple='stack', palette='pastel', shrink=0.8)
    ax.set_title('Stacked Bar Plot for Depression, Anxiety, Panic Attack by Gender')
    st.pyplot(fig)

    st.subheader("💫 CGPA Distribution by Year of Study")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='Year', y='CGPA', data=df, palette='pastel')
    ax.set_title('CGPA Distribution by Year of Study')
    st.pyplot(fig)

    st.subheader("💫 Box Plot of Age by Depression, Anxiety, and Panic Attack")
    fig, ax = plt.subplots(figsize=(12, 8))
    df_melted_age = df.melt(id_vars='Age', value_vars=mental_health_conditions, var_name='Condition', value_name='Presence')
    sns.boxplot(x='Condition', y='Age', hue='Presence', data=df_melted_age, palette='pastel')
    ax.set_title('Box Plot of Age by Depression, Anxiety, and Panic Attack')
    st.pyplot(fig)

    st.subheader("💫 Treatment by Marital Status and Year of Study")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(x='Year', hue='Treatment', data=df, palette='pastel')
    sns.despine()
    ax.set_title('Treatment by Year of Study')
    plt.legend(title='Treatment', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

    st.subheader("💫 Treatment by Marital Status and Year of Study")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(x='Marital_Status', hue='Treatment', data=df, palette='pastel')
    sns.despine()
    ax.set_title('Treatment by Marital Status')
    plt.legend(title='Treatment', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

    st.subheader("💫 Bar Plot of Course vs Treatment with Gender Hue")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(y='Course', hue='Treatment', data=df, palette='pastel', order=df['Course'].value_counts().index)
    ax.set_title('Course vs Treatment with Gender Hue')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("💫 Treatment by Year of Study with Gender Hue")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(x='Year', hue='Treatment', data=df, palette='pastel')
    ax.set_title('Treatment by Year of Study with Gender Hue')
    plt.legend(title='Treatment', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

# Predictive Modeling
elif menu == "Predictive Modeling":
    st.title("🪷Predictive Modeling")
    
    st.subheader("BEST MODEL(Treatment)")
    st.write(f"Accuracy: {st.session_state.accuracy_rf_treat:.2%}")
    st.text(st.session_state.report_rf_treat)
    

# User Input
elif menu == "User Input":
    st.title("🪷User Input")
    st.write("This section allows you to input your own data to see the prediction.")

    gender = st.selectbox('👫 Choose your gender', ('Male', 'Female'))
    age = st.slider('👵🏻 Enter your age', 15, 30, 20)
    course = st.selectbox('👩🏻‍🎓 Choose your course', df['Course'].unique())
    year = st.selectbox('🧑🏻‍💻 Choose your current year of study', df['Year'].unique())
    cgpa = st.slider('💯 Enter your CGPA', 0.0, 4.0, 3.0)
    marital_status = st.selectbox('👰🏻‍♀️ Choose your marital status', ('No', 'Yes'))
    panic_attack = st.selectbox('😱 Have you had a panic attack?', ('Yes', 'No'))
    anxiety = st.selectbox('😰 Do you experience anxiety?', ('Yes', 'No'))
    depression = st.selectbox('😞 Do you suffer from depression?', ('Yes', 'No'))
    
    def predict_treatment(panic_attack, anxiety, depression):
        if panic_attack == 'Yes' or anxiety == 'Yes' or depression == 'Yes':
            return 'Yes'
        else:
            return 'No'
    
    import streamlit as st

    def treatment(panic_attack, anxiety, depression):
        treatments = []

        if panic_attack == 'Yes':
                    treatments.append('''\
            Treatment for Panic Attack:
            - Psikoterapi:
                - Cognitive Behavioral Therapy (CBT): Membantu mahasiswa mengenali dan mengelola pemicu serangan panik.
            - Teknik Relaksasi:
                - Latihan pernapasan dalam dan teknik relaksasi otot progresif untuk menenangkan diri.
            - Obat-obatan:
                - Benzodiazepines atau Antidepresan dapat diresepkan dalam kasus tertentu.
            - Eksposur Terhadap Pemicu:
                - Terapi paparan untuk membantu mahasiswa menghadapi situasi yang memicu serangan panik secara bertahap.\
            ''')

        if anxiety == 'Yes':
                    treatments.append('''\
            Treatment for Anxiety:
            - Psikoterapi:
                - Cognitive Behavioral Therapy (CBT): Efektif dalam mengubah pola pikir yang menyebabkan kecemasan.
                - Acceptance and Commitment Therapy (ACT): Membantu mahasiswa menerima pikiran dan perasaan tanpa menghakimi dan tetap fokus pada tindakan yang bermakna.
            - Teknik Relaksasi:
                - Meditasi, yoga, dan mindfulness untuk mengurangi kecemasan.
            - Obat-obatan:
                - Anxiolytics atau Antidepresan sesuai resep dokter.
            - Perubahan Gaya Hidup:
                - Menjaga pola tidur, diet seimbang, dan aktivitas fisik rutin.\
            ''')

        if depression == 'Yes':
                    treatments.append('''\
            Treatment for Depression:
            - Psikoterapi:
                - Cognitive Behavioral Therapy (CBT): Terapi ini membantu mahasiswa mengidentifikasi dan mengubah pola pikir negatif.
                - Interpersonal Therapy (IPT): Fokus pada hubungan dan membantu mengatasi masalah interpersonal yang dapat mempengaruhi suasana hati.
            - Obat-obatan:
                - Antidepresan: Seperti selective serotonin reuptake inhibitors (SSRIs) atau serotonin and norepinephrine reuptake inhibitors (SNRIs). Penggunaan obat harus di bawah pengawasan dokter.
            - Dukungan Sosial:
                - Membangun jaringan dukungan dengan teman, keluarga, atau bergabung dengan kelompok dukungan.
            - Aktivitas Fisik:
                - Olahraga secara teratur dapat membantu meningkatkan suasana hati.
            - Perubahan Gaya Hidup:
                - Tidur yang cukup, makan makanan seimbang, dan menghindari alkohol serta narkoba.\
            ''')

        if not treatments:
                    return '''\
            AYO TERUS JAGA KESEHATAN MENTAL KAMU, DENGAN MENERAPKAN TIPS DI BAWAH INI!!
            Tips Menjaga Kesehatan Mental untuk Mahasiswa:

            1. Kelola Waktu dengan Baik:
            - Buat jadwal yang terorganisir untuk mengelola tugas akademik dan kegiatan lainnya.

            2. Berolahraga Secara Teratur:
            - Aktivitas fisik dapat membantu mengurangi stres dan meningkatkan mood.

            3. Istirahat yang Cukup:
            - Pastikan tidur yang cukup setiap malam untuk memulihkan energi dan fungsi otak.

            4. Konsumsi Makanan Sehat:
            - Diet seimbang yang kaya akan nutrisi dapat mendukung kesehatan mental dan fisik.

            5. Tetap Terhubung:
            - Jalin hubungan sosial yang positif dengan teman dan keluarga.

            6. Praktikkan Mindfulness:
            - Meditasi atau latihan mindfulness dapat membantu mengelola stres dan meningkatkan konsentrasi.

            7. Cari Bantuan Saat Diperlukan:
            - Jangan ragu untuk mencari bantuan dari konselor atau layanan kesehatan mental di kampus jika merasa tertekan atau butuh seseorang untuk diajak bicara.

            8. Lakukan Kegiatan yang Menyenangkan:
            - Luangkan waktu untuk hobi atau kegiatan yang disukai untuk menjaga keseimbangan hidup.

            9. Jauhi Pengaruh Negatif:
            - Hindari alkohol, narkoba, dan perilaku berisiko yang dapat merugikan kesehatan mental.

            10. Terlibat dalam Kegiatan Kampus:
                - Bergabung dengan klub atau organisasi yang sesuai minat untuk menambah jaringan sosial dan pengalaman positif.\
            '''
        else:
                    return '\n\n'.join(treatments)




    st.title("Mental Health Treatment Predictor")

    if st.button('Predict'):
        if 'model_rf_treat' in st.session_state and 'model_xgb_treat' in st.session_state:
            treatment_prediction = predict_treatment(panic_attack, anxiety, depression)
            treatmentrec = treatment(panic_attack, anxiety, depression)
            st.subheader("Treatment Prediction")
            st.write(treatment_prediction)
            st.write(treatmentrec)
        else:
            st.write("Models are not trained yet. Please train the models in the Predictive Modeling section.")




        
        # user_input = {
        #     'Gender': 1 if gender == 'Female' else 0,
        #     'Age': age,
        #     'Course': df['Course'].tolist().index(course),
        #     'Year': year,
        #     'CGPA': cgpa,
        #     'Marital_Status': 1 if marital_status == 'Yes' else 0
        # }

        # # if st.button('Predict'):
        # #     if 'model_rf_treat' in st.session_state and 'model_xgb_treat' in st.session_state:
        # #         treatment_prediction = predict_treatment(panic_attack, anxiety, depression)
        # #         treatmentrec = treatment(panic_attack, anxiety, depression)


        # #         st.subheader("Treatment Prediction")
        # #         st.write(f"Needs treatment: {treatment_prediction}")
        # #         st.write(f"Rekomendasi: {treatmentrec}")
        # #     else:
        # #         st.write("Models are not trained yet. Please train the models in the Predictive Modeling section.")

        