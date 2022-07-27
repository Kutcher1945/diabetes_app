#Core Pkgs
from PIL import Image
import streamlit as st
import cv2 as cv

#import maze

#EDA Pkgs
import pandas as pd 
import numpy as np 
import codecs
from pandas_profiling import ProfileReport

#Components
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

#Utils 
import os
import joblib
import hashlib
import io

#passlib, bcrypt

#Data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib 
matplotlib.use('Agg')

#DB
from managed_db import *

#Password
def generate_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hashes(password,hashed_text):
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False



feature_names_best = ['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']

gender_dict = {"male":1,"female":2}
feature_dict = {"No":1,"Yes":2}


def get_value(val,my_dict):
	for key, value in my_dict.items():
		if val == key:
			return value

def get_key(val,my_dict):
	for key, value in my_dict.items():
		if val == key:
			return key

def get_fvalue(val):
	feature_dict = {"No":1,"Yes":2}
	for key, value in feature_dict.items():
		if val == key:
			return value

# Load ML models
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


# ML Interpretation
import lime
import lime.lime_tabular
result_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

result_temp2 ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/{}" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

prescriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily. If you're not active now, it’s time to start. You don't have to join a gym and do cross-training. Just walk, ride a bike, or play active video games. Your goal should be 30 minutes of activity that makes you sweat and breathe a little harder most days of the week. An active lifestyle helps you control your diabetes by bringing down your blood sugar. It also lowers your chances of getting heart disease. Plus, it can help you lose extra pounds and ease stress.</li>
		<li style="text-align:justify;color:black;padding:10px">Stop smoking. Diabetes makes you more likely to have health problems like heart disease, eye disease, stroke, kidney disease, blood vessel disease, nerve damage, and foot problems. If you smoke, your chance of getting these problems is even higher. Smoking also can make it harder to exercise. Talk with your doctor about ways to quit.</li>
		<li style="text-align:justify;color:black;padding:10px">Manage stress. When you're stressed, your blood sugar levels go up. And when you're anxious, you may not manage your diabetes well. You may forget to exercise, eat right, or take your medicines. Find ways to relieve stress -- through deep breathing, yoga, or hobbies that relax you.</li>
		<li style="text-align:justify;color:black;padding:10px">Watch your alcohol. It may be easier to control your blood sugar if you don’t get too much beer, wine, and liquor. So if you choose to drink, don't overdo it. The American Diabetes Association says that women who drink alcohol should have no more than one drink a day and men should have no more than two. Alcohol can make your blood sugar go too high or too low. Check your blood sugar before you drink, and take steps to avoid low blood sugars. If you use insulin or take drugs for your diabetes, eat when you're drinking. Some drinks -- like wine coolers -- may be higher in carbs, so take this into account when you count carbs.</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet. Eat healthy. This is crucial when you have diabetes, because what you eat affects your blood sugar. No foods are strictly off-limits. Focus on eating only as much as your body needs. Get plenty of vegetables, fruits, and whole grains. Choose nonfat dairy and lean meats. Limit foods that are high in sugar and fat. Remember that carbohydrates turn into sugar, so watch your carb intake. Try to keep it about the same from meal to meal. This is even more important if you take insulin or drugs to control your blood sugars.</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Get checkups. See your doctor at least twice a year. Diabetes raises your odds of heart disease. So learn your numbers: cholesterol, blood pressure, and A1c (average blood sugar over 3 months). Get a full eye exam every year. Visit a foot doctor to check for problems like foot ulcers and nerve damage.</li>
		<ul>
	</div>
	"""


descriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Definition</h3>
		<p>can cause both acute and chronic disease.</p>
	</div>
	"""

@st.cache
def load_image(img):
	image = Image.open(os.path.join(img))
	st.image(image, use_column_width=True)

footer_html = """
	 <!-- CSS  -->
	  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
	  <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	  <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	   <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
	 <footer class="page-footer grey darken-4">
	    <div class="container" id="aboutapp">
	      <div class="row">
	        <div class="col l6 s12">
	          <h5 class="white-text">About Diabetes Web App</h5>
	          <p class="grey-text text-lighten-4">Diabetes prediction app with neural network and machine learning</p>
	        </div>
	      
	   <div class="col l3 s12">
	          <h5 class="white-text">Contact me if u have any issues with app</h5>
	          <ul>
	            <a href="https://www.facebook.com/adilan.akhramovich/" target="_blank" class="white-text">
	            <i class="fab fa-facebook fa-4x"></i>
	          </a>
	          <a href="https://www.linkedin.com/in/dylan-akhramovich-914132158/" target="_blank" class="white-text">
	            <i class="fab fa-linkedin fa-4x"></i>
	          </a>
	          <a href="https://www.youtube.com/channel/UCrk4Kk4-zURx9V7pVycEhHA?view_as=subscriber" target="_blank" class="white-text">
	            <i class="fab fa-youtube-square fa-4x"></i>
	          </a>
	           <a href="https://github.com/" target="_blank" class="white-text">
	            <i class="fab fa-github-square fa-4x"></i>
	          </a>
	          </ul>
	        </div>
	      </div>
	    </div>
	    <div class="footer-copyright">
	      <div class="container">
	      Made by <a class="white-text text-lighten-3" href="https://www.facebook.com/adilan.akhramovich/">Adilan Akhramovich and ECHIDNA Inc.</a><br/>
	      <a class="white-text text-lighten-3" href="https://www.facebook.com/adilan.akhramovich/">Jesus Saves @ECHIDNA Inc.</a>
	      </div>
	    </div>
	  </footer>
	"""

def main():		
	"""Diabetes Prediction App"""
	html_temp = """
		<div style="background-color:navy;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">MYSDS AI</h1>
		<h5 style="color:white;text-align:center;">Toxicity Prediction WEB APP</h5>
		</div>
		"""

	components.html(html_temp)
	image = Image.open('C:/Users/Adila/Documents/hep_app/LOGO.png')
	st.image(image, use_column_width=True)

	menu = ["Home", "Login", "SignUp", "Book An Appointment", "Profile Report", "Symbol Recognition", "About", "Privacy Policy"]
	submenu = ["Plot", "Prediction",]

	choice = st.sidebar.selectbox("Menu", menu)
	if choice == "Home":
		st.subheader("---------------WELCOME TO THE DIABETES PREDICTION APP-----------------")
		




		html_temp2 = """
		<div style="background-color:navy;padding:3px;border-radius:10px">
		<h1 style="color:white;text-align:center;">How to Login?</h1>
		<h5 style="color:white;text-align:center;">press the arrow on the top left corner and choose the LOGIN from menu to get started</h5>
		</div>
		"""
		components.html(html_temp2)

		html_temp3 = """
		<div style="background-color:navy;padding:3px;border-radius:10px">
		<h1 style="color:white;text-align:center;">How to Sign Up?</h1>
		<h5 style="color:white;text-align:center;">press the arrow on the top left corner and choose the SIGN UP from menu to get started</h5>
		</div>
		"""
		components.html(html_temp3)
		st.title("Brief explanation on Diabetes Mellitus")
		st.subheader("------Diabetes mellitus (DM), commonly known as diabetes, is a group of metabolic disorders characterized by a high blood sugar level over a prolonged period. Symptoms often include frequent urination, increased thirst, and increased appetite. If left untreated, diabetes can cause many complications. Acute complications can include diabetic ketoacidosis, hyperosmolar hyperglycemic state, or death. Serious long-term complications include cardiovascular disease, stroke, chronic kidney disease, foot ulcers, damage to the nerves, damage to the eyes and cognitive impairment.")
		st.subheader("------Diabetes is due to either the pancreas not producing enough insulin, or the cells of the body not responding properly to the insulin produced. There are three main types of diabetes mellitus:")
		st.subheader("------Type 1 diabetes results from the pancreas's failure to produce enough insulin due to loss of beta cells. This form was previously referred to as insulin-dependent diabetes mellitus (IDDM) or  juvenile diabetes . The loss of beta cells is caused by an autoimmune response. The cause of this autoimmune response is unknown.") 
		st.subheader("------Type 2 diabetes begins with insulin resistance, a condition in which cells fail to respond to insulin properly. As the disease progresses, a lack of insulin may also develop. This form was previously referred to as non-insulin-dependent diabetes mellitus  (NIDDM) or  adult-onset diabetes .  The most common cause is a combination of excessive body weight and insufficient exercise.") 
		st.subheader("------Gestational diabetes is the third main form and occurs when pregnant women without a previous history of diabetes develop high blood sugar levels.")
		st.subheader("------Type 1 diabetes must be managed with insulin injections. Prevention and treatment of type 2 diabetes involves maintaining a healthy diet, regular physical exercise, a normal body weight, and avoiding use of tobacco. Type 2 diabetes may be treated with medications such as insulin sensitizers with or without insulin. Control of blood pressure and maintaining proper foot and eye care are important for people with the disease. Insulin and some oral medications can cause low blood sugar.  Weight loss surgery in those with obesity is sometimes an effective measure in those with type 2 diabetes. Gestational diabetes usually resolves after the birth of the baby.")
		st.subheader("------As of 2019, an estimated 463 million people had diabetes worldwide (8.8% of the adult population), with type 2 diabetes making up about 90% of the cases.  Rates are similar in women and men. Trends suggest that rates will continue to rise. Diabetes at least doubles a person's risk of early death. In 2019, diabetes resulted in approximately 4.2 million deaths. It is the 7th leading cause of death globally. The global economic cost of diabetes related health expenditure in 2017 was estimated at US$727 billion. In the United States, diabetes cost nearly US$327 billion in 2017. Average medical expenditures among people with diabetes are about 2.3 times higher.")

	elif choice == "Login":
		username = st.sidebar.text_input("Username")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = generate_hashes(password)
			result = login_user(username,verify_hashes(password, hashed_pswd))
			#if password == "12345":
			if result:
				st.success("Welcome {}".format(username))

				activity = st.selectbox("Activity", submenu)
				if activity == "Plot":
					st.subheader("Data Vis Plot")
					df = pd.read_csv("data/clean_hepatitis_dataset.csv")
					st.dataframe(df)

					df['class'].value_counts().plot(kind='bar')
					st.pyplot()

					#Freq Dist Plot
					freq_df = pd.read_csv("data/freq_df_hepatitis_dataset.csv")
					st.bar_chart(freq_df['count'])

					
					if st.checkbox("Area Chart"):
						all_columns = df.columns.to_list()
						feat_choices = st.multiselect("Choose a Feature",all_columns)
						new_df = df[feat_choices]
						st.area_chart(new_df)

				elif activity == "Prediction":
					st.subheader("Predictive Analytics")

					age = st.number_input("Age",7,80)
					sex = st.radio("Sex",tuple(gender_dict.keys()))
					steroid = st.radio("Do You Take Steroids?",tuple(feature_dict.keys()))
					antivirals = st.radio("Do You Take Antivirals?",tuple(feature_dict.keys()))
					fatigue = st.radio("Do You Have Fatigue",tuple(feature_dict.keys()))
					spiders = st.radio("Presence of Spider Naeve",tuple(feature_dict.keys()))
					ascites = st.selectbox("Ascities",tuple(feature_dict.keys()))
					varices = st.selectbox("Presence of Varices",tuple(feature_dict.keys()))
					bilirubin = st.number_input("bilirubin Content",0.0,8.0)
					alk_phosphate = st.number_input("Alkaline Phosphate Content",0.0,296.0)
					sgot = st.number_input("Sgot",0.0,648.0)
					albumin = st.number_input("Albumin",0.0,6.4)
					protime = st.number_input("Prothrombin Time",0.0,100.0)
					histology = st.selectbox("Histology",tuple(feature_dict.keys()))
					feature_list = [age,get_value(sex,gender_dict),get_fvalue(steroid),get_fvalue(antivirals),get_fvalue(fatigue),get_fvalue(spiders),get_fvalue(ascites),get_fvalue(varices),bilirubin,alk_phosphate,sgot,albumin,int(protime),get_fvalue(histology)]

					st.write(feature_list)
					pretty_result = {"age":age,"sex":sex,"steroid":steroid,"antivirals":antivirals,"fatigue":fatigue,"spiders":spiders,"ascites":ascites,"varices":varices,"bilirubin":bilirubin,"alk_phosphate":alk_phosphate,"sgot":sgot,"albumin":albumin,"protime":protime,"histolog":histology}
					st.json(pretty_result)
					single_sample = np.array(feature_list).reshape(1,-1)

					#ML
					model_choice = st.selectbox("Select Model", ["KNN", "DecisionTree", "LR"]) 
					if st.button("Predict"):
						if model_choice == "KNN":
							loaded_model = load_model("models/knn_hepB_model.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						elif model_choice == "DecisionTree":
							loaded_model = load_model("models/decision_tree_clf_hepB_model.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						else:
							loaded_model = load_model("models/logistic_regression_hepB_model.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)

						#st.write(prediction)
						#prediction_label = {"You have a risk to have diabetes":1, "You don't have a risk to have diabetes":2}
						#final_result = get_key(prediction,prediction_label)
						#st.write(Ffinal_result)
						if prediction == 1:
							st.warning("Patient has a risk to have Diabetes")
							pred_probability_score = {"Dibetes":pred_prob[0][0]*100,"No Diabetes":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using Neural network with {}".format(model_choice))
							st.json(pred_probability_score)
							st.subheader("Prescriptive Analytics")
							st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
							
						else:
							st.success("Patient don't have a risk of having Diabetes")
							pred_probability_score = {"Has a risk":pred_prob[0][0]*100,"No Risk":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using Neural network with {}".format(model_choice))
							st.json(pred_probability_score)



					if st.checkbox("Interpret"):
						if model_choice == "KNN":
							loaded_model = load_model("models/knn_hepB_model.pkl")
							
						elif model_choice == "DecisionTree":
							loaded_model = load_model("models/decision_tree_clf_hepB_model.pkl")
							
						else:
							loaded_model = load_model("models/logistic_regression_hepB_model.pkl")
							

						# loaded_model = load_model("models/logistic_regression_model.pkl")							
						# 1 Die and 2 Live
						df = pd.read_csv("data/clean_hepatitis_dataset.csv")
						x = df[['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']]
						feature_names = ['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']
						class_names = ['Die(1)','Live(2)']
						explainer = lime.lime_tabular.LimeTabularExplainer(x.values,feature_names=feature_names, class_names=class_names,discretize_continuous=True)
						# The Explainer Instance
						exp = explainer.explain_instance(np.array(feature_list), loaded_model.predict_proba,num_features=13, top_labels=13)
						exp.show_in_notebook(show_table=True, show_all=True)
						# exp.save_to_file('lime_oi.html')
						st.write(exp.as_list())
						new_exp = exp.as_list()
						label_limits = [i[0] for i in new_exp]
						# st.write(label_limits)
						label_scores = [i[1] for i in new_exp]
						plt.barh(label_limits,label_scores)
						st.pyplot()
						plt.figure(figsize=(20,10))
						fig = exp.as_pyplot_figure()
						st.pyplot()	
				


			else:		
				st.warning("Incorrect Username or Password")
	elif choice == "SignUp":
		new_username = st.text_input("User name")
		new_password = st.text_input("Password", type='password')

		confirm_password = st.text_input("Confirm Password",type='password')
		if new_password == confirm_password:
			st.success("Password Confirmed")
		else:
			st.warning("Passwords not the same")

		if st.button("Submit"):
			create_usertable()
			hashed_new_password = generate_hashes(new_password)
			add_userdata(new_username,hashed_new_password)
			st.success("You have created a new account")
			st.info("Login to get started")

	elif choice == "Profile Report":
		st.title("What is Profile Report?")
		st.subheader("------This technology can be used to upload datasets that you may have as analyst or if you are a doctor who wants to get a clear idea about percentage of patients who has this or that problem. And then the technology itself will analyze it.")
		st.title("How to start to analyze the data?")
		st.subheader("------To analyze the data you just simply need to upload a CSV file") 
		st.title("What kind of files this technology accepts?")
		st.subheader("------Basically a CSV files")
		st.title("How is it analyzing the data?")
		st.subheader("------Mainly, technology itself uses a neural networks to analyze the data, and then it will represent the data as a charts")
		


		data_file = st.file_uploader("UPLOAD CSV",type=['csv'])
		st.set_option('deprecation.showfileUploaderEncoding', False)
		if data_file is not None:
			df = pd.read_csv(data_file)
			st.dataframe(df.head())
			profile = ProfileReport(df)
			st_profile_report(profile)

	elif choice == "Symbol Recognition":
		st.title("What is Symbol Recognition?")
		st.subheader("------This technology can be used to upload photos or videos that you may have. And then the technology itself will analyze it.")
		st.title("How to start to recognize the symbols from the photos or videos?")
		st.subheader("------To analyze the data you just simply need to upload an image or video ") 
		st.title("What kind of files this technology accepts?")
		st.subheader("------Basically a JPEG, MP4, and etc.")
		st.title("How is it analyzing the photos and videos?")
		st.subheader("------Mainly, technology itself uses a neural networks to analyze the data, and then it will represent the data as a charts")
		


		data_file = st.file_uploader("UPLOAD CSV",type=['csv'])
		st.set_option('deprecation.showfileUploaderEncoding', False)
		if data_file is not None:
			df = pd.read_csv(data_file)
			st.dataframe(df.head())
			profile = ProfileReport(df)
			st_profile_report(profile)
			
	elif choice == "Book An Appointment":
		st.title("Book An Appointments")
		st.title("Integration with Nilai medical center website")
		st.subheader("------Developer integrated this WebApp with existed website to make sure that patients can book an appointment to a real medical") 
		components.iframe('https://nmc.encoremed.io/',width=700,height=2000)

	elif choice == "About":
		st.title("About App")
		st.title("F.A.Q.")
		st.title("What is Echidna AI?")
		st.subheader("------Basically, its an a WEB APP that can help people to predict Diabetes")
		st.title("What kind of functions do Echidna Have?")
		st.subheader("------The main purpose of the Echidna AI is to provide a solution for people to predict diabetes and to help analysts to analyze the data in a better way. And the data itself can be stored inside this WEB APP because it has a neural network that can store data inside nodes")
		st.title("Is it Open source Alghorithm?")
		st.subheader("------The Echidna AI® algorithm")
		st.subheader("------This web app was released as open source software under the GNU Affero General Public Licence, version 3. This ensures that academics and others interested in the core of the algorithms at least start with a working implementation. The terms of the licence ensure that implementations are transparent and open, and are, in turn, open for others to use and/or modify under the same licence.")
		st.title("Is Echidna AI can be recommended for clinical use?")
		st.subheader("------It can be recommended for clinical use, software developers can use this professionally supported software development kits.")
		st.title("Would Echidna AI be supported in future?")
		st.subheader("------Echidna AI®-2020 will be released to licencees of our Echidna AI® software development kit in the new year, for deployment from August. Which means that it will be suported")
		st.title("Do Echidna AI patented or is it has a copyright?")
		st.subheader("Yes, Echidna AI has  a copyright, but it is an open source software that can be modified")
		st.subheader("------Copyright ©Echidna 2020. ALL RIGHTS RESERVED.")
		st.subheader("------Materials on this web site are protected by copyright law. Access to the materials on this web site for the sole purpose of personal educational and research use only. Where appropriate a single print out of a reasonable proportion of these materials may be made for personal education, research and private study. Materials should not be further copied, photocopied or reproduced, or distributed in electronic form. Any unauthorised use or distribution for commercial purposes is expressly forbidden. Any other unauthorised use or distribution of the materials may constitute an infringement of ClinRisk's copyright and may lead to legal action.")
		st.subheader("------For avoidance of doubt, any use of this site as a web service to obtain a Echidna AI® for any purpose is expressly forbidden. Similarly, use of this website for developing or testing software of any sort is forbidden unless permission has been explicitly granted.")
		st.subheader("------BMI predictor algorithm © 2020 Echidna Inc.")
		st.subheader("------WebApp and risk engine built by Adilan Akhramovich WebApp design ©Echidna 2020.")
		#components.iframe('https://quickdraw.withgoogle.com',height=2000)
		components.html(footer_html,height=500)

	elif choice == "Privacy Policy":
		st.title("Privacy Policy of Echidna Inc.")
		st.subheader("------At ECHIDNA AI, one of our main priorities is the privacy of our visitors. This Privacy Policy document contains types of information that is collected and recorded by ECHIDNA AI and how we use it.")

		st.subheader("------If you have additional questions or require more information about our Privacy Policy, do not hesitate to contact us.")

		st.subheader("------This Privacy Policy applies only to our online activities and is valid for visitors to our webapp with regards to the information that they shared and/or collect in ECHIDNA AI. This policy is not applicable to any information collected offline or via channels other than this webapp.")

		st.title("Consent")
		st.subheader("------By using our webapp, you hereby consent to our Privacy Policy and agree to its terms.")

		st.title("Information we collect")
		st.subheader("------The personal information that you are asked to provide, and the reasons why you are asked to provide it, will be made clear to you at the point we ask you to provide your personal information.")

		st.subheader("------If you contact us directly, we may receive additional information about you such as your name, email address, phone number, the contents of the message and/or attachments you may send us, and any other information you may choose to provide.")

		st.title("How we use your information?")
		st.subheader("We use the information we collect in various ways, including to:")

		st.subheader("------Provide, operate, and maintain our webapp")
		st.subheader("------Improve, personalize, and expand our webapp")
		st.subheader("------Understand and analyze how you use our webapp")
		st.subheader("------Develop new products, services, features, and functionality")
		st.subheader("------Communicate with you, either directly or through one of our partners, including for customer service, to provide you with updates and other information relating to the webapp, and for marketing and promotional purposes")
		st.subheader("------Send you emails")
		st.subheader("------Find and prevent fraud")

		st.title("Log Files")
		st.subheader("------ECHIDNA AI follows a standard procedure of using log files. These files log visitors when they visit websites. All hosting companies do this and a part of hosting services' analytics. The information collected by log files include internet protocol (IP) addresses, browser type, Internet Service Provider (ISP), date and time stamp, referring/exit pages, and possibly the number of clicks. These are not linked to any information that is personally identifiable. The purpose of the information is for analyzing trends, administering the site, tracking users' movement on the website, and gathering demographic information.")

		st.title("Advertising Partners Privacy Policies")
		st.subheader("------You may consult this list to find the Privacy Policy for each of the advertising partners of ECHIDNA AI.")

		st.subheader("------Third-party ad servers or ad networks uses technologies like cookies, JavaScript, or Web Beacons that are used in their respective advertisements and links that appear on ECHIDNA AI, which are sent directly to users' browser. They automatically receive your IP address when this occurs. These technologies are used to measure the effectiveness of their advertising campaigns and/or to personalize the advertising content that you see on websites that you visit.")

		st.subheader("------Note that ECHIDNA AI has no access to or control over these cookies that are used by third-party advertisers.")

		st.title("Third Party Privacy Policies")
		st.subheader("------ECHIDNA AI's Privacy Policy does not apply to other advertisers or websites. Thus, we are advising you to consult the respective Privacy Policies of these third-party ad servers for more detailed information. It may include their practices and instructions about how to opt-out of certain options.")

		st.subheader("------You can choose to disable cookies through your individual browser options. To know more detailed information about cookie management with specific web browsers, it can be found at the browsers' respective websites.")

		st.title("MCPA Privacy Rights (Do Not Sell My Personal Information)")
		st.subheader("Under the MCPA, among other rights, consumers have the right to:")

		st.subheader("------Request that a business that collects a consumer's personal data disclose the categories and specific pieces of personal data that a business has collected about consumers.")

		st.subheader("------Request that a business delete any personal data about the consumer that a business has collected.")

		st.subheader("------Request that a business that sells a consumer's personal data, not sell the consumer's personal data.")

		st.subheader("------If you make a request, we have one month to respond to you. If you would like to exercise any of these rights, please contact us.")

		st.title("GDPR Data Protection Rights")
		st.subheader("We would like to make sure you are fully aware of all of your data protection rights. Every user is entitled to the following:")

		st.subheader("------The right to access – You have the right to request copies of your personal data. We may charge you a small fee for this service.")

		st.subheader("------The right to rectification – You have the right to request that we correct any information you believe is inaccurate. You also have the right to request that we complete the information you believe is incomplete.")

		st.subheader("------The right to erasure – You have the right to request that we erase your personal data, under certain conditions.")

		st.subheader("------The right to restrict processing – You have the right to request that we restrict the processing of your personal data, under certain conditions.")

		st.subheader("------The right to object to processing – You have the right to object to our processing of your personal data, under certain conditions.")

		st.subheader("------The right to data portability – You have the right to request that we transfer the data that we have collected to another organization, or directly to you, under certain conditions.")

		st.subheader("------If you make a request, we have one month to respond to you. If you would like to exercise any of these rights, please contact us.")

		st.title("Children's Information")
		st.subheader("------Another part of our priority is adding protection for children while using the internet. We encourage parents and guardians to observe, participate in, and/or monitor and guide their online activity.")

		st.subheader("------ECHIDNA AI does not knowingly collect any Personal Identifiable Information from children under the age of 13. If you think that your child provided this kind of information on our website, we strongly encourage you to contact us immediately and we will do our best efforts to promptly remove such information from our records.")







if __name__ == '__main__':
	main()