from flask import Flask, escape, request, render_template
import pickle
import xgboost as xgb
import pandas as pd

model = pickle.load(open("final_model.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))

model_xgb_2 = xgb.Booster()
model_xgb_2.load_model("xgb_model.json")

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        try:
            if request.form:
                age = (request.form['age'])
                fnlwght = float(request.form['fnlwght'])
                educationnum = float(request.form['educationnum'])
                gain = float(request.form['gain'])
                loss = float(request.form['loss'])
                hour = float(request.form['hour'])
                work = (request.form['work'])
                education = request.form['education']
                marital = request.form['marital']
                occupation = request.form['occupation']
                relation = request.form['relation']
                race = request.form['race']
                gender = request.form['gender']
                country = request.form['country']


                # work class
                if(work=='Private'):
                    work_private = 1
                    work_federal = 0
                    work_local = 0
                    work_never = 0
                    work_self = 0
                    work_not_self = 0
                    work_state = 0
                    work_0 = 0
                elif(work=='Self-emp-not-inc'):
                    work_private = 0
                    work_federal = 0
                    work_local = 0
                    work_never = 0
                    work_self = 0
                    work_not_self = 1
                    work_state = 0
                    work_0 = 0
                elif(work=='Self-emp-inc'):
                    work_private = 0
                    work_federal = 0
                    work_local = 0
                    work_never = 0
                    work_self = 1
                    work_not_self = 0
                    work_state = 0
                    work_0 = 0
                elif(work=='Federal-gov'):
                    work_private = 0
                    work_federal = 1
                    work_local = 0
                    work_never = 0
                    work_self = 0
                    work_not_self = 0
                    work_state = 0
                    work_0 = 0
                elif(work=='Local-gov'):
                    work_private = 0
                    work_federal = 0
                    work_local = 1
                    work_never = 0
                    work_self = 0
                    work_not_self = 0
                    work_state = 0
                    work_0 = 0
                elif(work=='State-gov'):
                    work_private = 0
                    work_federal = 0
                    work_local = 0
                    work_never = 0
                    work_self = 0
                    work_not_self = 0
                    work_state = 1
                    work_0 = 0
                elif(work=='Never-worked'):
                    work_private = 0
                    work_federal = 0
                    work_local = 0
                    work_never = 1
                    work_self = 0
                    work_not_self = 0
                    work_state = 0
                    work_0 = 0
                elif(work=='0'):
                    work_private = 0
                    work_federal = 0
                    work_local = 0
                    work_never = 0
                    work_self = 0
                    work_not_self = 0
                    work_state = 0
                    work_0 = 1

                # education 
                if(education=='Primary'):
                    education_primary = 1
                    education_acdm = 0
                    education_voc = 0
                    education_bachelors = 0
                    education_doctorate = 0
                    education_grad = 0
                    education_masters = 0
                    education_preschool = 0
                    education_prof = 0
                    education_some = 0
                elif(education=='Bachelors'):
                    education_primary = 0
                    education_acdm = 0
                    education_voc = 0
                    education_bachelors = 1
                    education_doctorate = 0
                    education_grad = 0
                    education_masters = 0
                    education_preschool = 0
                    education_prof = 0
                    education_some = 0
                elif(education=='Some-college'):
                    education_primary = 0
                    education_acdm = 0
                    education_voc = 0
                    education_bachelors = 0
                    education_doctorate = 0
                    education_grad = 0
                    education_masters = 0
                    education_preschool = 0
                    education_prof = 0
                    education_some = 1
                elif(education=='HS-grad'):
                    education_primary = 0
                    education_acdm = 0
                    education_voc = 0
                    education_bachelors = 0
                    education_doctorate = 0
                    education_grad = 1
                    education_masters = 0
                    education_preschool = 0
                    education_prof = 0
                    education_some = 0
                elif(education=='Prof-school'):
                    education_primary = 0
                    education_acdm = 0
                    education_voc = 0
                    education_bachelors = 0
                    education_doctorate = 0
                    education_grad = 0
                    education_masters = 0
                    education_preschool = 0
                    education_prof = 1
                    education_some = 0
                elif(education=='Assoc-acdm'):
                    education_primary = 0
                    education_acdm = 1
                    education_voc = 0
                    education_bachelors = 0
                    education_doctorate = 0
                    education_grad = 0
                    education_masters = 0
                    education_preschool = 0
                    education_prof = 0
                    education_some = 0
                elif(education=='Assoc-voc'):
                    education_primary = 0
                    education_acdm = 0
                    education_voc = 1
                    education_bachelors = 0
                    education_doctorate = 0
                    education_grad = 0
                    education_masters = 0
                    education_preschool = 0
                    education_prof = 0
                    education_some = 0
                elif(education=='Masters'):
                    education_primary = 0
                    education_acdm = 0
                    education_voc = 0
                    education_bachelors = 0
                    education_doctorate = 0
                    education_grad = 0
                    education_masters = 1
                    education_preschool = 0
                    education_prof = 0
                    education_some = 0
                elif(education=='Doctorate'):
                    education_primary = 0
                    education_acdm = 0
                    education_voc = 0
                    education_bachelors = 0
                    education_doctorate = 1
                    education_grad = 0
                    education_masters = 0
                    education_preschool = 0
                    education_prof = 0
                    education_some = 0
                elif(education=='Preschool'):
                    education_primary = 0
                    education_acdm = 0
                    education_voc = 0
                    education_bachelors = 0
                    education_doctorate = 0
                    education_grad = 0
                    education_masters = 0
                    education_preschool = 1
                    education_prof = 0
                    education_some = 0


                # marital status 
                if(marital=='Married-civ-spouse'):
                    marital_divorce = 0
                    marital_civ = 1
                    marital_sabsent = 0
                    marital_never = 0
                    marital_seperated = 0
                    marital_widowed = 0
                elif(marital=='Divorced'):
                    marital_divorce = 1
                    marital_civ = 0
                    marital_sabsent = 0
                    marital_never = 0
                    marital_seperated = 0
                    marital_widowed = 0
                elif(marital=='Separated'):
                    marital_divorce = 0
                    marital_civ = 0
                    marital_sabsent = 0
                    marital_never = 0
                    marital_seperated = 1
                    marital_widowed = 0
                elif(marital=='Widowed'):
                    marital_divorce = 0
                    marital_civ = 0
                    marital_sabsent = 0
                    marital_never = 0
                    marital_seperated = 0
                    marital_widowed = 1
                elif(marital=='Married-spouse-absent'):
                    marital_divorce = 0
                    marital_civ = 0
                    marital_sabsent = 1
                    marital_never = 0
                    marital_seperated = 0
                    marital_widowed = 0
                elif(marital=='Never-married'):
                    marital_divorce = 0
                    marital_civ = 0
                    marital_sabsent = 0
                    marital_never = 1
                    marital_seperated = 0
                    marital_widowed = 0

                # relation
                if(relation=='Wife'):
                    relation_husband = 0
                    relation_not_family = 0
                    relation_other = 0
                    relation_child = 0
                    relation_unmarried = 0
                    relation_wife = 1
                elif(relation=='Own-child'):
                    relation_husband = 0
                    relation_not_family = 0
                    relation_other = 0
                    relation_child = 1
                    relation_unmarried = 0
                    relation_wife = 0
                elif(relation=='Husband'):
                    relation_husband = 1
                    relation_not_family = 0
                    relation_other = 0
                    relation_child = 0
                    relation_unmarried = 0
                    relation_wife = 0
                elif(relation=='Not-in-family'):
                    relation_husband = 0
                    relation_not_family = 1
                    relation_other = 0
                    relation_child = 0
                    relation_unmarried = 0
                    relation_wife = 0
                elif(relation=='Other-relative'):
                    relation_husband = 0
                    relation_not_family = 0
                    relation_other = 1
                    relation_child = 0
                    relation_unmarried = 0
                    relation_wife = 0
                elif(relation=='Unmarried'):
                    relation_husband = 0
                    relation_not_family = 0
                    relation_other = 0
                    relation_child = 0
                    relation_unmarried = 1
                    relation_wife = 0

                # race
                if(race =='White'):
                    race_indian = 0
                    race_asian = 0
                    race_black = 0
                    race_white = 1
                    race_other = 0
                elif(race=='Other'):
                    race_indian = 0
                    race_asian = 0
                    race_black = 0
                    race_white = 0
                    race_other = 1
                elif(race=='Black'):
                    race_indian = 0
                    race_asian = 0
                    race_black = 1
                    race_white = 0
                    race_other = 0
                elif(race=='Asian-Pac-Islander'):
                    race_indian = 0
                    race_asian = 1
                    race_black = 0
                    race_white = 0
                    race_other = 0
                elif(race=='Amer-Indian-Eskimo'):
                    race_indian = 1
                    race_asian = 0
                    race_black = 0
                    race_white = 0
                    race_other = 0

                # country
                if(country =='North_America'):
                    country_asian = 0
                    country_central = 0
                    country_EU = 0
                    country_North = 1
                    country_south = 0
                elif(country=='Central_America'):
                    country_asian = 0
                    country_central = 1
                    country_EU = 0
                    country_North = 0
                    country_south = 0
                elif(country=='South_America'):
                    country_asian = 0
                    country_central = 0
                    country_EU = 0
                    country_North = 0
                    country_south = 1
                elif(country=='EU'):
                    country_asian = 0
                    country_central = 0
                    country_EU = 1
                    country_North = 0
                    country_south = 0
                elif(country=='Asian'):
                    country_asian = 1
                    country_central = 0
                    country_EU = 0
                    country_North = 0
                    country_south = 0

                # gender
                if(gender =='Male'):
                    gender_male = 1
                    gender_female = 0 
                elif(gender=='Female'):
                    gender_male = 0
                    gender_female = 1 
                

                # occupation
                if(occupation =='Tech-support'):
                    occupation_adm = 0
                    occupation_craft = 0
                    occupation_managerial = 0
                    occupation_farming = 0
                    occupation_cleaner = 0
                    occupation_machine = 0
                    occupation_other = 0
                    occupation_house = 0
                    occupation_prof = 0
                    occupation_protective = 0
                    occupation_sales = 0
                    occupation_tech = 1
                    occupation_transport = 0
                    occupation_0 = 0
                elif(occupation=='0'):
                    occupation_adm = 0
                    occupation_craft = 0
                    occupation_managerial = 0
                    occupation_farming = 0
                    occupation_cleaner = 0
                    occupation_machine = 0
                    occupation_other = 0
                    occupation_house = 0
                    occupation_prof = 0
                    occupation_protective = 0
                    occupation_sales = 0
                    occupation_tech = 0
                    occupation_transport = 0
                    occupation_0 = 1
                elif(occupation=='Craft-repair'):
                    occupation_adm = 0
                    occupation_craft = 1
                    occupation_managerial = 0
                    occupation_farming = 0
                    occupation_cleaner = 0
                    occupation_machine = 0
                    occupation_other = 0
                    occupation_house = 0
                    occupation_prof = 0
                    occupation_protective = 0
                    occupation_sales = 0
                    occupation_tech = 0
                    occupation_transport = 0
                    occupation_0 = 0
                elif(occupation=='Other-service'):
                    occupation_adm = 0
                    occupation_craft = 0
                    occupation_managerial = 0
                    occupation_farming = 0
                    occupation_cleaner = 0
                    occupation_machine = 0
                    occupation_other = 1
                    occupation_house = 0
                    occupation_prof = 0
                    occupation_protective = 0
                    occupation_sales = 0
                    occupation_tech = 0
                    occupation_transport = 0
                    occupation_0 = 0
                elif(occupation=='Sales'):
                    occupation_adm = 0
                    occupation_craft = 0
                    occupation_managerial = 0
                    occupation_farming = 0
                    occupation_cleaner = 0
                    occupation_machine = 0
                    occupation_other = 0
                    occupation_house = 0
                    occupation_prof = 0
                    occupation_protective = 0
                    occupation_sales = 1
                    occupation_tech = 0
                    occupation_transport = 0
                    occupation_0 = 0
                elif(occupation=='Exec-managerial'):
                    occupation_adm = 0
                    occupation_craft = 0
                    occupation_managerial = 1
                    occupation_farming = 0
                    occupation_cleaner = 0
                    occupation_machine = 0
                    occupation_other = 0
                    occupation_house = 0
                    occupation_prof = 0
                    occupation_protective = 0
                    occupation_sales = 0
                    occupation_tech = 0
                    occupation_transport = 0
                    occupation_0 = 0
                elif(occupation=='Prof-specialty'):
                    occupation_adm = 0
                    occupation_craft = 0
                    occupation_managerial = 0
                    occupation_farming = 0
                    occupation_cleaner = 0
                    occupation_machine = 0
                    occupation_other = 0
                    occupation_house = 0
                    occupation_prof = 1
                    occupation_protective = 0
                    occupation_sales = 0
                    occupation_tech = 0
                    occupation_transport = 0
                    occupation_0 = 0
                elif(occupation=='Handlers-cleaners'):
                    occupation_adm = 0
                    occupation_craft = 0
                    occupation_managerial = 0
                    occupation_farming = 0
                    occupation_cleaner = 1
                    occupation_machine = 0
                    occupation_other = 0
                    occupation_house = 0
                    occupation_prof = 0
                    occupation_protective = 0
                    occupation_sales = 0
                    occupation_tech = 0
                    occupation_transport = 0
                    occupation_0 = 0
                elif(occupation=='Adm-clerical'):
                    occupation_adm = 1
                    occupation_craft = 0
                    occupation_managerial = 0
                    occupation_farming = 0
                    occupation_cleaner = 0
                    occupation_machine = 0
                    occupation_other = 0
                    occupation_house = 0
                    occupation_prof = 0
                    occupation_protective = 0
                    occupation_sales = 0
                    occupation_tech = 0
                    occupation_transport = 0
                    occupation_0 = 0
                elif(occupation=='Farming-fishing'):
                    occupation_adm = 0
                    occupation_craft = 0
                    occupation_managerial = 0
                    occupation_farming = 1
                    occupation_cleaner = 0
                    occupation_machine = 0
                    occupation_other = 0
                    occupation_house = 0
                    occupation_prof = 0
                    occupation_protective = 0
                    occupation_sales = 0
                    occupation_tech = 0
                    occupation_transport = 0
                    occupation_0 = 0
                elif(occupation=='Transport-moving'):
                    occupation_adm = 0
                    occupation_craft = 0
                    occupation_managerial = 0
                    occupation_farming = 0
                    occupation_cleaner = 0
                    occupation_machine = 0
                    occupation_other = 0
                    occupation_house = 0
                    occupation_prof = 0
                    occupation_protective = 0
                    occupation_sales = 0
                    occupation_tech = 0
                    occupation_transport = 1
                    occupation_0 = 0
                elif(occupation=='Priv-house-serv'):
                    occupation_adm = 0
                    occupation_craft = 0
                    occupation_managerial = 0
                    occupation_farming = 0
                    occupation_cleaner = 0
                    occupation_machine = 0
                    occupation_other = 0
                    occupation_house = 1
                    occupation_prof = 0
                    occupation_protective = 0
                    occupation_sales = 0
                    occupation_tech = 0
                    occupation_transport = 0
                    occupation_0 = 0
                elif(occupation=='Protective-serv'):
                    occupation_adm = 0
                    occupation_craft = 0
                    occupation_managerial = 0
                    occupation_farming = 0
                    occupation_cleaner = 0
                    occupation_machine = 0
                    occupation_other = 0
                    occupation_house = 0
                    occupation_prof = 0
                    occupation_protective = 1
                    occupation_sales = 0
                    occupation_tech = 0
                    occupation_transport = 0
                    occupation_0 = 0


                
                data = [age, fnlwght, educationnum, gain, loss, hour, work_federal, work_local, work_never, work_private, work_self, work_not_self, work_state, work_0, education_acdm, education_voc, education_bachelors, education_doctorate, education_grad, education_masters, education_preschool, education_prof, education_some, education_primary, marital_divorce, marital_civ, marital_sabsent, marital_never, marital_seperated, marital_widowed, occupation_adm, occupation_craft, occupation_managerial, occupation_farming, occupation_cleaner, occupation_machine, occupation_other, occupation_house, occupation_prof, occupation_protective, occupation_sales, occupation_tech, occupation_transport, occupation_0, relation_husband, relation_not_family, relation_other, relation_child, relation_unmarried, relation_wife, race_indian, race_asian, race_black, race_other, race_white, gender_female, gender_male, country_asian, country_central, country_EU, country_North, country_south ]
                # print(data)

                data_transform = scaler.fit_transform([data])
                # print(data_transform)
                da = pd.DataFrame(data_transform)
                # print(da)
                xgtest = xgb.DMatrix(da.values)

                response = model_xgb_2.predict(xgtest)[0]
                # print(response)

                # if(response=='1.0'):
                #     response="YES"
                # else:
                #     response="NO"
                return render_template("prediction.html", prediction_text="Chances of  income more than 50K => "+str(response))

        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}
            return render_template("prediction.html", prediction_text=error)

        return render_template("prediction.html")


    else:
        return render_template("prediction.html")


if __name__ == '__main__':
    app.debug = True
    app.run()