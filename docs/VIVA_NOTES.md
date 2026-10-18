# Viva / Interview Notes

Project me jo decisions liye gaye unke reasons. Har jawab ko apne words me bolne ki practice karo.
Numbers training ke baad `reports/metrics.json` ya Model Performance page se apne run ke dekh lena.

## 1. Project ka goal kya hai?
Machine fail hone se pehle prediction karna taaki maintenance planned ho. Sensor readings (temperature, speed,
torque, tool wear) se failure probability nikalte hain, use LOW / MEDIUM / HIGH risk me badalte hain, batate hain
ki kyun, aur maintenance record banane tak ka workflow dete hain.

## 2. Interface ko dashboard kyun banaya, sirf form kyun nahi?
Sirf form wala project "ML model + input box" lagta hai. Plant me log ek machine nahi, poori fleet dekhte hain:
kaun si machine at-risk hai, kya karna hai, kaun sa kaam pending hai. Isliye dashboard (fleet overview), machine
page (detail), maintenance page (action) aur model performance page (evaluation) banaye. Predict form ab sirf ek
page hai aur machine ki latest reading se auto-fill hota hai.

## 3. Random split kyun nahi kiya?
Data time series jaisa hai. Random split me future ki readings train set me chali jaati hain aur score zaroorat
se zyada achha dikhta hai (data leakage). Isliye time ke hisaab se split kiya: pehla 75% train, last 25% test.

## 4. Accuracy kyun nahi dekhi?
Failures sirf ~3% hain. Model har baar "no failure" bole to bhi accuracy ~97% aayegi, par ek bhi failure nahi
pakdega. Isliye precision, recall, F1, ROC-AUC aur PR-AUC dekhe. Predictive maintenance me missed failure
(false negative) zyada costly hai, false alarm sirf extra inspection hai.

## 5. Imbalance kaise handle kiya?
Class weights (`scale_pos_weight` XGBoost me, `class_weight` baaki models me). SMOTE use nahi kiya kyunki time series
rows ke synthetic copies banana risky hai.

## 6. XGBoost kyun?
Tabular data par achha chalta hai, non-linear relations aur feature interactions seekhta hai, aur imbalance ke
parameters hain. Par seedha nahi choose kiya: Logistic Regression, Decision Tree aur Random Forest se compare kiya.
(Is repo ke saath diye result me XGBoost ki jagah sklearn ka gradient boosting use hua tha, kyunki us system par
XGBoost install nahi ho paya. Apne system par train karke apne numbers note karo.)

## 7. Feature engineering me kya kiya?
Rolling mean/std (stability), delta (sudden change), slope (gradual trend), power = torque x speed, torque/speed
ratio, wear rate, temperature difference (process - air). Single reading se degradation nahi dikhta, trend se dikhta hai.

## 8. Leakage kaise roka?
Saare rolling features sirf pichli readings se bante hain (backward-looking window). Test me check bhi hai:
pehli 8 rows se banaya feature poore data wale feature ke barabar aata hai (`test_rolling_features_use_only_past_data`).

## 9. Training aur Flask me features same kaise rakhe?
Dono jagah `preprocessing.add_features` hi chalta hai. Flask me nayi reading ke saath us machine ki pichli readings
database se lekar jodte hain, phir wahi function chalta hai. Model file me feature list bhi saved hai.

## 10. "Why is this machine at risk?" kaise nikalta hai? Kya ye SHAP hai?
Nahi, SHAP nahi hai. 17 features ko 8 sensor groups me baanta hoon (jaise Tool wear, Torque). Har group ko
training median se replace karke dekhta hoon ki model ka failure score (log-odds) kitna girta hai. Jo group hatane
par sabse zyada girta hai wahi risk ka bada kaaran hai. Bars us girawat ka hissa (%) dikhate hain, isliye sab milkar
100% hote hain. Log-odds isliye liye ki probability 100% ke paas saturate ho jaati hai. SHAP better hoga kyunki wo
features ke interactions fairly baantta hai; wo future work me hai.

## 11. Anomaly detection ka kya role hai?
Isolation Forest sirf healthy data se normal behaviour seekhta hai. Ye un faults ke liye hai jinke labels nahi hain.
Score ko LOW / MEDIUM / HIGH level me badla (decision boundary paar karte hi MEDIUM, kaafi door ho to HIGH). Iska
overlap real failures se ~48% precision / 47% recall hai, isliye ise failure model ki jagah nahi, uske saath use
karte hain: high risk + anomaly = turant inspection, low risk + anomaly = sensor issue ya unknown fault ho sakta hai.

## 12. RUL ka result kaisa hai?
Experimental. MAE ~7.7 readings vs average-baseline ~10.2, yaani behtar hai par accurate countdown nahi.
RUL ek fixed horizon (30 readings) par capped hai. UI me "experimental" likha hai. Real run-to-failure data
(NASA C-MAPSS) par ye zyada achha hona chahiye.

## 13. Dataset real hai?
Nahi, synthetic hai (`data_generator.py`), AI4I 2020 jaisa structure aur failure rules ke saath, aur kuch
"degrading" machines taaki demo me at-risk machines dikhein. Real AI4I CSV `import_ai4i.py` se load ho sakta hai.
Synthetic data ki limitation report me honestly likhni chahiye.

## 14. Test period me failure rate train se zyada kyun hai?
Degradation timeline ke aakhri hisse me daali gayi hai, aur time-based split me wahi hissa test hai. Isliye test me
failure rate (~5.3%) train (~2.2%) se zyada hai. Ye synthetic data ka side effect hai, real plant me aise nahi hoga.
Isliye test score ko "demo data par" hi bolna chahiye.

## 15. Threshold kaise choose karte hain?
Business cost se. Missed failure mehenga ho to threshold kam rakho (recall badhega, false alarms bhi badhenge).
`config.py` me `HIGH_THRESHOLD` / `MEDIUM_THRESHOLD` hain. `03_model_training.ipynb` me alag thresholds par
precision/recall ki table hai.

## 16. Maintenance workflow kaise kaam karta hai?
Dashboard / machine page se "Create maintenance record" par jaate hain; form machine ke latest prediction se
pre-fill hota hai (issue, priority, action). Record "Scheduled" status me save hota hai, aur "Mark completed" se
Completed ho jaata hai. "Maintenance due" card = at-risk machines jinka koi open record nahi hai.

## 17. Charts kaise bane? Koi library kyun nahi?
`src/charts.py` me inline SVG. Isse koi JavaScript library ya internet nahi chahiye, aur database ka text
`html.escape` se guzarta hai isliye chart me kuch bigaad nahi sakta.

## 18. Drift kya hai aur kaise check kiya?
Live sensor readings ka average training average se kitna hata, training std ke unit me. 0.5 se kam stable, 0.5-1.0
watch, 1.0 se zyada drift. Drift ho to model dobara train karna chahiye.

## 19. Database design?
7 tables: machines, sensor_readings, failure_events, predictions, maintenance, ml_models, user_session. Ek machine ke
bahut saare readings / predictions / maintenance records (one-to-many). Predictions me model_version save hoti hai
taaki pata rahe kaun se model ne kya bola. Purani database par naye columns `init_db()` apne aap add karta hai.

## Honest limitations (poochne par bolna)
- Data synthetic hai, real plant data par accuracy alag hogi.
- Anomaly detector aur RUL supporting / experimental hain.
- Login / authentication aur CSRF protection nahi hai (single user demo).
- Flask development server hai, production me gunicorn/waitress ke peeche chalana chahiye.
- Explanation SHAP nahi, simple what-if hai.
- Test set chhota hai aur uska failure rate train se alag hai, isliye metrics me variance ho sakta hai.

## 20. Failure type model ka score kitna reliable hai?
Sirf failure rows par train (163) aur test (133) hota hai. Weighted F1 0.95 hai kyunki 108 test failures Heat
Dissipation type ke hain. Macro F1 0.79 hai kyunki Power (2), Random (4), Tool Wear (6) ke examples bahut kam hain,
isliye unke individual scores par bharosa kam hai.

## 21. Real AI4I dataset par kya hua?
Same pipeline chalayi (`import_ai4i.py` se convert karke): F1 ~0.79, ROC-AUC 0.984. Synthetic (F1 ~0.90) se kam hai
kyunki AI4I ki rows independent hain, machine history nahi hai, isliye rolling/trend features kam kaam aate hain.

## 22. SECRET_KEY aur model version
SECRET_KEY environment variable se aati hai, nahi to har run par random bani hai. Model file me scikit-learn ka version
saved hai; mismatch par app default model ko khud dobara train kar leti hai.
