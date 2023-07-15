# Manipulação e analise de dados
import pandas as pd 
#usado para plotar grafico
import matplotlib.pyplot as plt

#Preprocessanento
from sklearn.preprocessing import label_binarize

# Metrics for Evaluation of model
from sklearn.metrics  import roc_curve, auc, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay, classification_report
 
# Importa a modelo de classificação binário
from sklearn.linear_model import LogisticRegression
 
# Usado para dividr os dados entre treio e teste
from sklearn.model_selection import train_test_split

data_unformatted = pd.read_csv('wdbc.data', header=None)
data_unformatted.drop(labels=0, axis='columns', inplace=True)#Remove os ID

target = data_unformatted[1]
data_unformatted.drop(labels=1, axis='columns', inplace=True)#Remove coluna 'target'

target = label_binarize(target, classes=['M', 'B'])#Converte M = 0 e B = 1
features = data_unformatted.copy()

#Divide entre treino e teste, teste é 30%
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

#Treina o modelo de classificação binário
model = LogisticRegression(random_state=0).fit(X_train,y_train)

print(f'Model Accuracy : {model.score(X_test,y_test)}')


#Preve com base no teste
y_predicted = model.predict(X_test)

#Gera curva ROC e AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
auc_score = auc(fpr, tpr)

print(f'AUC score : {auc_score}')

fig,ax = plt.subplots(figsize=(5,5))
plt.title(label="ROC Curve", loc='center')

RocCurveDisplay.from_predictions(y_test, y_predicted,ax=ax, name='ROC for M,B')
plt.show()

cm = confusion_matrix(y_test, y_predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign','Malignant'])
disp.plot()
plt.title(label='Confusion Matrix')
plt.show()


print(classification_report(y_test, y_predicted))