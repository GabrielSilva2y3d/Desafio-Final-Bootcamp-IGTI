import pandas as pd
from sklearn.preprocessing import MinMaxScaler
arquivo = pd.read_csv('/content/pima-indians-diabetes.csv',header=None)
arquivo

y = arquivo[8]
x1 = arquivo.drop(8, axis = 1)


normaliza = MinMaxScaler() #objeto para a normalização
x=normaliza.fit_transform(x1)

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.30,random_state=42)

from sklearn.neural_network import MLPClassifier
modelo = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,
5), random_state=1)
modelo.fit(x_treino,y_treino)

resultado = modelo.score(x_teste, y_teste)
resultado

resultado2 = modelo.predict(x_teste)
resultado2

import joblib

#salvando o melhor modelo no disco
arquivo_salvo = 'melhor_modelo.sav'
joblib.dump('melhor_modelo', arquivo_salvo)

modelo_salvo = joblib.load(arquivo_salvo)

import numpy as np
from flask import Flask, request, jsoninfy, render_template

app = Flask(__name__)

def previsao_diabets(lista,valores,formulario):
  prever= formulario.reshape(1,8)
  modelo_salvo = joblib.load(arquivo_salvo)
  resultado = valores.predict(prever)
  return resultado[0]

  @app.route('/')
  def home():
    return render_template('index.html')

  @app.route('/')
  def result():
    if  request.method == 'POST':
      lista_formulario = request.form.to.dict()
      lista_formulario = list(lista_formulario.values())
      lista_formulario = list(map())
      resultado = previsao_diabets(lista_formulario)
      if int(resultado) == 1:
        previsao = 'Possui diabets'
      else:
        previsao = 'Não possiu diabets'

      return render_template('resultado.html', previsao = previsao)

if __name__ == '__main__':
  app.run(debug = True)