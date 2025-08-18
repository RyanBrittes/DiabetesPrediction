### :chart_with_upwards_trend: Regressão Logística:
> É uma técnica estatística utilizada para encontrar a probabilidade de um determinado resultado ser 0 ou 1, ou seja, a possibilidade de ser verdadeiro ou falso. Para isto utilizaremos a função de sigmoide que nos auxiliará na conversão dos resultados para um valor entre 0 e 1.

- Na Regressão Logística utilizaremos alguns conceitos técnicos que nos auxiliarão a encontrar nossos resultados:
  1. Equação de Regressão Logística
  2. Perda
  3. Limite de classificação/ Matriz de confusão
  4. Gradiente descendente
  5. Hiperparâmetros
1. :heavy_division_sign: **Equação de Regressão Logística**
> $z = b + w_1 * x_1 + w_2 * x_2 + w_n * x_n ...$
  - Onde:
    - b: viés
    - w: peso
    - x: atributo de entrada
    - z: saída linear
  - Caso sejam dados com vários atributos, adicionamos um peso para cada tipo de atributo, como representado por $w_n * x_n$
  - Após o cálculo da saída linear iremos transformar este valor em uma saída entre 0 e 1 utilizando a função de sigmoide:
> $y' = 1 / (1 + e^{-z})$
 - Onde:
    - y': saída probabilística
    - z: saída linear 
2. :chart_with_downwards_trend: **Perda:**
> Representa uma forma de quantificarmos o quão longe o algoritmo está nas suas predições com os dados reais, ou seja, o quanto está errando.
  - _Log Loss:_ Tipo utilizado para quantificar perdas de dados no contexto de dados probabilísticos entre 0 e 1:
> $logLoss=∑_{(x,y)∈D}-ylog(y')-(1-y)log(1-y')$
 - Onde:
    - $(x, y)∈D$: conjunto de dados que contém muitos exemplos rotulados, que são (x, y) pares
    - y: valor de rótulo de um exemplo
    - y': previsão do algoritmo com os mesmos parâmetros de y
3. :chart_with_downwards_trend: **Limite de Classificação/ Matriz de Confusão:**
> É um parâmetro utilizado  para ajustar a precisão que as previsões devem atender, criamos um limite entre 0 e 1 para determinar a partir de que ponto consideramos algo positivo e algo negativo. Com este limite iremos conseguir criar 04 possíveis cenários de resultados, que chamamos de Matriz de Confusão, esta provê informações sobre as saídas do algoritmo que nos auxiliará a definir suas métricas posteriormente.
  - Classificações Negativas: Quando algo é considerado negativo significa que é falso, então o algoritmo classificará como 0.
  - Classificações Positivas: Quando algo é considerado positivo significa que é verdadeiro, então o algoritmo classificará como 1.
  - Matriz de confusão: Baseado nos resultados, teremos as quatro possíveis saídas:
    - Positivo Verdadeiro: Quando o modelo prevê corretamente o estado positivo
    - Negativo Verdadeiro: Quando o modelo prevê corretamente o estado negativo
    - Positivo Falso: Quando o modelo prevê erroneamente o estado positivo
    - Negativo Falso: Quando o modelo prevê erroneamente o estado negativo
  4. :arrow_heading_down: **Gradiente descendente:**
> Técnica iterativa que faz cálculos e ajusta o viés e os pesos até encontrar valores com perdas muito parecidas, ou seja, quando a perda não muda mais de maneira significativa com o tempo das iterações. Podemos dizer que este loop de iteração é onde o treinamento é feito.
  - Está técnica consiste em iniciar com viés e pesos com valores próximos de zero, o usuário escolhe um número de iterações e de acordo com ele o modelo irá tentar encontrar um melhor valor para os pesos e vies que irá se basear em reduzir a perda, o objetivo é reduzir ao máximo possível a perda.
  - Este conceito funciona como uma parábola, onde o seu pico máximo é denominado convergência (ponto em que o modelo não tem mudanças significativas em suas perdas). Quando o modelo chega na convergência falamos que ele convergiu e chegamos em nosso objetivo.
5. :earth_americas: **Hiperparâmetros:**
> Variáveis que nos possibilitam controlar diferentes aspectos do treinamento, temos:
  - _Taxa de aprendizado:_ velocidade com que o modelo atualiza os pesos, valor pré-definido para controlar como o modelo irá aprender. Nele existem particularidades que precisamos nos atentar como:
    - Não deixar o valor muito alto, pois se não os resultados são muito inconstantes e o modelo por consequência nunca irá convergir.
    - Não deixar o valor muito baixo, pois quando menor for maior será o tempo para o modelo convergir.
    - Por tando, é necessário realizar testes e utilizar o valor que mais se adeque ao modelo.
  - _Tamanho do Lote:_ número de amostrar que o exemplo irá processar antes de atualizar os pesos. Pode ser pouco prático utilizar apenas um exemplo por vez antes de atualizar os pesos, por tanto podemos fazer isso de maneira mais eficiente e colocar um número de exemplos por iteração, definindo assim lotes para que o modelo processe os pesos antes de atualizar. Temos duas formas de fazer esse processo:
    - Utilizando o Grandiente descendente estocástico (SGD), onde utilizamos apenas um exemplo por iteração escolhido de forma aleatória. Este resulta em resultados ruidosos.
    - Utilizando o Grandiente descendente estocástico com mini-lotes, onde utilizamos um número de amostras por iteração escolhidas de forma aleatória antes da atualização dos pesos. Os resultados apresentam menos ruido.
  - _Eras:_ É a definição de quantas vezes o modelo irá processar todos os exemplos do conjunto, então se dizermos que Eras = 1, o modelo processou todas as amostras uma vez. Quanto maior o número de Eras melhor o modelo pode ficar, porém o tempo de treinamento também será maior.