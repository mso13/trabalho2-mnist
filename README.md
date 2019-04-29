## Trabalho 2 - Tópicos em Engenharia (IA)
**Segundo trabalho de implementação computacional 1/2019** 

##### Grupo:

- Matheus Schmitz Oliveira - 15/0018371
- Pedro Aurélio Coelho de Almeida - 14/0158103

##### Bibliotecas utilizadas:

1. *Tensorflow*
2. *Mahotas*

```python
# Para instalar as bibliotecas utilizadas, abra o terminal neste diretório
# e digite o seguinte comando:

pip install -r requirements.txt
```



#### ***Rede "Perceptron Multi-camada" e base de dados "MNIST"***

1. ##### Introdução

   O(a) estudante deverá demonstrar conhecer a Rede Neural Artificial conhecida como
   "Backpropagation" (ou "Perceptron Multicamada"), aplicando-a na solução de um problema de reconhecimento de padrões.
   
   Usaremos a base de dados "MNIST", disponível em http://yann.lecun.com/exdb/mnist/ . Trata-se de um conjunto de imagens manuscritas  dos dígitos de 0 a 9, em níveis de cinza. Cada imagem é um conjunto de 28x28 pixels em níveis de cinza de 0 (branco) a 255 (preto). As    imagens foram centralizadas pelo método do centro de gravidade. Há 60000 imagens para treinamento e 10000 para teste. Outros detalhes do  pré-processamento e dos formatos dos arquivos, bem como os arquivos propriamente ditos, podem ser buscados no sítio indicado acima.
   
2. ##### Requisitos Básicos

   A) Demonstrar um código computacional capaz de:
   
   A.1) Ler os arquivos de entrada (ou algum outro que tenha sido preparado a partir deste para facilitar a montagem da rede). Os dados propriamente ditos (quantidade e conteúdo das imagens) não deve ser modificado, para efeito de comparação, mas pré-processamento adicional dos dados é permitido.

   A.2) Treinar uma rede *Backpropagation* com uma camada escondida, usando algoritmo padrão.

   A.3) Testar a rede contra a saída desejada em um arquivo de teste, informando a taxa de erro (porcentagem de exemplos erradamente classificados).

   B) Fazer uma análise dos resultados obtidos, verificando a evolução da taxa de erro (e o erro quadrático) no arquivo de treinamento, a taxa de erro no arquivo de teste, as dificuldades encontradas, as soluções propostas, os valores usados para os parâmetros de treinamento. O arquivo de teste não deve ser usado para nenhuma otimização da rede. Apenas para testar uma rede já treinada.

3. ##### **Requisitos adicionais (pontuação extra)**

   - [ ] Uso de múltiplas camadas escondidas
   - [ ] Uso de Entropia Cruzada como função de custo.
   - [ ] Uso de Regularização (*L1, L2, Dropout*)
   - [ ] Uso de saída em camada *softmax* (Regressão Logística) e custo *log-likelihood*.
   - [ ] Uso de sequências de camadas RBM com treinamento não-supervisionado (*Contrast*
     *Divergenc*e)
   - [ ]  Teste com padrões adicionais (obtidos pelos autores)

4. ##### Regras Gerais e Observações

   A) Não será pré-definida uma linguagem de programação. 

   B)  Observe, entre as informações contidas no sítio indicado, o desempenho de MLPs com uma camada escondida situa-se na faixa de 4% de erro. Não se espera necessariamente alcançar, em um primeiro trabalho de implementação, desempenho igual ou superior a este, mas pode-se tomá-lo como um parâmetro de comparação.

   C) Apresentar o código, os resultados obtidos e a análise.

------
