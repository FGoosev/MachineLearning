import numpy as np

#Функция активации
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#Производная функции активации
def deriv_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)

class OurNeuralNetwork:

  def __init__(self):
    # Веса
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()
    self.w7 = np.random.normal()
    self.w8 = np.random.normal()
    self.w9 = np.random.normal()
    self.w10 = np.random.normal()
    self.w11 = np.random.normal()
    self.w12 = np.random.normal()
    self.w13 = np.random.normal()
    self.w14 = np.random.normal()
    self.w15 = np.random.normal()
    self.w16 = np.random.normal()
    self.w17 = np.random.normal()
    self.w18 = np.random.normal()
    self.w19 = np.random.normal()
    self.w20 = np.random.normal()



    # Пороги
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()
    self.b4 = np.random.normal()
  #прямое распространение. Проходим по сети и возвращаем результат
  def feedforward(self, x):

    h1 = sigmoid(self.w1 * x[0] + self.w5 * x[1] + self.w9 * x[2] + self.w13 * x[3] + self.b1)
    h2 = sigmoid(self.w2 * x[0] + self.w6 * x[1] + self.w10 * x[2] + self.w14 * x[3] + self.b2)
    h3 = sigmoid(self.w3 * x[0] + self.w7 * x[1] + self.w11 * x[2] + self.w15 * x[3] + self.b3)
    h4 = sigmoid(self.w4 * x[0] + self.w8 * x[1] + self.w12 * x[2] + self.w16 * x[3] + self.b4)

    o1 = sigmoid(self.w17 * h1 + self.w18 * h2 + self.w19 * h3 + self.w20 * h4 + self.b4)


    return o1

  def train(self, data, all_y_trues):

    learn_rate = 0.1
    epochs = 1000

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        #Устанавливаем значения в скрытый слой
        sum_h1 = self.w1 * x[0] + self.w5 * x[1] + self.w9 * x[2] + self.w13 * x[3] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w2 * x[0] + self.w6 * x[1] + self.w10 * x[2] + self.w14 * x[3] + self.b2
        h2 = sigmoid(sum_h2)

        sum_h3 = self.w3 * x[0] + self.w7 * x[1] + self.w11 * x[2] + self.w15 * x[3] + self.b3
        h3 = sigmoid(sum_h3)

        sum_h4 = self.w4 * x[0] + self.w8 * x[1] + self.w12 * x[2] + self.w16 * x[3] + self.b4
        h4 = sigmoid(sum_h4)

        #получаем значение и устанавливаем в выходной нейрон
        sum_o1 = self.w17 * h1 + self.w18 * h2 + self.w19 * h3 + self.w20 * h4 + self.b4
        o1 = sigmoid(sum_o1)
        #предсказание выходного числа
        y_predOne = o1

        #высчитываем ошибку
        d_L_d_ypred = -2 * (y_true - y_predOne)

        #Начинаем выполнять обратное распространение
        # Neuron o1
        d_ypred_d_w17 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w18 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_w19 = h3 * deriv_sigmoid(sum_o1)
        d_ypred_d_w20 = h4 * deriv_sigmoid(sum_o1)
        d_ypred_d_b4 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w17 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w18 * deriv_sigmoid(sum_o1)
        d_ypred_d_h3 = self.w19 * deriv_sigmoid(sum_o1)
        d_ypred_d_h4 = self.w20 * deriv_sigmoid(sum_o1)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w5 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_w9 = x[2] * deriv_sigmoid(sum_h1)
        d_h1_d_w13 = x[3] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Neuron h2
        d_h2_d_w2 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w6 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_w10 = x[2] * deriv_sigmoid(sum_h2)
        d_h2_d_w14 = x[3] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # Neuron h3
        d_h3_d_w3 = x[0] * deriv_sigmoid(sum_h3)
        d_h3_d_w7 = x[1] * deriv_sigmoid(sum_h3)
        d_h3_d_w11 = x[2] * deriv_sigmoid(sum_h3)
        d_h3_d_w15 = x[3] * deriv_sigmoid(sum_h3)
        d_h3_d_b3 = deriv_sigmoid(sum_h3)

        # Neuron h4
        d_h4_d_w4 = x[0] * deriv_sigmoid(sum_h4)
        d_h4_d_w8 = x[1] * deriv_sigmoid(sum_h4)
        d_h4_d_w12 = x[2] * deriv_sigmoid(sum_h4)
        d_h4_d_w16 = x[3] * deriv_sigmoid(sum_h4)
        d_h4_d_b4 = deriv_sigmoid(sum_h4)

        # Обновляем все значения, исходя из тех данных, которые получили ранее
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w5
        self.w9 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w9
        self.w13 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w13
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w2
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
        self.w10 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w10
        self.w14 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w14
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Neuron h3
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w3
        self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w7
        self.w11 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w11
        self.w15 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w15
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_b3

        # Neuron h4
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w4
        self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w8
        self.w12 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w12
        self.w16 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w16
        self.b4 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_b4

        # Neuron o1
        self.w17 -= learn_rate * d_L_d_ypred * d_ypred_d_w17
        self.w18 -= learn_rate * d_L_d_ypred * d_ypred_d_w18
        self.w19 -= learn_rate * d_L_d_ypred * d_ypred_d_w19
        self.w20 -= learn_rate * d_L_d_ypred * d_ypred_d_w20
        self.b4 -= learn_rate * d_L_d_ypred * d_ypred_d_b4



#Обучающая выборка
data = np.array([[3,2,3,1],[2,3,3,1],[3,3,3,1],[2,2,3,2],[3,2,3,1], # Легковая
                 [4,4,5,5],[4,5,4,5],[5,5,5,4],[5,5,4,5],[5,5,5,4], # Паркетник
                 [7,6,7,6],[6,6,6,7],[6,6,7,6],[6,6,6,6],[7,6,7,6], # Внедорожник
                 [8,8,8,8],[8,9,8,9],[8,9,9,8],[9,8,8,9],[8,8,8,9]]) # Грузовая

#Ожидаемый результат от обучающей выборки
all_y_trues = np.array([0, 0, 0, 0, 0,
                        0.4, 0.4, 0.4, 0.4, 0.4,
                        0.7, 0.7, 0.7, 0.7, 0.7,
                        1, 1, 1, 1, 1])

# Train
network = OurNeuralNetwork()
network.train(data, all_y_trues)


auto = np.array([7,6,6,6])
train = network.feedforward(auto)
print(train)

if train < 0.2:
  print("Легковой автомобиль")
elif train < 0.8 and train >= 0.2:
  print("Паркетник")
elif train < 0.93 and train >= 0.8:
  print("Внедорожник")
else:
  print("Грузовой автомобиль")


