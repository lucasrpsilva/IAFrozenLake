import numpy as np
import gym
import random

env = gym.make("FrozenLake-v0") #Cria o ambiente

#Pega o número de ações possíveis e estados totais para montar a Qtable
action_size = env.action_space.n 
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))

total_episodes = 15000 #Número de iterações para treinar o algoritmo
learning_rate = 0.8 #Taxa de aprendizado (Define o peso da recompensa de cada ação, portanto quanto mais próximo de 1 maior será o valor da recompensa por uma ação)
max_steps = 99 #Número de passos máximo por iteração
gamma = 0.95 #Fator de desconto (Quantifica a importância de recompensas futuras. Quanto mais próximo de 1, o algoritmo estará mais disposto a atrasar uma recompensa para que ela seja maior no futuro)

#Parâmetros de exploração
epsilon = 1.0 #Taxa de exploração desejada (Se a recompensa estimada for menor que este valor, a exploração terá preferência ao invés da reclamação da recompensa)
max_epsilon = 1.0 #Taxa de exploração máxima
min_epsilon = 0.01 #Taxa de exploração mínima
decay_rate = 0.005 #Taxa de subtração da exploração

rewards = []

#Repetição até que o número máximo de passos seja atingido
for episode in range(total_episodes):
    
    #Reinicia o ambiente
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0, 1) #Estipula a recompensa imediata
        
        #Se a recompensa for maior que a exploração, aproveite-a
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])

        #Se não, explore
        else:
            action = env.action_space.sample()

        #Realiza a ação e pega os resultados obtidos dela
        new_state, reward, done, info = env.step(action)

        #Atualiza a Qtable de acordo com os dados obtidos
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
        total_rewards += reward #Incrementa o total das recompensas obtidas
        state = new_state #Atualiza o estado
        
        #Caso o aprendizado já tenha acabado, isto é, a Qtable está completamente preenchida, este passo acabada
        if done == True: 
            break
        
    #Reduz a exploração
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards.append(total_rewards)

env.reset()

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("-------------------------------------------------------")
    print("ITERACAO ", episode+1)

    for step in range(max_steps):
        
        #Pega a ação que dará a maior recompensa para o estado atual
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            env.render()
            print("\nNumero de passos: ", step)
            break
        state = new_state
env.close()