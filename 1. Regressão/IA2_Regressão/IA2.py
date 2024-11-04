import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("aerogerador.dat") 
x_data, y_data = data[:, 0], data[:, 1] 

plt.scatter(x_data, y_data)
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.title('Gráfico de Dispersão - Velocidade do Vento vs Potência Gerada')
plt.show()

n = len(x_data)
x = np.column_stack((np.ones(n), x_data)) 

lambdas = [0,0.25,0.5,0.75, 1] 
NRodadas = 500 
rssMqo = []
rssMedia = []
rssTikhonov = {l: [] for l in lambdas} 

for _ in range(NRodadas):
    indices = np.arange(n)
    np.random.shuffle(indices)
    dividirIndex = int(0.8 * n) 

    treinoIdx, testeIdx = indices[:dividirIndex], indices[dividirIndex:] 
    xTreino, xTeste = x[treinoIdx], x[testeIdx] 
    yTreino, yTeste = y_data[treinoIdx], y_data[testeIdx]

    bMqo = np.linalg.inv(xTreino.T @ xTreino) @ xTreino.T @ yTreino 
    yPredMqo = xTeste @ bMqo
    rssMqo.append (np.sum((yTeste - yPredMqo) ** 2))

    for l in lambdas:
        bTikhonov = np.linalg.inv(xTreino.T @ xTreino + l * np.eye(xTreino.shape[1])) @ xTreino.T @ yTreino 
        yPredTik = xTeste @ bTikhonov 
        rssTikhonov[l].append(np.sum((yTeste - yPredTik) ** 2))

    yMediaPred = np.mean(yTreino) 
    rssMedia.append(np.sum((yTeste - yMediaPred) ** 2)) 
def calculoMedia (rssLista): 
    return np.mean(rssLista), np.std(rssLista), np.min(rssLista), np.max(rssLista)

estatisticasMqo = calculoMedia(rssMqo) 
estatisticasMedia = calculoMedia(rssMedia)
estatisticasTikhonov = {l: calculoMedia(rss) for l, rss in rssTikhonov.items()} 

print(f"{"Modelo" :<20} {"Média RSS " :<10} {"Desvio Padrão RSS " :<15} {"Min RSS " :<10} {"Max RSS" :<10}")
print(f"{"MQO" :<20} {estatisticasMqo[0] :<10.2f} {estatisticasMqo[1] :15.2f} {estatisticasMqo[2] :10.2f} {estatisticasMqo[3] :<10.2f}")

for l in lambdas:
    estatistica = estatisticasTikhonov[l]
    print(f"Tikhonov lambda = {l:<7} {estatistica[0] :10.2f} {estatistica[1] :<15.2}{estatistica[2] :<10.2} {estatistica[3] :<10.2}")

print(f"{"Média" :<20} {estatisticasMedia[0] :<10.2f} {estatisticasMedia[1] :15.2f} {estatisticasMedia[2] :10.2f} {estatisticasMedia[3] :<10.2f}")

rotulos = ["MQO"] + [f"Tikhonov λ={l}" for l in lambdas] + ["Média"] #Rotular cada modelo
medias = [estatisticasMqo[0]] + [estatisticasTikhonov[l][0] for l in lambdas] + [estatisticasMedia[0]]
desvioPadrao = [estatisticasMqo[1]] + [estatisticasTikhonov[l][1] for l in lambdas] + [estatisticasMedia[1]] #cria lista DesvioPadrão

plt.bar(rotulos, medias, yerr=desvioPadrao, capsize = 5)
plt.ylabel("Média RSS com Desvio Padrão")
plt.title("Desempenho Dos Modelos de Regressão")
plt.xticks(rotation = 45)
plt.show()

