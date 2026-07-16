# Predição de Risco de Crédito com Aprendizado de Máquina Supervisionado

Este repositório contém o trabalho final desenvolvido para a disciplina de Inteligência Artificial da Universidade Federal do Piauí (UFPI). O objetivo do projeto é construir um pipeline completo de ciência de dados para classificar potenciais tomadores de crédito em categorias de Baixo Risco ou Alto Risco.

## 📌 Contextualização do Problema
A concessão de crédito é uma das atividades mais críticas do setor financeiro nacional. Identificar com precisão os riscos de inadimplência protege as instituições de perdas severas. Este projeto aborda o problema sob a ótica de classificação binária complexa, lidando diretamente com o desafio de **dados desbalanceados** (onde o número de inadimplentes é significativamente menor que o de bons pagadores).

## 🛠️ Tecnologias e Bibliotecas Utilizadas
*   **Python 3.10**
*   **Pandas** e **NumPy**: Manipulação de matrizes e estruturação dos dados tabulares.
*   **Scikit-Learn**: Divisão amostral, escalonamento e implementação dos algoritmos de Machine Learning.

## 📊 Pipeline de Engenharia de Dados
O pré-processamento dos dados brutos seguiu as seguintes etapas automatizadas no script:
1. **Tratamento de Dados Ausentes:** Imputação de valores nulos utilizando a mediana amostral para mitigar o impacto dos *outliers*.
2. **Codificação Categórica:** Aplicação de *One-Hot Encoding* em variáveis nominais (`tipo_moradia`).
3. **Escalonamento de Atributos:** Padronização estatística via *StandardScaler* (média 0, variância de 1) das variáveis contínuas, garantindo a convergência dos algoritmos baseados em distância e gradiente.

## 🤖 Modelos Avaliados e Resultados
Foram testados três algoritmos com abordagens matemáticas distintas. Devido à assimetria das classes, a avaliação priorizou métricas mais rigorosas como **Recall (Sensibilidade)** e **AUC-ROC** em vez da acurácia global:

| Algoritmo Classificador | Acurácia | Precisão (Classe 1) | Recall (Classe 1) | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística** | 82,1% | 51,2% | 12,4% | 20,0% | 0,642 |
| **Árvore de Decisão (Otimizada)** | **84,6%** | **63,8%** | **41,2%** | **50,1%** | **0,718** |
| **K-Nearest Neighbors (k-NN)** | 80,5% | 43,1% | 23,5% | 30,4% | 0,611 |

### Conclusão dos Experimentos
A **Árvore de Decisão Otimizada** apresentou o melhor desempenho global. Ela demonstrou maior capacidade de capturar a inadimplência latente (Recall de 41,2%), enquanto modelos lineares como a Regressão Logística falharam por serem severamente afetados pelo desbalanceamento natural da base de dados.

## 🚀 Como Executar o Projeto

1. Clone este repositório:
```bash
   git clone [https://github.com/helderedias/predicao-risco-credito.git](https://github.com/helderedias/predicao-risco-credito.git)
