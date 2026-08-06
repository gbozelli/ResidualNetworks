# ResidualNetworks

## Objetivo
Este repositório integra análise de constelações de 16-QAM e modelos de redes neurais para compensação de distorções em sistemas ópticos. O foco é comparar:

- **MLP direto**: regressão de símbolos sem conexão residual.
- **ResNet residual**: predição do desvio entre o símbolo recebido central e o símbolo transmitido.
- **NSGA2 + redes neurais**: otimização multiobjetivo de BER e complexidade computacional.

## Estrutura do repositório

- `resnet.py`: script exploratório para visualização de dados de constelação e primeiras tentativas de ResNet.
- `data_loader.py`: utilitários para carregar diagramas de constelação e normalizar dados.
- `utils.py`: funções de pré-processamento, janelamento e cálculo de BER.
- `normal_nn.py`: exemplo de uso de rede neural normal (MLP) para mapeamento direto.
- `nsga2_nn.py`: exemplo de uso de NSGA2 com redes neurais ResNet.
- `requirements.txt`: dependências mínimas para reproduzir os exemplos.

## Dados de constelações

O conjunto de dados base deve conter arquivos do tipo:

- `DP_RealConstellationDiagram_{dbm}.csv`
- `DP_IdealConstellationDiagram_0dbm_{distance}.csv`

Cada arquivo representa as componentes dual-polarization `XI`, `XQ`, `YI`, `YQ` de um sistema óptico. As amostras de treinamento usam essas quatro dimensões como entrada e alvo.

### Tipos de dados

- `real_data`: sinais recebidos afetados por ruído e distorção.
- `ideal_data`: constelações de referência sem ruído, usadas como alvo de regressão.

## Métodos implementados

### 1. Rede neural normal (MLP direto)

O modelo implementado em `normal_nn.py` utiliza `sklearn.neural_network.MLPRegressor` para aprender uma função direta: 

`received -> transmitted`.

- Entrada: vetores planos de dimensão `4` (XI, XQ, YI, YQ).
- Saída: regressão dos mesmos quatro componentes.
- Métrica: BER estimado por quantização de símbolos e comparação de bits Gray.

### 2. ResNet residual

O modelo em `nsga2_nn.py` usa um bloco MLP com conexão residual:

- A rede prediz o **resíduo** entre o símbolo central e o símbolo desejado.
- A saída final é `x_center + residual`.
- Esse paradigma reduz a carga da rede e melhora a robustez quando o desvio é pequeno.

### 3. NSGA2 multiobjetivo

O fluxo de `nsga2_nn.py` demonstra como usar NSGA2 para otimizar dois objetivos:

- **BER**: qualidade da reconstrução de símbolos.
- **FLOPs**: complexidade do modelo medida em operações de multiplicação-acumulação.

A otimização retorna uma fronteira de Pareto explícita que equilibra precisão e eficiência.

## Como usar

### Instalação

```bash
python3 -m pip install -r requirements.txt
```

### Exemplo inicial para iniciantes

Para começar sem precisar de dados reais, execute a partir da raiz do projeto:

```bash
cd /home/bozelli/ResidualNetworks
. .venv/bin/activate
python3 example_usage.py
```

Esse script usa um conjunto de símbolos 16-QAM sintéticos e mostra:

- como construir um diagrama de constelação;
- como treinar uma rede neural simples;
- como estimar o BER de forma didática.

### Exemplo 1: MLP direto

```bash
python3 -m ResidualNetworks.normal_nn
```

### Exemplo 2: NSGA2 + ResNet

```bash
python3 -m ResidualNetworks.nsga2_nn
```

> Se você tiver dados reais, coloque-os em `data/` ou ajuste os caminhos no código antes de executar.

## Guia para iniciantes

Se você nunca trabalhou com programação ou comunicações ópticas digitais, comece por ler:

- `BEGINNER_GUIDE.md`: explicações simples sobre o que é uma constelação, BER e o papel das redes neurais.
- `example_usage.py`: um tutorial passo a passo com código comentado.

## Métricas e avaliação

### BER estimado

A BER é calculada a partir de uma quantização de 4-PAM para cada componente de polarização, convertendo símbolos em bits Gray e comparando bit a bit.

### Análise de trade-off

O uso de NSGA2 permite estudar:

- Modelos compactos com FLOPs baixos, mas BER aceitável.
- Modelos mais complexos com BER reduzido ao custo de maior custo computacional.

## Reprodutibilidade

- Os scripts estão organizados para separar carregamento de dados, modelagem e avaliação.
- `requirements.txt` lista dependências mínimas.
- O código está documentado para facilitar experimentos científicos e relatórios.
- Instale o pacote local para usar o comando `python -m ResidualNetworks.example_usage`.

### Instalar como pacote local

```bash
cd /home/bozelli/ResidualNetworks
. .venv/bin/activate
python3 -m pip install -e .
```

Depois disso, você pode executar o tutorial como pacote:

```bash
cd /home/bozelli
python3 -m ResidualNetworks.example_usage
```

## Referências internas

- `NSGA2/`: notebooks de otimização e comparação de MLPs com NSGA2.
- `Pesquisa/25-26/resnets/NSGA2_ResNet.py`: exemplo de integração entre NSGA2 e redes residuais.

## Próximos passos

1. Adicionar dataset `data/` com arquivos CSV de constelação.
2. Ajustar `load_dual_polarization_dataset` aos nomes e formatos exatos de seu conjunto.
3. Executar `normal_nn.py` para validar o comportamento direto.
4. Executar `nsga2_nn.py` para encontrar uma fronteira de Pareto entre BER e FLOPs.
