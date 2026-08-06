# Guia para iniciantes

Este guia foi criado para quem não tem experiência em programação ou em sistemas de comunicação óptica digital.

## O que é um diagrama de constelação?

Em sistemas de comunicação digital, informações são transmitidas usando sinais analógicos que mudam de amplitude e fase. Um diagrama de constelação mostra esses sinais em um plano:

- eixo horizontal: componente "in-phase" (I)
- eixo vertical: componente "quadrature" (Q)

Para um esquema 16-QAM, existem 16 pontos possíveis. Cada ponto representa um conjunto de bits.

## O que é BER?

BER significa **Bit Error Ratio**, ou razão de erro de bits. É uma medida de quantos bits foram traduzidos incorretamente após a recepção:

- 0% = tradução perfeita
- 1% = 1 em cada 100 bits foi errado

No contexto deste projeto, usamos uma estimativa de BER em que os valores contínuos do sinal são convertidos de volta em símbolos 16-QAM.

## Por que usar redes neurais?

Redes neurais podem aprender a compensar distorções e ruídos que ocorrem durante a transmissão óptica.

- Uma **rede neural normal** tenta aprender a mapear diretamente o sinal recebido para o sinal transmitido.
- Uma **rede residual** (ResNet) tenta aprender apenas o erro entre o sinal recebido e o sinal ideal, o que geralmente facilita o aprendizado.

## O que é NSGA2?

NSGA2 é um algoritmo de otimização que busca duas metas ao mesmo tempo:

- reduzir a razão de erro (BER)
- reduzir o custo computacional do modelo (FLOPs)

Isso é útil quando queremos um modelo que seja ao mesmo tempo preciso e leve.

## Passo a passo rápido

1. Instale as dependências:

```bash
python3 -m pip install -r requirements.txt
```

2. Rode o exemplo didático:

```bash
python3 -m ResidualNetworks.example_usage
```

3. Experimente com dados reais colocando arquivos CSV em `data/`.

## Organização dos arquivos

- `README.md`: visão geral do projeto e comandos de uso.
- `BEGINNER_GUIDE.md`: explicações básicas.
- `example_usage.py`: tutorial em código.
- `normal_nn.py`: rede neural direta.
- `nsga2_nn.py`: NSGA2 com rede residual.
- `data_loader.py`: carregamento de dados de constelação.
- `utils.py`: campos auxiliares, janelamento e cálculo de BER.
