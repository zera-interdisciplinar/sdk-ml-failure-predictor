# sdk-ml-failure-predictor

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Modelo preditivo de quebra do sistema Zera: uma regressão em PyTorch que estima **em quantos meses
um equipamento quebra**, contados da data de aquisição (`acquiredAt`), não de hoje. O mesmo processo
serve as ferramentas MCP e o `POST /predict-batch`, na porta 8000. É consumido pelo
[`ms-inventory`](https://github.com/zera-interdisciplinar/ms-inventory) (atualização diária da data
prevista de quebra de cada item) e pelo `ms-artificial-intelligence-core` (assistente, via MCP).

## Sumário

- [Domínio](#domínio)
- [Arquitetura](#arquitetura)
- [Stack](#stack)
- [Executando localmente](#executando-localmente)
- [Configuração](#configuração)
- [API](#api)
- [Modelo e bundle](#modelo-e-bundle)
- [Testes](#testes)
- [Deploy](#deploy)

## Domínio

Entram no modelo `category`, `manufacturer`, `model`, `climateZone`, `usageIntensity` e a idade na
aquisição. A idade é o número de meses de 1º de janeiro do ano de fabricação até `acquiredAt`
(`dias / 30.44`). Identificadores, `status`, `unitId` e `nextPredictionDate` ficam de fora de
propósito: não carregam sinal preditivo ou vazariam o alvo.

A saída da rede é linear; o valor devolvido é `max(0, saída)`, então nunca é negativo. É uma
estimativa estatística, não uma garantia.

Categoria, fabricante ou modelo fora do vocabulário viram `<UNK>`. O treino mascara 5 % dos índices
categóricos para essa linha do embedding receber gradiente, e 20 % dos nomes de modelo ficam de fora
do treino de propósito — o erro nesse corte vai no manifest como `unseen_model_mae`.

## Arquitetura

```
src/
├── data/        Schema do registro (DeviceRecord), codificação das features (FeatureEncoder) e Dataset
├── models/      TTFModel: embeddings das 4 categóricas + MLP [64, 32] sobre as numéricas
├── training/    config.yaml e train.py (treino, métricas, gráficos e gravação do bundle)
└── serving/     Servidor MCP + POST /predict-batch, e o TTFPredictor que carrega o bundle
checkpoints/
└── serving/     O bundle que o servidor lê (modelo, encoder e manifest)
k8s/             Manifests de QA e produção
tests/           Testes do modelo, do bundle e do contrato
```

## Stack

| | |
|---|---|
| Linguagem | **Python 3.12** |
| Modelo | PyTorch (embeddings + MLP, `LeakyReLU`) |
| Serving | SDK `mcp` 2.x, streamable HTTP (sem estado) + rota HTTP simples |
| Validação | pydantic 2 |
| Treino | numpy, pandas, scikit-learn, matplotlib, Faker |
| Testes | pytest |

## Executando localmente

Pré-requisito: Python 3.12 e **Git LFS** (`git lfs install && git lfs pull`). Os arquivos de
`checkpoints/` são versionados com LFS; sem isso o `manifest.json` é só um ponteiro e o servidor se
recusa a subir.

```bash
pip install -r requirements.txt         # só o necessário para servir
python -m src.serving.mcp_server        # MCP em /mcp e POST /predict-batch, em :8000
```

Para treinar ou rodar os testes, instale o conjunto completo:

```bash
pip install -r requirements-train.txt
python -m src.training.train            # lê src/training/config.yaml
pytest
```

O treino lê `data/raw/devices.json`, que não é versionado. O gerador sintético que costuma
alimentá-lo (`python scripts/synthetic_generator.py --n 100000 --seed 2026`) também **não está no
repositório** — `scripts/` está no `.gitignore` —, então para retreinar é preciso ter esse arquivo
de fora. Ao terminar, o treino regrava `checkpoints/serving/` (o bundle) e os gráficos e relatório
em `checkpoints/`.

## Configuração

O servidor não lê o `config.yaml` (esse é só do treino). Toda a configuração de runtime é por
variável de ambiente, e as duas são opcionais:

| Variável | Default | Efeito |
|---|---|---|
| `MCP_HTTP_PORT` | `8000` | Porta HTTP do servidor |
| `MODEL_DIR` | `checkpoints/serving` | Diretório do bundle; relativo, é resolvido a partir da raiz do repositório |

O serviço **não tem autenticação própria**: quem protege o acesso é o gateway
([`infra-gtw-kong`](https://github.com/zera-interdisciplinar/infra-gtw-kong)), com API key. Não
exponha a porta 8000 diretamente.

## API

- **MCP**, streamable HTTP, em `/mcp`: `predict_time_to_failure`, `predict_time_to_failure_batch`,
  `list_valid_categories`, `list_valid_climate_zones`.
- **REST**: `POST /predict-batch` com `{"devices": [ ... ]}` (1 a 500 itens). A resposta é
  `{"results": [{"estimated_remaining_months", "error"}]}`, na mesma ordem da entrada. Cada item é
  validado de forma independente: um item inválido volta com `error` preenchido e não derruba os
  demais. JSON inválido responde 422.

Cada dispositivo tem os campos `category`, `manufacturer`, `model`, `climateZone`, `usageIntensity`
(inteiro), `manufacturingDate` (ano de fabricação) e `acquiredAt` (data ISO, `YYYY-MM-DD`):

```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"devices": [{"category": "notebook", "manufacturer": "Dell", "model": "Latitude 5420",
        "climateZone": "TROPICAL", "usageIntensity": 5, "manufacturingDate": 2022,
        "acquiredAt": "2023-03-01"}]}'
```

O MCP existe para agentes de IA decidirem em runtime se chamam uma ferramenta; o `/predict-batch`
existe para chamadores determinísticos (como o job do `ms-inventory`), que não precisam acompanhar
a evolução do envelope do protocolo MCP. É o mesmo processo e o mesmo preditor — só roteamento.

## Modelo e bundle

O servidor lê `checkpoints/serving/` (ou o diretório em `MODEL_DIR`):

- `ttf_model.pt` — pesos do epoch com menor MSE de validação
- `feature_encoder.json` — vocabulário e normalização das numéricas
- `manifest.json` — dims, tamanhos de vocabulário, seed, epoch, MAE/RMSE e o sha256 do peso

Se o encoder ou o hash não baterem com o manifest, o processo não sobe. Gráficos e
`checkpoints/data.txt` não vão para a imagem Docker.

O bundle versionado é o treino de 2026-10-17: 100000 linhas, seed 2026, melhor epoch 118 (MSE de
validação 13,66). Vocabulário, com `<UNK>`: 9 categorias, 41 fabricantes, 240 modelos, 5 zonas
climáticas.

| Corte | MAE | RMSE | ±3 meses | MAPE |
| --- | --- | --- | --- | --- |
| Treino | 2,59 meses | 3,51 | 67,6% | 6,5% |
| Validação (modelo já visto) | 2,74 meses | 3,70 | 65,2% | 6,8% |
| Modelo novo | 3,52 meses | 4,70 | 54,7% | 8,8% |

## Testes

```bash
pytest
```

Não dependem de infraestrutura externa. Cobrem o cálculo da idade, a recusa de subir quando o
manifest diverge do encoder ou do hash, o piso de zero na predição e o contrato do batch (JSON
inválido → 422, item inválido sem derrubar os demais). O teste que carrega o servidor de verdade
precisa do bundle em `checkpoints/serving/` (Git LFS) e é pulado se ele não existir.

## Deploy

Não há workflow de CI neste repositório: os testes **não** rodam automaticamente em PR, então rode
`pytest` localmente antes de abrir uma. O deploy de QA (`deploy-qa.yml`) acontece no push na branch
`qa`; produção (`deploy-prod.yaml`), no push de uma tag `v*.*.*`. Os dois fazem checkout com Git
LFS, constroem a imagem (que copia `src/` e `checkpoints/serving/`, e instala só
`requirements.txt`) e aplicam os manifests de `k8s/` no GKE, com autenticação por Workload Identity
Federation. Por isso, **para publicar um modelo novo basta commitar o bundle** e fazer o deploy.
A rota no gateway fica em `infra-gtw-kong` (`manifests/{qa,prod}/ml-failure-predictor.yaml`):
`/predictor` para o MCP e `/predictor-rest` para o REST.

## Licença

[MIT](LICENSE)
