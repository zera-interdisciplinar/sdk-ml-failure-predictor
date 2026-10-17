# sdk-ml-failure-predictor

Regressão em PyTorch: em quantos meses um equipamento quebra, contados da data de aquisição (`acquiredAt`), não de hoje. O mesmo processo serve as tools MCP e o `POST /predict-batch` na porta 8000.

## O que entra no modelo

`category`, `manufacturer`, `model`, `climateZone`, `usageIntensity` e a idade na aquisição. A idade é o número de meses de 1º de janeiro do ano de fabricação até `acquiredAt` (`dias / 30.44`). Identificadores, `status`, `unitId` e `nextPredictionDate` ficam de fora.

A saída da rede é linear. O valor devolvido é `max(0, saída)`.

## Bundle

O servidor não lê o `config.yaml`. Ele lê `checkpoints/serving/` (ou o diretório em `MODEL_DIR`):

- `ttf_model.pt` — pesos do epoch com menor MSE de validação
- `feature_encoder.json` — vocabulário e normalização das numéricas
- `manifest.json` — dims, tamanhos de vocabulário, seed, epoch, MAE/RMSE e o sha256 do peso

Se o encoder ou o hash não baterem com o manifest, o processo não sobe. Gráficos e `checkpoints/data.txt` não vão para a imagem.

## Comandos

```bash
pip install -r requirements-train.txt
python scripts/synthetic_generator.py --n 100000 --seed 2026
python -m src.training.train
pytest
python -m src.serving.mcp_server
```

O bundle em `checkpoints/serving/` é o treino de 2026-10-17: 100000 linhas, seed 2026, melhor epoch 118 (MSE de validação 13,66). Vocabulário, com `<UNK>`: 9 categorias, 41 fabricantes, 240 modelos, 5 zonas climáticas.

| Corte | MAE | RMSE | ±3 meses | MAPE |
| --- | --- | --- | --- | --- |
| Treino | 2,59 meses | 3,51 | 67,6% | 6,5% |
| Validação (modelo já visto) | 2,74 meses | 3,70 | 65,2% | 6,8% |
| Modelo novo | 3,52 meses | 4,70 | 54,7% | 8,8% |

A imagem de serving instala só `requirements.txt`. Ela ainda precisa de rebuild para levar este bundle.

## Contrato

- MCP, streamable HTTP, em `/mcp`: `predict_time_to_failure`, `predict_time_to_failure_batch`, `list_valid_categories`, `list_valid_climate_zones`.
- `POST /predict-batch` com `{"devices": [ ... ]}` (1 a 500). A resposta é `{"results": [{"estimated_remaining_months", "error"}]}`, na mesma ordem. JSON inválido responde 422.
- Categoria, fabricante ou modelo fora do vocabulário viram `<UNK>`. O treino mascara 5% dos índices categóricos para essa linha do embedding receber gradiente. 20% dos nomes de modelo ficam de fora do treino de propósito; o erro nesse corte vai no manifest como `unseen_model_mae`.
