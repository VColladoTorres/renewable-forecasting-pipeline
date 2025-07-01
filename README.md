# Renewable-Forecasting-Pipeline

Workflow reproductible para previsión horaria de **eólica on-shore** y **PV utility-scale** en España (2018-2024).

## Uso rápido
```bash
cp .env.example .env         # añade tus tokens
pip install -e .
python -m src.data.make_dataset     --config configs/default.yaml
python -m src.features.build_features --config configs/default.yaml
python -m src.models.train_xgb       --config configs/default.yaml
python -m src.models.evaluate        --config configs/default.yaml
```
