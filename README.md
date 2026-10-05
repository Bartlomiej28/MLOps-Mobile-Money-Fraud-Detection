# MLOps Production Ready Machine Learning Project — Mobile Money Fraud Detection

End to end MLOps project: data ingestion, validation with drift detection, feature
engineering, model training with automated hyper parameter search, model evaluation
against the production model, model publishing, a FastAPI test interface, Docker
packaging and a GitHub Actions pipeline that deploys to AWS ECR + EC2.

Dataset: [`CiferAI/Cifer-Fraud-Detection-Dataset-AF`](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF)
— 21M mobile money transactions across 14 csv shards.

---

## ⚠️ A note on the dataset

The pipeline is complete and correct, but the **metrics it reports are at chance level**,
and this is a property of the data, not of the code.

`Cifer-Fraud-Detection-Dataset-AF` is an SDV re-synthesis of PaySim in which the columns
were generated close to independently, so the `isFraud` label is statistically independent
of the features:

| check | result |
| --- | --- |
| best univariate ROC AUC over 27 raw + engineered features | 0.4825 (`hour_of_day`), everything else within ±0.01 of 0.5 |
| 5-fold CV ROC AUC of LogReg, HistGB, LightGBM, XGBoost, CatBoost, RF, ExtraTrees | 0.49 – 0.51 for every one of them |
| PR AUC vs. the positive class share | 0.047 – 0.051 vs. 0.048 |
| best reachable F1 at any threshold | 0.091 = the F1 of labelling **every** row as fraud |
| fraud rate per transaction type | identical across all 5 types |
| fraud rate in the original PaySim | non-zero **only** for `TRANSFER` and `CASH_OUT` |
| anonymised ids (`nameOrig`, `nameDest`) | random `sdv-pii-*` strings, no `C`/`M` prefix |
| `isFlaggedFraud` | raised on 1 row out of 82 803 |

Notebook `2_feature_engineering_and_model_training.ipynb` reproduces the whole comparison
and ends with an explicit signal check.

### What the pipeline does about it

The modelling stack is tuned to squeeze out whatever is there and, more importantly, to
report it honestly:

| technique | why |
| --- | --- |
| **class weights** (`class_weight=balanced`) instead of SMOTEENN | SMOTEENN inflated the cross validated F1 to 0.99 on synthetic points while the test F1 stayed at 0.045; weighting keeps CV scores comparable with the test set and cuts training time by ~80× |
| **model selection by ROC AUC** (`config/model.yaml → scoring: roc_auc`) | threshold free, insensitive to the class prior, so the grid search cannot be fooled by resampling |
| **decision threshold tuned on out-of-fold probabilities** | the threshold that maximises F1 is chosen on the training folds and persisted inside `FraudDetectionModel`; the test split is never touched |
| **acceptance gate on ROC AUC** (`MODEL_TRAINER_EXPECTED_SCORE = 0.5`) | a model that ranks worse than random is refused, whatever its accuracy |
| **production comparison by ROC AUC** | the pusher replaces the live model only when the new one ranks strictly better on the held out split |

Last run on the working dataset (`LogisticRegression(C=0.01, class_weight=balanced)`, threshold 0.492):

| metric (untouched test split) | before | now |
| --- | --- | --- |
| F1 | 0.045 | **0.086** |
| recall | 0.044 | **0.582** |
| precision | 0.046 | 0.046 |
| ROC AUC | 0.490 | 0.496 |
| PR AUC | – | 0.047 |

F1 doubled only because the threshold now sits where the maximum of a signal free problem
is; ROC AUC did not move because it cannot. Everything downstream — schema, feature
engineering, weighting, training, registry, serving — is written against the real PaySim
column contract. Pointing `FraudData` at the original PaySim csv produces a strong model
**without a single code change**, and every technique above carries over.

---

## Project structure

```
├── app.py                       FastAPI app: html form, /predict json api, /train, /health
├── demo.py                      runs the whole training pipeline
├── download_dataset.py          builds the local working dataset from Hugging Face
├── template.py                  scaffolds the project skeleton
├── Dockerfile / .dockerignore
├── config/
│   ├── schema.yaml              raw column contract + transformer column groups
│   └── model.yaml               class weighted scikit-learn model zoo + ROC AUC grid search for neuro_mf
├── docs/architecture.md          mermaid diagrams of every pipeline stage
├── notebook/
│   ├── 1_EDA_fraud_detection.ipynb
│   ├── 2_feature_engineering_and_model_training.ipynb
│   └── 3_data_drift_evidently.ipynb
├── templates/fraud_detection.html
├── static/css/style.css
└── fraud_detection/
    ├── cloud_storage/aws_storage.py       S3 helper
    ├── components/                        the six pipeline stages
    ├── configuration/aws_connection.py    boto3 session
    ├── constants/                         every tunable constant
    ├── data_access/fraud_data.py          Hugging Face source (replaces MongoDB)
    ├── entity/
    │   ├── config_entity.py               stage configuration dataclasses
    │   ├── artifact_entity.py             stage output dataclasses
    │   ├── estimator.py                   FraudFeatureEngineer + FraudDetectionModel (persisted threshold)
    │   └── model_registry.py              S3 registry + local registry + resolver
    ├── exception/  logger/  utils/
    └── pipeline/
        ├── training_pipeline.py
        ├── prediction_pipeline.py
        └── seed_model_registry.py         trains + publishes when the registry is empty
```

---

## Data flow

`data.py` in the original repository only called `load_dataset(...)`. That is now a real
ingestion layer in `fraud_detection/data_access/fraud_data.py`:

1. Download `DATA_INGESTION_N_PARTS` csv shards with `huggingface_hub` (~138 MB each, 1.5M rows each).
2. Keep **every** fraud row, randomly down-sample the legitimate rows to
   `DATA_INGESTION_NON_FRAUD_PER_FRAUD` per fraud row (default 20).
   The upstream fraud rate is 0.13 %, so this is what makes 21M rows tractable on one machine.
3. Cache the result at `data/fraud_transactions.csv` and reuse it on the next run.

Defaults produce ~83 000 rows with ~3 900 fraud cases. Tune in `fraud_detection/constants/__init__.py`
or via `python download_dataset.py --parts 4 --non-fraud-per-fraud 10`.

---

## Features

Raw columns from the source: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`,
`newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.

`FraudFeatureEngineer` is the **first step of the persisted preprocessing pipeline**, so the
identical derivation runs during training, evaluation and online prediction:

| derived feature | definition |
| --- | --- |
| `hour_of_day`, `day_of_month` | `step % 24`, `step // 24` |
| `balance_change_orig` / `_dest` | balance moved on each side |
| `error_balance_orig` / `_dest` | residual of the balance identity — the classic PaySim fraud signal |
| `amount_to_balance_ratio` | `amount / (oldbalanceOrg + 1)` |
| `orig_balance_emptied` | sender account drained to zero (`Y`/`N`) |
| `dest_previously_empty` | receiver started at zero (`Y`/`N`) |
| `amount_band` | ordinal band: `low` / `medium` / `high` / `very_high` |

Dropped (`config/schema.yaml → drop_columns`): `step` (replaced by the two temporal
features), `nameOrig` / `nameDest` (random ids), `isFlaggedFraud` (platform rule, leakage).

Transformers: `OneHotEncoder` on `type`, `OrdinalEncoder` with explicit category order on the
three ordinal columns, Yeo-Johnson `PowerTransformer` on the ten skewed monetary columns,
`StandardScaler` on the two temporal columns.

---

## Quick start

```bash
pip install -r requirements_dev.txt      # requirements.txt alone is enough to serve
python download_dataset.py               # optional, the pipeline does it on demand
python demo.py                           # full training pipeline
python app.py                            # http://localhost:8080
```

Train from the browser instead: open `http://localhost:8080/train`.

### Prediction API

```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{
  "step": 200, "type": "TRANSFER", "amount": 181000,
  "oldbalanceOrg": 181000, "newbalanceOrig": 0,
  "oldbalanceDest": 21182, "newbalanceDest": 202182
}'
```

```json
{"prediction": 0, "label": "Legitimate", "fraud_probability": 0.18}
```

---

## Model registry

MongoDB is gone. The model store is resolved at runtime by
`fraud_detection/entity/model_registry.py`:

| `MODEL_STORAGE` | backend |
| --- | --- |
| `local` | `saved_models/model.pkl` |
| `s3` | `s3://$MODEL_BUCKET_NAME/model.pkl` |
| unset | S3 when AWS credentials are present, otherwise local |

That is why training, evaluation and prediction all work out of the box with no cloud
account, while the AWS deployment path stays fully functional.

---

## Docker

```bash
docker build -t fraud-detection .

docker run -d -p 8080:8080 --name fraud-detection \
  -v "$(pwd)/saved_models:/app/saved_models" \
  -v "$(pwd)/data:/app/data" \
  fraud-detection
```

With S3 as the registry:

```bash
docker run -d -p 8080:8080 --name fraud-detection \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e MODEL_STORAGE=s3 \
  -e MODEL_BUCKET_NAME=your-bucket \
  fraud-detection
```

The image is a two stage build: the compiler toolchain only lives in the builder stage and
the runtime stage ships the virtual environment plus `libgomp1` and `curl`, which keeps the
final image at ~1.5 GB instead of 2 GB. `requirements.txt` is fully pinned to the versions the model was
trained with, so the pickle and the serving runtime always match. The image exposes `8080`
and carries a `HEALTHCHECK` against `/health`.

Seed the registry from inside a running container (skips when a model already exists):

```bash
docker exec fraud-detection python -m fraud_detection.pipeline.seed_model_registry
```

---

## AWS deployment

`.github/workflows/aws.yaml` runs on every push to `main`:

1. **Continuous-Integration** (`ubuntu-latest`) — install dependencies, import the package and
   validate `config/schema.yaml` + `config/model.yaml`, then build and push the image to ECR
   tagged with both the commit sha and `latest`.
2. **Continuous-Deployment** (`self-hosted` runner on the EC2 box) — pull `latest`, remove the
   previous container, start the new one with `--restart unless-stopped`, wait for `/health`,
   kick off the S3 registry seeding in the background if the bucket is empty, prune old images.

### One time setup

1. IAM user with `AmazonEC2ContainerRegistryFullAccess` **and `AmazonS3FullAccess`** (or a
   policy scoped to the registry bucket). The container reads and writes the production
   model in S3, so ECR access alone is not enough. `AmazonEC2FullAccess` is not required by
   the workflow.
2. Create the ECR repository and note its URI.
3. Create the S3 bucket used as the model registry. It may stay empty: the deployment job
   starts `python -m fraud_detection.pipeline.seed_model_registry` **detached and niced**
   inside the container, which trains and publishes a model when the bucket holds none
   (10–25 min on 2 vCPU, follow it with
   `docker exec fraud-detection tail -f /app/logs/seed_model_registry.log`). It runs in the
   background on purpose: a synchronous grid search starves the GitHub runner of CPU and the
   job dies with *"The self-hosted runner lost communication with the server"*. To skip that
   first training entirely, upload a model beforehand:

   ```bash
   aws s3 cp saved_models/model.pkl s3://<MODEL_BUCKET_NAME>/model.pkl
   ```
4. Launch an EC2 instance (Ubuntu, **`t3.medium` or larger** — the seed training downloads
   ~280 MB from Hugging Face and needs ~3 GB RAM, a `t2.micro` runs out of memory), install
   Docker, add the runner user to the `docker` group, register it as a self-hosted runner
   (*Settings → Actions → Runners*), open port `8080` in the security group.
5. Repository secrets:

| secret | example |
| --- | --- |
| `AWS_ACCESS_KEY_ID` | |
| `AWS_SECRET_ACCESS_KEY` | |
| `AWS_DEFAULT_REGION` | `us-east-1` |
| `ECR_REPO` | `fraud-detection` |
| `MODEL_BUCKET_NAME` | `fraud-detection-model-registry` |

---

## Pipeline stages

| stage | input | output |
| --- | --- | --- |
| Data Ingestion | Hugging Face shards | `feature_store/`, stratified `train.csv` / `test.csv` |
| Data Validation | train + test csv | schema check, Evidently drift report (`.yaml` + `.html`) |
| Data Transformation | train + test csv | fitted preprocessing pipeline, `train.npy` / `test.npy` |
| Model Trainer | transformed arrays | best model via `neuro_mf` grid search (ROC AUC), out-of-fold threshold, test metrics |
| Model Evaluation | test csv + trained model | accept / reject against the production model by ROC AUC |
| Model Pusher | accepted model | published to the resolved registry |

Everything is written under `artifact/<timestamp>/`.

**The training split is never resampled.** The class imbalance is handled by the class
weights of the estimators declared in `config/model.yaml`, and the test split stays untouched
and imbalanced, so the reported precision, recall, F1, ROC AUC and PR AUC describe the real
distribution.
