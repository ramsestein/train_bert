# Medical NER Training System

## Overview
Sistema NER médico optimizado para entrenar modelos BERT en reconocimiento de entidades nombradas en textos clínicos de ictus.

## Training Script
- **Main file**: `train/train_NER.py`
- **Type**: Medical BERT-based NER training system
- **Purpose**: Train models to recognize medical entities in stroke/ictus clinical texts

## Dependencies
```bash
pip install torch transformers datasets seqeval scikit-learn pandas tqdm numpy
```

## Supported Models
1. **PubMedBERT** - `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
2. **RoBERTa Clinical ES** - `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`
3. **RoBERTa Biomedical ES** - `PlanTL-GOB-ES/roberta-base-biomedical-es`
4. **BioBERT** - `dmis-lab/biobert-base-cased-v1.2`
5. **SapBERT** - `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`

## Entity Labels
- `PATOLOGIA` - Medical pathologies
- `LOCALIZACION_ANATOMICA` - Anatomical locations
- `ARTERIA` - Arteries
- `MEDIDA` - Measurements
- `HALLAZGO_NEGATIVO` - Negative findings
- `LATERALIDAD` - Laterality
- `PUNTUACION_CLINICA` - Clinical scores
- `TEMPORALIDAD` - Temporal aspects
- `TECNICA_IMAGEN` - Imaging techniques

## Configuration
- **Max sequence length**: 512 tokens
- **Batch sizes**: 48-64 (model dependent)
- **Learning rates**: 2e-5 to 3e-5
- **Epochs**: 8
- **Loss function**: Focal Loss with class weights

## Usage
```python
# Run training for all models
python train/train_NER.py
```

## Data Format
Input: JSONL file with structure:
```json
{"input": "text", "output": {"entities": [{"type": "PATOLOGIA", "start": 0, "end": 10}]}}
```

## Output
- Trained models saved to `/content/drive/MyDrive/ner_bert_models/`
- Metrics saved as JSON
- Results summary CSV

## Features
- Robust data processing with error handling
- Class imbalanced handling via Focal Loss
- Early stopping and best model selection
- Comprehensive evaluation metrics
- Multi-GPU support

## Commands
- **Train models**: `python train/train_NER.py`
- **View logs**: Check tensorboard outputs in model directories