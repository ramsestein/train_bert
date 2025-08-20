#!/usr/bin/env python3
"""
Sistema NER Médico Optimizado - Solo modelos BERT
Sin PEFT, sin Llama - Versión estable
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from tqdm import tqdm
import gc
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Silenciar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Transformers - SIN PEFT
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== CONFIGURACIÓN GPU ==================
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# ================== ETIQUETAS Y PESOS ==================
NER_LABELS = [
    "O",
    "B-PATOLOGIA", "I-PATOLOGIA",
    "B-LOCALIZACION_ANATOMICA", "I-LOCALIZACION_ANATOMICA",
    "B-ARTERIA", "I-ARTERIA",
    "B-MEDIDA", "I-MEDIDA",
    "B-HALLAZGO_NEGATIVO", "I-HALLAZGO_NEGATIVO",
    "B-LATERALIDAD", "I-LATERALIDAD",
    "B-PUNTUACION_CLINICA", "I-PUNTUACION_CLINICA",
    "B-TEMPORALIDAD", "I-TEMPORALIDAD",
    "B-TECNICA_IMAGEN", "I-TECNICA_IMAGEN"
]

LABEL2ID = {label: idx for idx, label in enumerate(NER_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# Pesos para manejar desbalance
CLASS_WEIGHTS = {
    "O": 0.05,
    "B-HALLAZGO_NEGATIVO": 1.0,
    "I-HALLAZGO_NEGATIVO": 0.8,
    "B-PATOLOGIA": 5.0,
    "I-PATOLOGIA": 4.0,
    "B-LOCALIZACION_ANATOMICA": 8.0,
    "I-LOCALIZACION_ANATOMICA": 6.0,
    "B-ARTERIA": 20.0,
    "I-ARTERIA": 15.0,
    "B-MEDIDA": 15.0,
    "I-MEDIDA": 12.0,
    "B-LATERALIDAD": 15.0,
    "I-LATERALIDAD": 12.0,
    "B-PUNTUACION_CLINICA": 20.0,
    "I-PUNTUACION_CLINICA": 15.0,
    "B-TEMPORALIDAD": 12.0,
    "I-TEMPORALIDAD": 10.0,
    "B-TECNICA_IMAGEN": 10.0,
    "I-TECNICA_IMAGEN": 8.0
}

WEIGHT_TENSOR = torch.tensor([CLASS_WEIGHTS[label] for label in NER_LABELS])

# ================== FOCAL LOSS ==================
class FocalLoss(nn.Module):
    """Focal Loss para manejar desbalance extremo"""
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs.view(-1, inputs.size(-1)), 
            targets.view(-1), 
            weight=self.alpha.to(inputs.device) if self.alpha is not None else None,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()

# ================== PROCESADOR DE DATOS ==================

class RobustMedicalNERProcessor:
    """Procesador robusto que maneja todos los casos edge"""
    
    def __init__(self, tokenizer, max_length=512, label2id=None, id2label=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.id2label = id2label
        
    def load_and_analyze(self, file_path):
        """Carga y analiza datos con manejo robusto"""
        import json
        import numpy as np
        from collections import Counter
        
        data = []
        entity_counts = Counter()
        text_lengths = []
        errors = []
        
        logger.info(f"📂 Cargando datos de {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    if line.strip():
                        item = json.loads(line)
                        data.append(item)
                        
                        text_lengths.append(len(item['input']))
                        
                        # Parsear entidades de forma robusta
                        try:
                            if isinstance(item['output'], str):
                                output = json.loads(item['output'])
                                entities = output.get('entities', [])
                            else:
                                entities = item['output'].get('entities', [])
                        except:
                            entities = []
                        
                        for entity in entities:
                            if 'type' in entity:
                                entity_counts[entity['type']] += 1
                                
                except Exception as e:
                    errors.append((i, str(e)))
                    continue
        
        # Estadísticas
        logger.info(f"📊 Total ejemplos cargados: {len(data)}")
        logger.info(f"📊 Errores de carga: {len(errors)}")
        
        if text_lengths:
            logger.info(f"📊 Longitud promedio: {np.mean(text_lengths):.0f} chars")
            logger.info(f"📊 Longitud máxima: {max(text_lengths)} chars")
        
        if entity_counts:
            logger.info("🏷️ Distribución de entidades:")
            total_entities = sum(entity_counts.values())
            for entity_type, count in entity_counts.most_common():
                percentage = (count / total_entities * 100) if total_entities > 0 else 0
                logger.info(f"  • {entity_type}: {count} ({percentage:.1f}%)")
        
        return data, entity_counts
    
    def safe_parse_entities(self, item):
        """Parsea entidades de forma segura"""
        try:
            if isinstance(item['output'], str):
                output = json.loads(item['output'])
                return output.get('entities', [])
            else:
                return item['output'].get('entities', [])
        except:
            return []
    
    def validate_entity(self, entity, text_length):
        """Valida que una entidad sea procesable"""
        if not entity:
            return False
        
        # Verificar campos requeridos
        if 'type' not in entity or 'start' not in entity or 'end' not in entity:
            return False
        
        # Verificar índices
        start = entity['start']
        end = entity['end']
        
        if start < 0 or end <= start:
            return False
        
        if end > text_length:
            return False
        
        # Verificar tipo válido
        entity_type = entity['type']
        if f"B-{entity_type}" not in self.label2id:
            return False
        
        return True
    
    def process_dataset(self, data):
        """Procesa dataset con máxima robustez"""
        import json
        
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        processed = 0
        skipped = 0
        entity_errors = 0
        
        progress_bar = tqdm(data, desc="Procesando datos")
        
        for idx, item in enumerate(progress_bar):
            try:
                text = item.get('input', '')
                
                # Validar texto
                if not text or len(text.strip()) == 0:
                    skipped += 1
                    continue
                
                # Tokenizar texto
                try:
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                        return_offsets_mapping=True,
                        add_special_tokens=True,
                        return_tensors="pt"
                    )
                except Exception as e:
                    skipped += 1
                    continue
                
                # Inicializar etiquetas
                labels = torch.zeros(self.max_length, dtype=torch.long)
                labels.fill_(self.label2id['O'])
                
                # Obtener entidades
                entities = self.safe_parse_entities(item)
                
                # Procesar cada entidad
                offset_mapping = encoding.offset_mapping[0]
                
                for entity in entities:
                    try:
                        # Validar entidad
                        if not self.validate_entity(entity, len(text)):
                            entity_errors += 1
                            continue
                        
                        start_char = entity['start']
                        end_char = min(entity['end'], len(text))  # Asegurar que no exceda
                        entity_type = entity['type']
                        
                        # Encontrar tokens correspondientes
                        entity_tokens = []
                        
                        for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                            # Saltar tokens especiales
                            if token_start == 0 and token_end == 0:
                                continue
                            
                            # Verificar overlap con la entidad
                            # Usar lógica más permisiva para capturar todos los tokens relevantes
                            if token_start < end_char and token_end > start_char:
                                if token_idx < self.max_length:
                                    entity_tokens.append(token_idx)
                        
                        # Asignar etiquetas B- e I-
                        for i, token_idx in enumerate(entity_tokens):
                            if token_idx < self.max_length:
                                if i == 0:
                                    # Primera token: B-
                                    label_id = self.label2id.get(f'B-{entity_type}', self.label2id['O'])
                                else:
                                    # Tokens siguientes: I-
                                    label_id = self.label2id.get(f'I-{entity_type}', self.label2id['O'])
                                
                                labels[token_idx] = label_id
                    
                    except Exception as e:
                        entity_errors += 1
                        continue
                
                # Marcar tokens especiales y padding con -100
                try:
                    special_tokens_mask = encoding.special_tokens_mask[0]
                    labels[special_tokens_mask == 1] = -100
                    
                    attention_mask = encoding.attention_mask[0]
                    labels[attention_mask == 0] = -100
                except:
                    pass
                
                # Agregar al dataset
                all_input_ids.append(encoding.input_ids[0])
                all_attention_masks.append(encoding.attention_mask[0])
                all_labels.append(labels)
                
                processed += 1
                
                # Actualizar barra de progreso
                if processed % 100 == 0:
                    progress_bar.set_postfix({
                        'procesados': processed,
                        'omitidos': skipped,
                        'errores_entidad': entity_errors
                    })
                
            except Exception as e:
                skipped += 1
                continue
        
        progress_bar.close()
        
        # Resumen final
        logger.info(f"\n📊 Procesamiento completado:")
        logger.info(f"  ✅ Procesados exitosamente: {processed}/{len(data)}")
        logger.info(f"  ⚠️ Ejemplos omitidos: {skipped}")
        logger.info(f"  ⚠️ Entidades con errores: {entity_errors}")
        
        if processed == 0:
            raise ValueError("❌ No se pudo procesar ningún ejemplo! Revisa el formato de los datos.")
        
        # Crear dataset
        dataset = Dataset.from_dict({
            'input_ids': all_input_ids,
            'attention_mask': all_attention_masks,
            'labels': all_labels
        })
        
        logger.info(f"✅ Dataset creado con {len(dataset)} ejemplos")
        
        return dataset


# Función auxiliar para reemplazar el procesador en tu script
def crear_procesador_robusto(tokenizer, max_length=512):
    """Crea una instancia del procesador robusto"""
    
    LABEL2ID = {label: idx for idx, label in enumerate(NER_LABELS)}
    ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
    
    return RobustMedicalNERProcessor(
        tokenizer=tokenizer,
        max_length=max_length,
        label2id=LABEL2ID,
        id2label=ID2LABEL
    )

# ================== MÉTRICAS ==================
def compute_metrics(eval_preds):
    """Calcula métricas F1, precision y recall"""
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
    
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)
    
    # Convertir a etiquetas
    true_labels = []
    true_predictions = []
    
    for prediction, label in zip(predictions, labels):
        true_label = []
        true_pred = []
        
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:
                true_label.append(ID2LABEL[label_id])
                true_pred.append(ID2LABEL[pred_id])
        
        if true_label:
            true_labels.append(true_label)
            true_predictions.append(true_pred)
    
    # Métricas generales
    results = {
        "f1": f1_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
    }
    
    # F1 sin contar 'O'
    true_labels_no_o = []
    true_preds_no_o = []
    
    for labels, preds in zip(true_labels, true_predictions):
        label_filtered = [l for l in labels if l != 'O']
        pred_filtered = [p for p, l in zip(preds, labels) if l != 'O']
        if label_filtered:
            true_labels_no_o.append(label_filtered)
            true_preds_no_o.append(pred_filtered)
    
    if true_labels_no_o:
        results["f1_no_O"] = f1_score(true_labels_no_o, true_preds_no_o)
        results["precision_no_O"] = precision_score(true_labels_no_o, true_preds_no_o)
        results["recall_no_O"] = recall_score(true_labels_no_o, true_preds_no_o)
    
    print("\n" + "="*60)
    print("📊 REPORTE POR ENTIDAD:")
    print("="*60)
    print(classification_report(true_labels, true_predictions))
    
    return results

# ================== CUSTOM TRAINER ==================
class CustomNERTrainer(Trainer):
    """Trainer con Focal Loss"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=WEIGHT_TENSOR, gamma=2.0)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ================== CONFIGURACIÓN DE MODELOS ==================
BERT_MODELS = [
    {
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "model_type": "pubmedbert-optimized",
        "description": "BERT biomédico PubMed",
        "batch_size": 64,  # Reducido para estabilidad
        "learning_rate": 3e-5,
        "num_epochs": 8,
    },
    {
        "model_name": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        "model_type": "roberta-clinical-es",
        "description": "RoBERTa clínico español",
        "batch_size": 64,
        "learning_rate": 2e-5,
        "num_epochs": 8,
    },
    {
        "model_name": "PlanTL-GOB-ES/roberta-base-biomedical-es",
        "model_type": "roberta-biomedical-es",
        "description": "RoBERTa biomédico español",
        "batch_size": 64,
        "learning_rate": 3e-5,
        "num_epochs": 8,
    },
    {
        "model_name": "dmis-lab/biobert-base-cased-v1.2",
        "model_type": "biobert-dmis",
        "description": "BioBERT radiología",
        "batch_size": 64,
        "learning_rate": 2e-5,
        "num_epochs": 8,
    },
    {
        "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "model_type": "sapbert",
        "description": "SapBERT multilingüe",
        "batch_size": 48,
        "learning_rate": 2e-5,
        "num_epochs": 8,
    }
]

# ================== CONFIGURACIÓN ==================
@dataclass
class NERConfig:
    """Configuración para entrenamiento NER"""
    model_name: str
    model_type: str
    output_dir: str = "/content/drive/MyDrive/ner_bert_optimized"
    batch_size: int = 64
    learning_rate: float = 2e-5
    num_epochs: int = 8
    max_length: int = 512
    warmup_ratio: float = 0.2
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 50
    seed: int = 42

# ================== ENTRENADOR PRINCIPAL ==================
class NERModelTrainer:
    """Clase principal de entrenamiento"""
    
    def __init__(self, config: NERConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            torch.cuda.empty_cache()
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, train_file, val_split=0.05):
        logger.info("📚 Preparando datos...")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # USAR EL PROCESADOR ROBUSTO
        processor = RobustMedicalNERProcessor(
            self.tokenizer, 
            self.config.max_length,
            label2id=LABEL2ID,
            id2label=ID2LABEL
        )
        
        # Cargar datos
        all_data, entity_counts = processor.load_and_analyze(train_file)
        
        # Dividir
        train_data, val_data = train_test_split(
            all_data, 
            test_size=val_split, 
            random_state=self.config.seed
        )
        
        logger.info(f"📊 Train: {len(train_data)}, Val: {len(val_data)}")
        
        # Procesar con el método robusto
        self.train_dataset = processor.process_dataset(train_data)
        self.val_dataset = processor.process_dataset(val_data)
    
    def load_model(self):
        """Carga modelo"""
        logger.info(f"🤖 Cargando: {self.config.model_name}")
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(NER_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
    
    def train(self):
        """Entrena modelo"""
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            self.tokenizer,
            padding=True,
            max_length=self.config.max_length
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=2,
            metric_for_best_model="f1_no_O",
            greater_is_better=True,
            load_best_model_at_end=True,
            fp16=self.config.fp16,
            dataloader_num_workers=2,
            logging_steps=self.config.logging_steps,
            report_to=["tensorboard"],
            seed=self.config.seed,
            remove_unused_columns=False,
        )
        
        # Callbacks
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
        ]
        
        # Trainer
        trainer = CustomNERTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        
        # Entrenar
        logger.info("🚀 Iniciando entrenamiento...")
        train_result = trainer.train()
        
        # Guardar
        logger.info("💾 Guardando modelo...")
        trainer.save_model()
        
        # Evaluar
        eval_results = trainer.evaluate()
        
        # Métricas
        metrics = {
            **train_result.metrics,
            **eval_results,
            'model': self.config.model_name
        }
        
        # Guardar métricas
        with open(Path(self.config.output_dir) / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

# ================== FUNCIÓN PRINCIPAL ==================
def main():
    """Función principal"""
    
    BASE_DIR = "/content/drive/MyDrive/ner_bert_models"
    TRAIN_FILE = "/content/training_synthetic_ictus.jsonl"
    
    if not os.path.exists(TRAIN_FILE):
        logger.error(f"❌ No se encontró {TRAIN_FILE}")
        return
    
    results = []
    
    logger.info("="*60)
    logger.info("🚀 ENTRENANDO MODELOS BERT")
    logger.info("="*60)
    
    for i, model_cfg in enumerate(BERT_MODELS, 1):
        logger.info(f"\n📊 Modelo {i}/{len(BERT_MODELS)}: {model_cfg['model_type']}")
        logger.info(f"   {model_cfg['description']}")
        
        try:
            # Configuración
            config = NERConfig(
                model_name=model_cfg['model_name'],
                model_type=model_cfg['model_type'],
                output_dir=os.path.join(BASE_DIR, model_cfg['model_type']),
                batch_size=model_cfg['batch_size'],
                learning_rate=model_cfg['learning_rate'],
                num_epochs=model_cfg['num_epochs'],
            )
            
            # Entrenar
            trainer = NERModelTrainer(config)
            trainer.prepare_data(TRAIN_FILE)
            trainer.load_model()
            metrics = trainer.train()
            
            results.append({
                'model': model_cfg['model_type'],
                'f1': metrics.get('eval_f1', 0),
                'f1_no_O': metrics.get('eval_f1_no_O', 0),
                'loss': metrics.get('eval_loss', 0),
            })
            
            # Limpiar
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Resumen
    if results:
        logger.info("\n" + "="*60)
        logger.info("📊 RESUMEN FINAL")
        logger.info("="*60)
        
        df = pd.DataFrame(results)
        df = df.sort_values('f1_no_O', ascending=False)
        print(df.to_string(index=False))
        
        df.to_csv(os.path.join(BASE_DIR, "resultados.csv"), index=False)
        
        if len(df) > 0:
            best = df.iloc[0]
            logger.info(f"\n🏆 MEJOR MODELO: {best['model']}")
            logger.info(f"   F1 (sin O): {best['f1_no_O']:.4f}")

if __name__ == "__main__":
    main()