import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime
import time
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    pipeline
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN GLOBAL
# ============================================

# Configuración de evaluación
EVALUATION_CONFIG = {
    'num_documents': None,  # None = procesar TODOS los documentos
    'batch_size': 10,      # Tamaño de lote para procesamiento
    'use_gpu': True,       # Usar GPU si está disponible
    'semantic_threshold': 0.85,  # Umbral general para sinónimos
    'preferred_model_subdir': None,  # Subdirectorio específico a usar (None para automático)
}

# Mapeo de modelos a sus archivos de test truncados específicos
MODEL_TEST_FILES = {
    "biobert-radiology": "test_real_ictus_512_biobert.jsonl",
    "biomedical-roberta": "test_real_ictus_512_biomedical_roberta.jsonl",
    "pubmedbert-ms": "test_real_ictus_512_pubmedbert.jsonl",
    "roberta-clinical-es": "test_real_ictus_512_roberta_clinical.jsonl",
    "sapbert-pubmed": "test_real_ictus_512_sapbert.jsonl",
    "llama3.2-3b-int4": "test_real_ictus_512_llama.jsonl",
    "llama3.2-3b-int8": "test_real_ictus_512_llama.jsonl",
    # Modelos con prefijo 'real-'
    "real-biobert-radiology": "test_syn_ictus_512_biobert.jsonl",
    "real-biomedical-roberta": "test_syn_ictus_512_biomedical_roberta.jsonl",
    "real-pubmedbert-ms": "test_syn_ictus_512_pubmedbert.jsonl",
    "real-roberta-clinical-es": "test_syn_ictus_512_roberta_clinical.jsonl",
    "real-sapbert-pubmed-real": "test_syn_ictus_512_sapbert.jsonl",
}

# Umbrales semánticos por tipo de entidad
SEMANTIC_THRESHOLDS = {
    'PATOLOGIA': 0.85,
    'MEDIDA': 0.95,
    'LOCALIZACION_ANATOMICA': 0.80,
    'HALLAZGO_NEGATIVO': 0.90,
    'ARTERIA': 0.85
}

# Umbrales de confianza para predicciones
CONFIDENCE_THRESHOLDS = {
    'global': 0.3,  # <-- AJUSTA ESTE VALOR (prueba 0.3, 0.4, 0.5, 0.6)
    'B-PATOLOGIA': 0.4,
    'I-PATOLOGIA': 0.4,
    'B-LOCALIZACION_ANATOMICA': 0.4,
    'I-LOCALIZACION_ANATOMICA': 0.4,
    'B-ARTERIA': 0.4,
    'I-ARTERIA': 0.4,
    'B-MEDIDA': 0.3,  # Más bajo para medidas que son importantes
    'I-MEDIDA': 0.3,
    'B-HALLAZGO_NEGATIVO': 0.5,
    'I-HALLAZGO_NEGATIVO': 0.5,
}


class NEREntity:
    """Representa una entidad NER con su información."""
    def __init__(self, text: str, entity_type: str, start: int, end: int):
        self.text = text
        self.type = entity_type
        self.start = start
        self.end = end
    
    def __repr__(self):
        return f"Entity('{self.text}', {self.type}, [{self.start}:{self.end}])"
    
    def overlaps(self, other: 'NEREntity', min_overlap: float = 0.5) -> bool:
        """Verifica si hay overlap suficiente con otra entidad."""
        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)
        
        if overlap_start >= overlap_end:
            return False
        
        overlap_len = overlap_end - overlap_start
        self_len = self.end - self.start
        other_len = other.end - other.start
        
        return (overlap_len / self_len >= min_overlap or 
                overlap_len / other_len >= min_overlap)


class SemanticMatcher:
    """Maneja la comparación semántica usando LABSE."""
    
    def __init__(self, threshold: float = None):
        print("Cargando modelo LABSE para similitud semántica...")
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        self.threshold = threshold or EVALUATION_CONFIG['semantic_threshold']
        self.cache = {}  # Cache para embeddings
        self.type_thresholds = SEMANTIC_THRESHOLDS
    
    def are_synonyms(self, text1: str, text2: str, entity_type: str = None) -> Tuple[bool, float]:
        """Determina si dos textos son sinónimos según LABSE."""
        if text1.lower() == text2.lower():
            return True, 1.0
        
        # Obtener embeddings (con cache)
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        # Calcular similitud
        similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        
        # Usar umbral específico del tipo si está disponible
        threshold = self.type_thresholds.get(entity_type, self.threshold)
        
        return similarity >= threshold, float(similarity)
    
    def _get_embedding(self, text: str):
        """Obtiene embedding con cache."""
        if text not in self.cache:
            self.cache[text] = self.model.encode(text)
        return self.cache[text]


class NERMetricsCalculator:
    """Calculador de métricas para evaluación NER."""
    
    def __init__(self, use_semantic: bool = True):
        self.use_semantic = use_semantic
        if use_semantic:
            self.semantic_matcher = SemanticMatcher()
        self.entity_types = set()
    
    def calculate_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Calcula todas las métricas comparando predicciones con ground truth."""
        
        metrics = {
            'summary': {
                'documents_evaluated': len(predictions),
                'total_entities_gold': 0,
                'total_entities_predicted': 0
            },
            'strict': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
            'semantic': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
            'partial': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
            'semantic_partial': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
            'type_confusion': defaultdict(int),
            'error_analysis': defaultdict(int),
            'semantic_matches': []
        }
        
        # Procesar cada documento
        for pred_doc, gold_doc in zip(predictions, ground_truth):
            assert pred_doc['document_id'] == gold_doc['document_id'], \
                   f"Document ID mismatch: {pred_doc['document_id']} != {gold_doc['document_id']}"
            
            pred_entities = [NEREntity(
                text=e['text'],
                entity_type=e['type'],
                start=e['start'],
                end=e['end']
            ) for e in pred_doc['entities']]
            
            gold_entities = [NEREntity(
                text=e['text'],
                entity_type=e['type'],
                start=e['start'],
                end=e['end']
            ) for e in gold_doc['entities']]
            
            metrics['summary']['total_entities_gold'] += len(gold_entities)
            metrics['summary']['total_entities_predicted'] += len(pred_entities)
            
            # Actualizar tipos de entidad conocidos
            for e in gold_entities + pred_entities:
                self.entity_types.add(e.type)
            
            # Evaluar predicciones
            self._evaluate_document(pred_entities, gold_entities, metrics)
        
        # Calcular precision, recall, F1
        final_metrics = self._compute_final_metrics(metrics)
        
        return final_metrics
    
    def _evaluate_document(self, pred_entities: List[NEREntity], 
                          gold_entities: List[NEREntity], metrics: Dict):
        """Evalúa las entidades de un documento."""
        
        # Tracking de entidades ya emparejadas
        matched_gold_strict = set()
        matched_gold_semantic = set()
        matched_gold_partial = set()
        matched_gold_semantic_partial = set()
        
        # Evaluar cada predicción
        for pred in pred_entities:
            strict_match = False
            semantic_match = False
            partial_match = False
            semantic_partial_match = False
            
            for i, gold in enumerate(gold_entities):
                # Match estricto
                if (pred.text == gold.text and 
                    pred.type == gold.type and 
                    pred.start == gold.start and 
                    pred.end == gold.end):
                    if i not in matched_gold_strict:
                        metrics['strict']['all']['tp'] += 1
                        metrics['strict'][pred.type]['tp'] += 1
                        matched_gold_strict.add(i)
                        strict_match = True
                        break
                
                # Match semántico (mismo tipo y posición, sinónimo)
                if (self.use_semantic and 
                    pred.type == gold.type and 
                    pred.start == gold.start and 
                    pred.end == gold.end):
                    is_synonym, similarity = self.semantic_matcher.are_synonyms(
                        pred.text, gold.text, pred.type
                    )
                    if is_synonym and i not in matched_gold_semantic:
                        metrics['semantic']['all']['tp'] += 1
                        metrics['semantic'][pred.type]['tp'] += 1
                        matched_gold_semantic.add(i)
                        semantic_match = True
                        metrics['semantic_matches'].append({
                            'pred': pred.text,
                            'gold': gold.text,
                            'type': pred.type,
                            'similarity': similarity
                        })
                        break
                
                # Match parcial (overlap de posiciones)
                if (pred.type == gold.type and pred.overlaps(gold)):
                    if i not in matched_gold_partial:
                        metrics['partial']['all']['tp'] += 1
                        metrics['partial'][pred.type]['tp'] += 1
                        matched_gold_partial.add(i)
                        partial_match = True
                        if not strict_match:  # Solo contar si no es match estricto
                            metrics['error_analysis']['wrong_span'] += 1
                        break
                
                # Match semántico parcial
                if (self.use_semantic and 
                    pred.type == gold.type and 
                    pred.overlaps(gold)):
                    is_synonym, _ = self.semantic_matcher.are_synonyms(
                        pred.text, gold.text, pred.type
                    )
                    if is_synonym and i not in matched_gold_semantic_partial:
                        metrics['semantic_partial']['all']['tp'] += 1
                        metrics['semantic_partial'][pred.type]['tp'] += 1
                        matched_gold_semantic_partial.add(i)
                        semantic_partial_match = True
                        break
                
                # Análisis de confusión de tipos
                if (pred.text == gold.text and 
                    pred.start == gold.start and 
                    pred.end == gold.end and 
                    pred.type != gold.type):
                    metrics['type_confusion'][f"{gold.type}->{pred.type}"] += 1
                    metrics['error_analysis']['wrong_type'] += 1
            
            # Si no hubo ningún match, es un falso positivo
            if not any([strict_match, semantic_match, partial_match, semantic_partial_match]):
                metrics['strict']['all']['fp'] += 1
                metrics['strict'][pred.type]['fp'] += 1
                metrics['semantic']['all']['fp'] += 1
                metrics['semantic'][pred.type]['fp'] += 1
                metrics['partial']['all']['fp'] += 1
                metrics['partial'][pred.type]['fp'] += 1
                metrics['semantic_partial']['all']['fp'] += 1
                metrics['semantic_partial'][pred.type]['fp'] += 1
        
        # Contar falsos negativos
        for i, gold in enumerate(gold_entities):
            if i not in matched_gold_strict:
                metrics['strict']['all']['fn'] += 1
                metrics['strict'][gold.type]['fn'] += 1
                metrics['error_analysis']['miss_total'] += 1
            if i not in matched_gold_semantic:
                metrics['semantic']['all']['fn'] += 1
                metrics['semantic'][gold.type]['fn'] += 1
            if i not in matched_gold_partial:
                metrics['partial']['all']['fn'] += 1
                metrics['partial'][gold.type]['fn'] += 1
            if i not in matched_gold_semantic_partial:
                metrics['semantic_partial']['all']['fn'] += 1
                metrics['semantic_partial'][gold.type]['fn'] += 1
    
    def _compute_final_metrics(self, metrics: Dict) -> Dict:
        """Calcula precision, recall y F1 finales."""
        
        def calc_prf(counts):
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            return {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'support': tp + fn
            }
        
        # Calcular métricas para cada nivel
        final = {
            'summary': metrics['summary'],
            'metrics': {},
            'per_type_metrics': {},
            'error_analysis': dict(metrics['error_analysis']),
            'type_confusion_matrix': dict(metrics['type_confusion'])
        }
        
        # Métricas globales
        for level in ['strict', 'semantic', 'partial', 'semantic_partial']:
            final['metrics'][level] = calc_prf(metrics[level]['all'])
        
        # Métricas por tipo
        for entity_type in sorted(self.entity_types):
            final['per_type_metrics'][entity_type] = {}
            for level in ['strict', 'semantic']:
                final['per_type_metrics'][entity_type][level] = calc_prf(metrics[level][entity_type])
        
        # Análisis semántico
        if self.use_semantic:
            if metrics['semantic_matches']:
                final['semantic_analysis'] = {
                    'entities_matched_by_synonym': len(metrics['semantic_matches']),
                    'average_similarity_score': round(
                        np.mean([m['similarity'] for m in metrics['semantic_matches']]), 4
                    ),
                    'threshold_used': self.semantic_matcher.threshold,
                    'top_synonyms': self._get_top_synonyms(metrics['semantic_matches'])
                }
            else:
                final['semantic_analysis'] = {
                    'entities_matched_by_synonym': 0,
                    'average_similarity_score': 0.0,
                    'threshold_used': self.semantic_matcher.threshold,
                    'top_synonyms': []
                }
            
            # Mejora por semántica
            strict_f1 = final['metrics']['strict']['f1']
            semantic_f1 = final['metrics']['semantic']['f1']
            final['semantic_analysis']['improvement_rate'] = round(semantic_f1 - strict_f1, 4)
        
        return final
    
    def _get_top_synonyms(self, semantic_matches: List[Dict], top_k: int = 10) -> List[Dict]:
        """Obtiene los sinónimos más frecuentes."""
        synonym_counts = Counter()
        for match in semantic_matches:
            pair = tuple(sorted([match['pred'], match['gold']]))
            synonym_counts[pair] += 1
        
        top_synonyms = []
        for (term1, term2), count in synonym_counts.most_common(top_k):
            # Encontrar la similitud promedio
            similarities = [m['similarity'] for m in semantic_matches 
                          if sorted([m['pred'], m['gold']]) == sorted([term1, term2])]
            top_synonyms.append({
                'term1': term1,
                'term2': term2,
                'count': count,
                'avg_similarity': round(np.mean(similarities), 4)
            })
        
        return top_synonyms


class ModelEvaluator:
    """Evalúa modelos NER sobre el conjunto de test."""
    
    def __init__(self, test_dir: str, models_dir: str, models_to_evaluate: List[str] = None):
        self.test_dir = Path(test_dir)  # Directorio donde están los archivos de test
        self.models_dir = Path(models_dir)
        self.models_to_evaluate = models_to_evaluate or []
        self.metrics_calculator = NERMetricsCalculator(use_semantic=True)
        self.results = {}
        self.test_data_cache = {}  # Cache para no recargar archivos idénticos

    def _load_test_data(self, model_name: str, test_file_path: Path) -> List[Dict]:
        """Carga datos de test para un modelo específico con cache."""
        # Usar cache si ya cargamos este archivo
        file_str = str(test_file_path)
        if file_str in self.test_data_cache:
            print(f"  Usando datos de test en cache para {test_file_path.name}")
            return self.test_data_cache[file_str]
        
        print(f"  Cargando archivo de test: {test_file_path.name}")
        test_data = self._load_jsonl(test_file_path)
        print(f"  Documentos de test cargados: {len(test_data)}")
        
        # Guardar en cache
        self.test_data_cache[file_str] = test_data
        
        return test_data

    def _process_text(self, text: str, doc_id: str, ner_pipeline) -> List[Dict]:
        """Procesa texto con umbral de confianza ajustable y fusión de wordpieces."""
        
        # Obtener predicciones SIN agregación para controlar el umbral
        ner_results_raw = ner_pipeline(text, aggregation_strategy="none")
        
        entities = []
        current_entity = None
        
        for pred in ner_results_raw:
            entity_type = pred['entity']
            score = pred['score']
            word = pred['word']
            
            # Obtener umbral específico o usar el global
            threshold = CONFIDENCE_THRESHOLDS.get(entity_type, CONFIDENCE_THRESHOLDS['global'])
            
            # FILTRAR: Solo procesar si supera el umbral Y no es O
            if entity_type not in ['O', 'LABEL_0'] and score >= threshold:
                
                if entity_type.startswith('B-'):
                    # Si había una entidad en proceso, guardarla
                    if current_entity:
                        entities.append(current_entity)
                    
                    # Iniciar nueva entidad
                    current_entity = {
                        'text': word.replace('##', ''),  # Quitar ## si lo tiene
                        'type': entity_type[2:],  # Quitar el prefijo B-
                        'start': pred['start'],
                        'end': pred['end']
                    }
                    
                elif entity_type.startswith('I-') and current_entity:
                    # Continuar la entidad actual si es del mismo tipo
                    expected_type = 'I-' + current_entity['type']
                    if entity_type == expected_type:
                        # Fusionar el token con la entidad actual
                        if word.startswith('##'):
                            # Es una subpalabra, fusionar sin espacio
                            current_entity['text'] += word.replace('##', '')
                        else:
                            # Es una palabra separada, añadir con espacio
                            current_entity['text'] += ' ' + word
                        current_entity['end'] = pred['end']
                
                # NUEVO: Manejar casos donde no hay B- o I- pero sí wordpieces
                elif not entity_type.startswith('B-') and not entity_type.startswith('I-'):
                    # Si el token actual empieza con ## y hay una entidad previa del mismo tipo
                    if word.startswith('##') and current_entity and current_entity['type'] == entity_type:
                        # Fusionar con la entidad anterior
                        current_entity['text'] += word.replace('##', '')
                        current_entity['end'] = pred['end']
                    else:
                        # Si había una entidad en proceso, guardarla
                        if current_entity:
                            entities.append(current_entity)
                        
                        # Iniciar nueva entidad sin prefijo B/I
                        current_entity = {
                            'text': word.replace('##', ''),
                            'type': entity_type,
                            'start': pred['start'],
                            'end': pred['end']
                        }
        
        # No olvidar la última entidad
        if current_entity:
            entities.append(current_entity)
        
        # NUEVO: Post-procesamiento para fusionar entidades contiguas del mismo tipo
        merged_entities = []
        i = 0
        while i < len(entities):
            current = entities[i]
            
            # Buscar entidades siguientes que puedan fusionarse
            j = i + 1
            while j < len(entities):
                next_entity = entities[j]
                
                # Si la siguiente entidad está justo después y es del mismo tipo
                if (next_entity['start'] == current['end'] and 
                    next_entity['type'] == current['type']):
                    # Fusionar
                    if next_entity['text'].startswith('##'):
                        current['text'] += next_entity['text'].replace('##', '')
                    else:
                        # Verificar si necesita espacio
                        if current['end'] == next_entity['start']:
                            current['text'] += next_entity['text']
                        else:
                            current['text'] += ' ' + next_entity['text']
                    current['end'] = next_entity['end']
                    j += 1
                else:
                    break
            
            merged_entities.append(current)
            i = j
        
        # Debug solo una vez
        if not hasattr(self, '_debug_done'):
            self._debug_done = True
            print(f"\n  DEBUG - Umbral global: {CONFIDENCE_THRESHOLDS['global']}")
            print(f"  DEBUG - Entidades detectadas (después de fusión): {len(merged_entities)}")
            if merged_entities[:5]:
                print(f"  Primeras entidades:")
                for e in merged_entities[:5]:
                    print(f"    {e}")
        
        return merged_entities
    
    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """Carga archivo JSONL."""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                
                # El formato tiene el document_id y entities dentro del campo 'output'
                if 'output' in item:
                    output_data = json.loads(item['output'])
                    # Crear estructura esperada
                    data.append({
                        'input': item['input'],
                        'document_id': output_data['document_id'],
                        'entities': output_data['entities']
                    })
                else:
                    # Si ya tiene el formato esperado, usar directamente
                    data.append(item)
        return data
    
    def evaluate_all_models(self):
        """Evalúa todos los modelos en el directorio."""
        if not self.models_to_evaluate:
            print("Error: No se especificaron modelos para evaluar")
            return
        
        print(f"\nModelos a evaluar: {len(self.models_to_evaluate)}")
        for model_name in self.models_to_evaluate:
            test_file = MODEL_TEST_FILES.get(model_name, "SIN ARCHIVO")
            print(f"  - {model_name} -> {test_file}")
        print("\nBuscando checkpoints adicionales...")
        checkpoints_added = []
        
        # Hacer una copia de la lista original para iterar
        original_models = self.models_to_evaluate.copy()
        
        for model_name in original_models:
            # Solo procesar modelos base (no los que ya son checkpoints)
            if 'checkpoint-' not in model_name and 'model-' not in model_name:
                model_dir = self.models_dir / model_name
                
                if model_dir.exists():
                    # Buscar checkpoints
                    checkpoints = sorted(model_dir.glob('checkpoint-*'))
                    
                    if checkpoints:
                        print(f"  📁 {model_name}: encontrados {len(checkpoints)} checkpoints")
                        
                        for checkpoint in checkpoints:
                            if checkpoint.is_dir():
                                checkpoint_name = f"{model_name}/{checkpoint.name}"
                            
                            # Añadir a la lista de evaluación
                            self.models_to_evaluate.append(checkpoint_name)
                            
                            # Heredar el archivo de test del modelo padre
                            if model_name in MODEL_TEST_FILES:
                                MODEL_TEST_FILES[checkpoint_name] = MODEL_TEST_FILES[model_name]
                            
                            checkpoints_added.append(checkpoint_name)
                            print(f"      + {checkpoint.name}")
        
        # Mostrar resumen actualizado
        if checkpoints_added:
            print(f"\n✅ Total de modelos a evaluar (incluyendo checkpoints): {len(self.models_to_evaluate)}")
            print("\nLista completa actualizada:")
            for model_name in self.models_to_evaluate:
                test_file = MODEL_TEST_FILES.get(model_name, "SIN ARCHIVO")
                if model_name in checkpoints_added:
                    print(f"  - {model_name} -> {test_file} [CHECKPOINT]")
                else:
                    print(f"  - {model_name} -> {test_file}")
        else:
            print("\n❌ No se encontraron checkpoints adicionales")
        
        print("=" * 70)
        
        # Verificar que existen los directorios y archivos de test
        missing_models = []
        missing_test_files = []
        
        for model_name in self.models_to_evaluate:
            model_path = self.models_dir / model_name
            if not model_path.exists():
                missing_models.append(model_name)
                print(f"  ⚠️  Modelo no encontrado: {model_path}")
            
            # Verificar archivo de test
            if model_name in MODEL_TEST_FILES:
                test_file = self.test_dir / MODEL_TEST_FILES[model_name]
                if not test_file.exists():
                    missing_test_files.append((model_name, test_file))
                    print(f"  ⚠️  Archivo de test no encontrado: {test_file}")
            else:
                print(f"  ⚠️  No hay mapeo de archivo de test para: {model_name}")
                missing_test_files.append((model_name, None))
        
        if missing_models or missing_test_files:
            print(f"\nAdvertencia: {len(missing_models)} modelos no encontrados, {len(missing_test_files)} archivos de test faltantes")
            response = input("¿Continuar con los modelos disponibles? (s/n): ")
            if response.lower() != 's':
                return
        
        # Evaluar cada modelo especificado
        for model_name in self.models_to_evaluate:
            model_dir = self.models_dir / model_name
            
            if not model_dir.exists():
                print(f"\n⚠️  Saltando {model_name}: directorio no existe")
                continue
            
            # Obtener archivo de test específico
            if model_name not in MODEL_TEST_FILES:
                print(f"\n⚠️  Saltando {model_name}: no hay archivo de test mapeado")
                continue
            
            test_file_path = self.test_dir / MODEL_TEST_FILES[model_name]
            if not test_file_path.exists():
                print(f"\n⚠️  Saltando {model_name}: archivo de test no existe")
                continue
            
            # Cargar datos de test para este modelo
            test_data = self._load_test_data(model_name, test_file_path)
            
            # Buscar subdirectorios model-*
            model_subdirs = [d for d in model_dir.glob('model*') if d.is_dir()]
            if not model_subdirs:
                model_subdirs = [model_dir]
            
            # Evaluar cada subdirectorio como un modelo independiente
            for subdir in model_subdirs:
                model_subname = f"{model_name}/{subdir.name}"
                
                print(f"\n{'='*60}")
                print(f"Evaluando: {model_subname}")
                print(f"Archivo de test: {MODEL_TEST_FILES[model_name]}")
                print(f"{'='*60}")
                
                try:
                    # Determinar tipo de modelo
                    if 'llama' in model_name.lower():
                        predictions = self._run_llama_model(subdir)
                    else:
                        predictions = self._run_bert_model(subdir)
                    
                    # Calcular métricas
                    if predictions:
                        print("Calculando métricas...")
                        metrics = self.metrics_calculator.calculate_metrics(
                            predictions, test_data
                        )
                        self.results[model_subname] = {
                            'metrics': metrics,
                            'timestamp': datetime.now().isoformat(),
                            'model_type': 'llama' if 'llama' in model_name.lower() else 'bert',
                            'test_file': MODEL_TEST_FILES[model_name],
                            'base_model': model_name,
                            'checkpoint': subdir.name
                        }
                        
                        # Mostrar resumen
                        self._print_summary(model_subname, metrics)
                        self.save_results()
                        print(f"  ✓ Resultados guardados en evaluation_results.json")
                        
                except Exception as e:
                    print(f"Error evaluando {model_subname}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.results[model_subname] = {
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }

    def _run_bert_model(self, model_dir: Path) -> List[Dict]:
        """Ejecuta modelo BERT sobre los datos de test."""
        print(f"  Cargando modelo BERT desde {model_dir}")
        
        # Obtener el nombre del modelo desde el path para buscar el archivo de test correcto
        if 'checkpoint-' in str(model_dir):
            # Para checkpoints, usar el nombre del modelo padre
            model_name = model_dir.parent.parent.name if model_dir.name == 'model.safetensors' else model_dir.parent.name
        elif model_dir.name.startswith('model'):
            model_name = model_dir.parent.name
        else:
            model_name = model_dir.name        
        # Cargar los datos de test específicos para este modelo
        if model_name in MODEL_TEST_FILES:
            test_file_path = self.test_dir / MODEL_TEST_FILES[model_name]
            test_data = self._load_test_data(model_name, test_file_path)
        else:
            print(f"  ⚠️  No hay archivo de test mapeado para {model_name}")
            return []
        
        all_predictions = []
        
        try:
            # Cargar modelo y tokenizer directamente del path recibido
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoModelForTokenClassification.from_pretrained(str(model_dir))

            print(f"    ID2LABEL del modelo: {model.config.id2label}")
            if hasattr(model.config, 'id2label') and '0' in model.config.id2label:
                print(f"    ⚠️  Detectado mapeo incorrecto de etiquetas")
            
            # Configurar tokenizer si es necesario
            tokenizer.model_max_length = 512
            
            # Crear pipeline NER
            device = -1
            if EVALUATION_CONFIG['use_gpu'] and torch.cuda.is_available():
                device = 0
                print(f"    Usando GPU")
            
            ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="none",
                device=device,
                ignore_labels=[]
            )
            
            # Procesar documentos
            predictions = []
            batch_size = EVALUATION_CONFIG['batch_size']
            
            # Procesar TODOS los documentos del test
            num_docs_to_process = len(test_data)
            
            print(f"  Procesando {num_docs_to_process} documentos (ya truncados a 512 tokens)...")
            start_time = time.time()
            
            for i in range(0, num_docs_to_process, batch_size):
                batch = test_data[i:i+batch_size]
                
                for doc in batch:
                    # Extraer texto del informe y document_id
                    text = doc['input']
                    doc_id = doc['document_id']
                    
                    # Procesar texto (ya truncado)
                    entities = self._process_text(text, doc_id, ner_pipeline)
                    
                    predictions.append({
                        'document_id': doc_id,
                        'entities': entities
                    })
                
                # Mostrar progreso
                processed = min(i + batch_size, num_docs_to_process)
                if processed % 50 == 0 or processed == num_docs_to_process:
                    elapsed = time.time() - start_time
                    percent = (processed / num_docs_to_process) * 100
                    
                    if processed > 0:
                        time_per_doc = elapsed / processed
                        remaining_docs = num_docs_to_process - processed
                        eta = time_per_doc * remaining_docs
                        
                        print(f"    Procesados: {processed}/{num_docs_to_process} ({percent:.1f}%) "
                              f"- Tiempo restante: {eta/60:.1f} min")
                    else:
                        print(f"    Procesados: {processed}/{num_docs_to_process} ({percent:.1f}%)")
            
            total_time = time.time() - start_time
            print(f"  ✓ Completado en {total_time/60:.1f} minutos")
            all_predictions = predictions
            
        except Exception as e:
            print(f"    Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        
        return all_predictions
    
    def _run_llama_model(self, model_dir: Path) -> List[Dict]:
        """Ejecuta modelo Llama sobre los datos de test."""
        print(f"  Cargando modelo Llama desde {model_dir}")
        
        # Obtener el nombre del modelo desde el path
        model_name = model_dir.name
        
        # Cargar los datos de test específicos para este modelo
        if model_name in MODEL_TEST_FILES:
            test_file_path = self.test_dir / MODEL_TEST_FILES[model_name]
            test_data = self._load_test_data(model_name, test_file_path)
        else:
            print(f"  ⚠️  No hay archivo de test mapeado para {model_name}")
            return []
    
        # Por ahora, retornar predicciones vacías
        predictions = []
        for doc in test_data[:10]:
            predictions.append({
                'document_id': doc['document_id'],
                'entities': []  # Implementar extracción real con Llama
            })
        
        return predictions
    
    def _print_summary(self, model_name: str, metrics: Dict):
        """Imprime resumen de métricas."""
        print(f"\nResultados para {model_name}:")
        print(f"  Documentos evaluados: {metrics['summary']['documents_evaluated']}")
        print(f"  Entidades gold: {metrics['summary']['total_entities_gold']}")
        print(f"  Entidades predichas: {metrics['summary']['total_entities_predicted']}")
        
        print("\n  Métricas globales:")
        for level in ['strict', 'semantic']:
            m = metrics['metrics'][level]
            print(f"    {level.capitalize()}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")
        
        if 'semantic_analysis' in metrics:
            sa = metrics['semantic_analysis']
            print(f"\n  Análisis semántico:")
            print(f"    Mejora por semántica: {sa['improvement_rate']:.3f}")
            print(f"    Entidades por sinónimos: {sa['entities_matched_by_synonym']}")
    
    def save_results(self, output_file: str = 'evaluation_results.json'):
        """Guarda resultados de evaluación."""
        output_path = Path(output_file)
        
        print(f"\nGuardando resultados en {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Mostrar resumen final en consola
        print("\n" + "="*70)
        print("RESUMEN DE EVALUACIÓN")
        print("="*70)
        
        for model_name, result in self.results.items():
            if 'error' in result:
                print(f"{model_name}: ERROR - {result['error']}")
            else:
                metrics = result['metrics']['metrics']
                print(f"{model_name}:")
                print(f"  - F1 Estricto: {metrics['strict']['f1']:.3f}")
                print(f"  - F1 Semántico: {metrics['semantic']['f1']:.3f}")
                print(f"  - Mejora: {metrics['semantic']['f1'] - metrics['strict']['f1']:+.3f}")
        
        print("="*70)


def main():
    """Función principal."""
    # Configuración
    TEST_FILE = "."
    MODELS_DIR = r"C:\Users\Ramses\Desktop\IAgen\bert_NER\models"
    
    # ============================================
    # LISTA DE MODELOS A EVALUAR
    # ============================================
    # Comenta/descomenta los modelos que quieras probar
    
    MODELS_TO_EVALUATE = [
        # Modelos base
        "biobert-radiology",
        "biomedical-roberta", 
        "pubmedbert-ms",
        "roberta-clinical-es",
        "sapbert-pubmed",
        
        # Modelos con prefijo 'real-' (descomenta los que quieras probar)
        "real-biobert-radiology",
        "real-biomedical-roberta",
        "real-pubmedbert-ms",
        "real-roberta-clinical-es",
        "real-sapbert-pubmed-real",
        
        # Modelos Llama (descomenta si quieres probarlos)
        "llama3.2-3b-int4",
        "llama3.2-3b-int8",
    ]
    
    # NOTA: Con 1500 documentos y 5 modelos, la evaluación completa puede tomar ~30-60 minutos
    # Para pruebas rápidas, cambia EVALUATION_CONFIG['num_documents'] a un número menor (ej: 50)
    
    print("="*70)
    print("SISTEMA DE EVALUACIÓN NER MÉDICO")
    print("="*70)
    print(f"Archivo de test: {TEST_FILE}")
    print(f"Directorio de modelos: {MODELS_DIR}")
    print(f"Modelos seleccionados: {len(MODELS_TO_EVALUATE)}")
    print(f"Documentos a procesar: TODOS (test completo)")
    print(f"Usar GPU: {'Sí' if EVALUATION_CONFIG['use_gpu'] and torch.cuda.is_available() else 'No'}")
    print(f"Umbral semántico: {EVALUATION_CONFIG['semantic_threshold']}")
    print("="*70)
    
    # Iniciar cronómetro
    start_time = time.time()
    
    # Crear evaluador
    evaluator = ModelEvaluator(TEST_FILE, MODELS_DIR, MODELS_TO_EVALUATE)
    
    # Evaluar todos los modelos
    evaluator.evaluate_all_models()
    
    # Guardar resultados
    evaluator.save_results()
    
    # Tiempo total
    total_time = time.time() - start_time
    
    print("\n¡Evaluación completada!")
    print(f"Tiempo total: {total_time/60:.1f} minutos")
    print(f"Resultados guardados en: evaluation_results.json")


if __name__ == "__main__":
    main()