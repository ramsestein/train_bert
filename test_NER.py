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
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN GLOBAL
# ============================================

# Configuración de evaluación
EVALUATION_CONFIG = {
    'num_documents': 20,  # Limitado a 20 documentos para prueba rápida
    'batch_size': 10,      # Tamaño de lote para procesamiento
    'use_gpu': True,       # Usar GPU si está disponible
    'semantic_threshold': 0.85,  # Umbral general para sinónimos
    'preferred_model_subdir': None,  # Subdirectorio específico a usar (None para automático)
}

# Directorios de datos reales
TC_ICTUS_DIR = "TC_ictus"        # Textos de entrada (.txt)
BENCHMARK_DIR = "benchmark"      # Anotaciones correctas (.json)

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
    
    def __init__(self, base_dir: str, models_dir: str, models_to_evaluate: List[str] = None):
        self.base_dir = Path(base_dir)  # Directorio base (donde están TC_ictus y benchmark)
        self.tc_ictus_dir = self.base_dir / TC_ICTUS_DIR
        self.benchmark_dir = self.base_dir / BENCHMARK_DIR
        self.models_dir = Path(models_dir)
        self.models_to_evaluate = models_to_evaluate or []
        self.metrics_calculator = NERMetricsCalculator(use_semantic=True)
        self.results = {}
        self.test_data_cache = {}  # Cache para no recargar archivos

    def _load_test_data(self) -> List[Dict]:
        """Carga datos de test desde TC_ictus/ y benchmark/."""
        if self.test_data_cache:
            print(f"  Usando datos de test en cache ({len(self.test_data_cache)} documentos)")
            return self.test_data_cache
        
        print(f"  Cargando datos de test desde {TC_ICTUS_DIR}/ y {BENCHMARK_DIR}/")
        
        # Verificar que existen los directorios
        if not self.tc_ictus_dir.exists():
            raise FileNotFoundError(f"No se encontró directorio: {self.tc_ictus_dir}")
        if not self.benchmark_dir.exists():
            raise FileNotFoundError(f"No se encontró directorio: {self.benchmark_dir}")
        
        # Obtener archivos .txt del directorio TC_ictus
        txt_files = sorted(self.tc_ictus_dir.glob("*.txt"))
        print(f"  Archivos .txt encontrados: {len(txt_files)}")
        
        test_data = []
        missing_annotations = []

        counter = 0
        
        for txt_file in txt_files:
            counter += 1
            if counter > EVALUATION_CONFIG['num_documents'] + 1 :
                break
            
            # Construir nombre del archivo de anotaciones correspondiente
            base_name = txt_file.stem  # Sin la extensión
            json_file = self.benchmark_dir / f"{base_name}.json"
            
            if not json_file.exists():
                missing_annotations.append(base_name)
                continue
            
            try:
                # Leer texto
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                # Leer anotaciones
                with open(json_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                
                # Crear estructura esperada
                test_data.append({
                    'document_id': annotations['document_id'],
                    'input': text,  # El texto completo del documento
                    'entities': annotations['entities']
                })
                
            except Exception as e:
                print(f"    Error procesando {txt_file.name}: {e}")
                continue
        
        if missing_annotations:
            print(f"  ⚠️  Archivos sin anotaciones: {len(missing_annotations)}")
            if len(missing_annotations) <= 5:
                print(f"    Faltantes: {missing_annotations}")
        
        print(f"  Documentos de test cargados: {len(test_data)}")
        
        # Guardar en cache
        self.test_data_cache = test_data
        
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
    
    
    def evaluate_all_models(self):
        """Evalúa todos los modelos en el directorio."""
        if not self.models_to_evaluate:
            print("Error: No se especificaron modelos para evaluar")
            return
        
        # Cargar datos de test una sola vez para todos los modelos
        try:
            test_data = self._load_test_data()
        except Exception as e:
            print(f"❌ Error cargando datos de test: {e}")
            return
        
        print(f"\nModelos a evaluar: {len(self.models_to_evaluate)}")
        for model_name in self.models_to_evaluate:
            print(f"  - {model_name}")
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
                                checkpoints_added.append(checkpoint_name)
                                print(f"      + {checkpoint.name}")
        
        # Mostrar resumen actualizado
        if checkpoints_added:
            print(f"\n✅ Total de modelos a evaluar (incluyendo checkpoints): {len(self.models_to_evaluate)}")
        else:
            print("\n❌ No se encontraron checkpoints adicionales")
        
        print("=" * 70)
        
        # Verificar que existen los directorios de modelos
        missing_models = []
        
        for model_name in self.models_to_evaluate:
            model_path = self.models_dir / model_name
            if not model_path.exists():
                missing_models.append(model_name)
                print(f"  ⚠️  Modelo no encontrado: {model_path}")
        
        if missing_models:
            print(f"\nAdvertencia: {len(missing_models)} modelos no encontrados")
            response = input("¿Continuar con los modelos disponibles? (s/n): ")
            if response.lower() != 's':
                return
        
        # Evaluar cada modelo especificado
        for model_name in self.models_to_evaluate:
            model_dir = self.models_dir / model_name
            
            if not model_dir.exists():
                print(f"\n⚠️  Saltando {model_name}: directorio no existe")
                continue
            
            # Buscar subdirectorios model-* o usar el directorio principal
            model_subdirs = [d for d in model_dir.glob('model*') if d.is_dir()]
            if not model_subdirs:
                model_subdirs = [model_dir]
            
            # Evaluar cada subdirectorio como un modelo independiente
            for subdir in model_subdirs:
                model_subname = f"{model_name}/{subdir.name}" if subdir != model_dir else model_name
                
                print(f"\n{'='*60}")
                print(f"Evaluando: {model_subname}")
                print(f"Datos de test: {len(test_data)} documentos")
                print(f"{'='*60}")
                
                try:
                    # Determinar tipo de modelo
                    if 'llama' in model_name.lower():
                        predictions = self._run_llama_model(subdir, test_data)
                    else:
                        predictions = self._run_bert_model(subdir, test_data)
                    
                    # Calcular métricas
                    if predictions:
                        print("Calculando métricas...")
                        metrics = self.metrics_calculator.calculate_metrics(
                            predictions, test_data
                        )
                        self.results[model_subname] = {
                            'metrics': metrics,
                            'predictions': predictions,  # Guardar predicciones del modelo
                            'timestamp': datetime.now().isoformat(),
                            'model_type': 'llama' if 'llama' in model_name.lower() else 'bert',
                            'base_model': model_name,
                            'checkpoint': subdir.name if subdir != model_dir else 'base'
                        }
                        
                        # Mostrar resumen
                        self._print_summary(model_subname, metrics)
                        self.save_results()
                        self.save_summary_csv()
                        self.save_model_predictions(model_subname, predictions)
                        print(f"  ✓ Resultados guardados en evaluation_results.json, resumen.csv y predicciones_{model_subname.replace('/', '_')}.json")
                        
                except Exception as e:
                    print(f"Error evaluando {model_subname}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.results[model_subname] = {
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }

    def _run_bert_model(self, model_dir: Path, test_data: List[Dict]) -> List[Dict]:
        """Ejecuta modelo BERT sobre los datos de test."""
        print(f"  Cargando modelo BERT desde {model_dir}")
        
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
            
            # Limitar número de documentos según configuración
            if EVALUATION_CONFIG['num_documents'] is not None:
                num_docs_to_process = min(EVALUATION_CONFIG['num_documents'], len(test_data))
                test_data = test_data[:num_docs_to_process]
            else:
                num_docs_to_process = len(test_data)
            
            print(f"  Procesando {num_docs_to_process} documentos (limitado por config)")
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
    
    def _run_llama_model(self, model_dir: Path, test_data: List[Dict]) -> List[Dict]:
        """Ejecuta modelo Llama sobre los datos de test."""
        print(f"  Cargando modelo Llama desde {model_dir}")
    
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
        # Usar ruta absoluta para asegurar guardado correcto
        if not os.path.isabs(output_file):
            output_path = Path.cwd() / output_file
        else:
            output_path = Path(output_file)
        
        # Crear directorio si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGuardando resultados en {output_path.absolute()}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"✅ Archivo guardado exitosamente: {output_path.absolute()}")
        except Exception as e:
            print(f"❌ Error guardando archivo: {e}")
            # Intentar guardar en directorio actual como backup
            backup_path = Path.cwd() / f"backup_{output_file}"
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, ensure_ascii=False)
                print(f"✅ Guardado en backup: {backup_path.absolute()}")
            except Exception as e2:
                print(f"❌ Error en backup también: {e2}")
        
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

    def save_summary_csv(self, output_file: str = 'evaluation_summary.csv'):
        """Guarda un resumen en CSV para análisis rápido."""
        
        # Crear ruta absoluta
        if not os.path.isabs(output_file):
            output_path = Path.cwd() / output_file
        else:
            output_path = Path(output_file)
        
        # Preparar datos para CSV
        summary_data = []
        for model_name, result in self.results.items():
            if 'error' not in result:
                metrics = result['metrics']['metrics']
                row = {
                    'Modelo': model_name,
                    'F1_Estricto': round(metrics['strict']['f1'], 4),
                    'Precision_Estricto': round(metrics['strict']['precision'], 4),
                    'Recall_Estricto': round(metrics['strict']['recall'], 4),
                    'F1_Semantico': round(metrics['semantic']['f1'], 4),
                    'Precision_Semantico': round(metrics['semantic']['precision'], 4),
                    'Recall_Semantico': round(metrics['semantic']['recall'], 4),
                    'Mejora_Semantica': round(metrics['semantic']['f1'] - metrics['strict']['f1'], 4),
                    'Entidades_Gold': result['metrics']['summary']['total_entities_gold'],
                    'Entidades_Predichas': result['metrics']['summary']['total_entities_predicted'],
                    'Documentos_Evaluados': result['metrics']['summary']['documents_evaluated'],
                    'Tipo_Modelo': result.get('model_type', 'unknown'),
                    'Timestamp': result['timestamp']
                }
                summary_data.append(row)
        
        if summary_data:
            try:
                df = pd.DataFrame(summary_data)
                df = df.sort_values('F1_Semantico', ascending=False)
                df.to_csv(output_path, index=False, encoding='utf-8')
                print(f"✅ Resumen CSV guardado: {output_path.absolute()}")
            except Exception as e:
                print(f"❌ Error guardando CSV: {e}")
        else:
            print("⚠️  No hay datos para guardar en CSV")

    def save_model_predictions(self, model_name: str, predictions: List[Dict], output_dir: str = 'predictions'):
        """Guarda las predicciones brutas de un modelo específico."""
        # Crear directorio para predicciones
        pred_dir = Path.cwd() / output_dir
        pred_dir.mkdir(exist_ok=True)
        
        # Nombre de archivo seguro (reemplazar caracteres problemáticos)
        safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        pred_file = pred_dir / f"predicciones_{safe_model_name}.json"
        
        # Preparar datos con formato más legible
        output_data = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'total_documents': len(predictions),
            'predictions': predictions
        }
        
        try:
            with open(pred_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"    📄 Predicciones guardadas: {pred_file}")
        except Exception as e:
            print(f"    ❌ Error guardando predicciones: {e}")

    def save_all_predictions_combined(self, output_file: str = 'all_predictions.json'):
        """Guarda todas las predicciones de todos los modelos en un solo archivo."""
        if not os.path.isabs(output_file):
            output_path = Path.cwd() / output_file
        else:
            output_path = Path(output_file)
        
        # Recopilar todas las predicciones
        all_predictions = {}
        for model_name, result in self.results.items():
            if 'predictions' in result:
                all_predictions[model_name] = {
                    'timestamp': result['timestamp'],
                    'model_type': result.get('model_type', 'unknown'),
                    'base_model': result.get('base_model', model_name),
                    'predictions': result['predictions']
                }
        
        # Guardar archivo combinado
        if all_predictions:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_predictions, f, indent=2, ensure_ascii=False)
                print(f"✅ Todas las predicciones guardadas: {output_path.absolute()}")
            except Exception as e:
                print(f"❌ Error guardando archivo combinado: {e}")


def main():
    """Función principal."""
    # Configuración
    BASE_DIR = "."  # Directorio donde están TC_ictus/ y benchmark/
    MODELS_DIR = "models"  # Cambiar por tu ruta de modelos
    
    # ============================================
    # LISTA DE MODELOS A EVALUAR
    # ============================================
    # Comenta/descomenta los modelos que quieras probar
    
    MODELS_TO_EVALUATE = [
        # Modelos BERT entrenados con train_NER.py
        "pubmedbert-optimized",
        "roberta-clinical-es",
        "roberta-biomedical-es", 
        "biobert-dmis",
        "sapbert",
    ]
    
    print("="*70)
    print("SISTEMA DE EVALUACIÓN NER MÉDICO")
    print("="*70)
    print(f"Directorio base: {BASE_DIR}")
    print(f"Textos de entrada: {TC_ICTUS_DIR}/")
    print(f"Anotaciones gold: {BENCHMARK_DIR}/")
    print(f"Directorio de modelos: {MODELS_DIR}")
    print(f"Modelos seleccionados: {len(MODELS_TO_EVALUATE)}")
    docs_msg = f"{EVALUATION_CONFIG['num_documents']}" if EVALUATION_CONFIG['num_documents'] else "TODOS"
    print(f"Documentos a procesar: {docs_msg}")
    print(f"Usar GPU: {'Sí' if EVALUATION_CONFIG['use_gpu'] and torch.cuda.is_available() else 'No'}")
    print(f"Umbral semántico: {EVALUATION_CONFIG['semantic_threshold']}")
    print("="*70)
    
    # Iniciar cronómetro
    start_time = time.time()
    
    # Crear evaluador
    evaluator = ModelEvaluator(BASE_DIR, MODELS_DIR, MODELS_TO_EVALUATE)
    
    # Evaluar todos los modelos
    evaluator.evaluate_all_models()
    
    # Guardar resultados finales
    evaluator.save_results()
    evaluator.save_all_predictions_combined()
    
    # Tiempo total
    total_time = time.time() - start_time
    
    print("\n¡Evaluación completada!")
    print(f"Tiempo total: {total_time/60:.1f} minutos")
    print(f"Resultados guardados en:")
    print(f"  - evaluation_results.json (métricas + predicciones)")
    print(f"  - evaluation_summary.csv (resumen de métricas)")
    print(f"  - predictions/ (predicciones por modelo)")
    print(f"  - all_predictions.json (todas las predicciones combinadas)")


if __name__ == "__main__":
    main()