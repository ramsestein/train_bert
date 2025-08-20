import json
import pandas as pd
from collections import Counter

def diagnosticar_datos(file_path):
    """Diagnóstico completo del archivo de datos"""
    
    print("="*60)
    print("🔍 DIAGNÓSTICO DE DATOS NER")
    print("="*60)
    
    errors = []
    valid = 0
    total = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            total += 1
            if i < 5:  # Ver primeros ejemplos
                try:
                    item = json.loads(line)
                    
                    print(f"\n📝 Ejemplo {i+1}:")
                    print(f"  Texto (primeros 100 chars): {item['input'][:100]}...")
                    
                    # Verificar entidades
                    if isinstance(item['output'], str):
                        output = json.loads(item['output'])
                        entities = output['entities']
                    else:
                        entities = item['output']['entities']
                    
                    print(f"  Número de entidades: {len(entities)}")
                    
                    # Verificar cada entidad
                    for j, entity in enumerate(entities[:3]):  # Primeras 3 entidades
                        text_span = item['input'][entity['start']:entity['end']]
                        print(f"  Entidad {j+1}:")
                        print(f"    - Tipo: {entity['type']}")
                        print(f"    - Texto: '{entity.get('text', 'N/A')}'")
                        print(f"    - Span extraído: '{text_span}'")
                        print(f"    - Posición: [{entity['start']}:{entity['end']}]")
                        
                        # Verificar si coinciden
                        if 'text' in entity and entity['text'] != text_span:
                            print(f"    ⚠️ PROBLEMA: El texto no coincide con el span!")
                            print(f"       Esperado: '{entity['text']}'")
                            print(f"       Obtenido: '{text_span}'")
                    
                    valid += 1
                    
                except Exception as e:
                    print(f"  ❌ Error en línea {i+1}: {e}")
                    errors.append((i, str(e)))
    
    print(f"\n📊 RESUMEN:")
    print(f"  Total líneas: {total}")
    print(f"  Válidas: {valid}")
    print(f"  Con errores: {len(errors)}")
    
    if errors:
        print(f"\n❌ Primeros errores:")
        for line_num, error in errors[:5]:
            print(f"  Línea {line_num}: {error}")
    
    return errors

# Ejecutar diagnóstico
file_path = "/content/training_real_ictus.jsonl"
errors = diagnosticar_datos(file_path)

# Verificar problema específico con índices
print("\n" + "="*60)
print("🔍 VERIFICACIÓN DE ÍNDICES")
print("="*60)

with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 100:  # Solo revisar primeros 100
            break
        try:
            item = json.loads(line)
            text = item['input']
            
            if isinstance(item['output'], str):
                entities = json.loads(item['output'])['entities']
            else:
                entities = item['output']['entities']
            
            for entity in entities:
                start = entity['start']
                end = entity['end']
                
                # Verificar índices válidos
                if start < 0 or end > len(text) or start >= end:
                    print(f"❌ Línea {i+1}: Índices inválidos [{start}:{end}] para texto de longitud {len(text)}")
                
                # Verificar si el texto coincide
                if 'text' in entity:
                    expected = entity['text']
                    actual = text[start:end]
                    if expected != actual:
                        print(f"❌ Línea {i+1}: Desalineación")
                        print(f"   Esperado: '{expected}'")
                        print(f"   Obtenido: '{actual}'")
                        print(f"   Índices: [{start}:{end}]")
                        
        except Exception as e:
            pass