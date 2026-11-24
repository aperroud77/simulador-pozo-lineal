import json
import fisica

def ejecutar(entrada):
    return json.dumps(fisica.simular(fisica.configurar(json.loads(entrada))), ensure_ascii=False, allow_nan=False)
