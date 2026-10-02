# Repositorio en GitHub

Proyecto publicado en [alexdmngz/rag-mpe-benchmark](https://github.com/alexdmngz/rag-mpe-benchmark).

| Campo | Valor |
| --- | --- |
| Propietario | `alexdmngz` |
| Nombre | `rag-mpe-benchmark` |
| Visibilidad | Público |
| Descripción sugerida | `Academic RAG benchmark comparing BM25, dense and hybrid retrieval on technical multiple-choice QA.` |
| Topics sugeridos | `rag`, `information-retrieval`, `bm25`, `semantic-search`, `llm-evaluation`, `python`, `chromadb`, `gemini` |

## Descargar y trabajar con el proyecto

```bash
git clone https://github.com/alexdmngz/rag-mpe-benchmark.git
cd rag-mpe-benchmark
python src/Script.py --mode prepare
```

Consulta el README para crear un entorno virtual, instalar dependencias y configurar Gemini. El modo `prepare` no usa la API.

## Subir cambios posteriores

Desde tu copia local, revisa los cambios y ejecuta:

```bash
git add .
git diff --cached --stat
git commit -m "Describe the change"
git push
```

No añadas `.env`, credenciales, entornos virtuales ni resultados generados. El `.gitignore` preparado los excluye. Conserva la autoría de ambos participantes y la atribución del artículo y dataset descrita en `ATTRIBUTION.md`.

## Comprobaciones automáticas

La tarea **Python checks** se ejecuta al enviar cambios o abrir una pull request. Puedes consultar su resultado en [Actions](https://github.com/alexdmngz/rag-mpe-benchmark/actions). Comprueba los helpers, los datos, BM25 y la conexión interna del experimento con dobles de prueba. No consume llamadas a Gemini ni descarga modelos neuronales.
