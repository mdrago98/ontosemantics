# OntoSemantics

*Building machines that learn to learn through ontology-grounded validation*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

OntoSemantics is a novel approach to biomedical relationship extraction that combines Large Language Models with ontology-grounded validation to create self-improving systems. Unlike traditional RAG systems with static knowledge bases, OntoSemantics creates a feedback loop where generated answers improve the knowledge base through semantic validation.

### Key Innovation

```
Traditional RAG: Query → Static Knowledge Base → Retrieve → Generate
OntoSemantics:   Query → Dynamic Knowledge Base ← Learn ← Validate ← Generate
```

**Results**: 3x improvement in F1-score (0.0 → 0.44) for biomedical relationship extraction through ontological context integration.

## Features

- **🔄 Self-Improving Architecture**: Knowledge base learns from every query through ontology validation
- **🧬 Biomedical Focus**: Specialized for medical literature and research applications  
- **📊 Multiple Ontologies**: Integrated support for MONDO, Gene Ontology, Human Phenotype Ontology
- **⚡ Real-time Validation**: Live checking against authoritative knowledge sources
- **📈 Measurable Progress**: Track knowledge graph growth and accuracy improvements over time
- **🔍 Relationship Extraction**: Advanced biomedical entity relationship discovery

## Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Ollama (for local LLM inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/mdrago98/ontosemantics.git
cd ontosemantics

# Install dependencies
pip install -r requirements.txt
# OR using conda
conda env create -f environment.yml
conda activate ontosemantics
```

### Setup

1. **Start required services**:
```bash
docker-compose up -d
```

2. **Download ontologies**:
```python
from knowledge_engine.ontology_manager import OntologyManager
om = OntologyManager()
await om.download_and_load_ontologies()
```

3. **Initialize LLM extractor**:
```python
from nlp_processor.llm_extractor import LLMRelationshipExtractor
extractor = LLMRelationshipExtractor('gemma3:1b')
```

## Usage

### Basic Relationship Extraction

```python
# Extract relationships without context
relationships = extractor.extract_relationships(text)

# Extract with entity context
relationships = extractor.extract_relationships(
    text, 
    context={'entities': ['insulin', 'diabetes', 'glucose']}
)

# Extract with full ontological context
semantic_context = om.get_semantic_context(['insulin', 'diabetes'])
relationships = extractor.extract_relationships(
    text,
    context={'semantic_relationships': semantic_context}
)
```

### Evaluation

```python
from utils.eval import RelationshipEvaluator

evaluator = RelationshipEvaluator(matching_strategy="fuzzy")
metrics = evaluator.evaluate(predicted_relationships, ground_truth)
print(f"F1-Score: {metrics.overall_metrics.f1_score}")
```

### Ontology Integration

```python
# Validate and enrich entities with ontological knowledge
matches = om.validate_and_enrich_entity('type 2 diabetes')
for match in matches:
    print(f"Parents: {[p.name for p in match.parents]}")
    print(f"Children: {[c.name for c in match.children]}")
```

## Project Structure

```
ontosemantics/
├── knowledge_engine/          # Core ontology processing
│   ├── ontology_manager.py   # Ontology loading and management
│   └── models/               # Data models for entities and relationships
├── nlp_processor/            # LLM-based extraction
│   └── llm_extractor.py     # Relationship extraction with context
├── utils/                    # Evaluation and utilities
│   └── eval.py              # Metrics calculation and evaluation
├── notebooks/                # Jupyter notebooks and experiments
│   └── ontology.ipynb       # Main experiment notebook
├── data/                     # Datasets and ontologies
│   ├── BioRED/              # BioRED challenge dataset
│   └── ontologies/          # Downloaded ontology files
└── docker-compose.yml       # Service orchestration
```

## Experimental Results

| Context Type | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| No Context   | 0.000     | 0.000  | 0.000    |
| Entity Context | 0.083   | 0.333  | 0.133    |
| **Semantic Context** | **0.333** | **0.667** | **0.444** |

*Results on BioRED dataset sample showing dramatic improvement with ontological context*

## Supported Ontologies

- **MONDO**: Disease Ontology (56,695+ terms)
- **Gene Ontology (GO)**: Biological processes and molecular functions (48,106+ terms)  
- **Human Phenotype Ontology (HP)**: Phenotypic abnormalities (19,653+ terms)
- **UBERON**: Anatomical structures *(planned)*
- **ChEBI**: Chemical entities *(planned)*

## Configuration

### LLM Models
Currently supports Ollama-compatible models:
- `gemma3:1b` (default, lightweight)
- `llama3:8b` (better performance)
- `mistral:7b` (alternative option)

### Ontology Sources
Ontologies are automatically downloaded from:
- [MONDO](http://purl.obolibrary.org/obo/mondo.obo)
- [Gene Ontology](http://purl.obolibrary.org/obo/go.obo)  
- [Human Phenotype Ontology](http://purl.obolibrary.org/obo/hp.obo)

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Ontologies
```python
# Add to ontology_manager.py ONTOLOGY_URLS
ONTOLOGY_URLS = {
    'your_ontology': 'http://purl.obolibrary.org/obo/your_ontology.obo'
}
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Work

- **Embedding-Based Context**: Pre-computed semantic embeddings for faster context selection
- **Multi-Modal Integration**: Combining structured graphs with LLM embeddings
- **Hierarchical Embeddings**: Preserving parent-child relationships in vector space
- **Real-time Knowledge Graph Updates**: Live integration of validated relationships
- **Cross-Domain Transfer**: Extending beyond biomedical to other knowledge domains

## Citation

If you use OntoSemantics in your research, please cite:

```bibtex
@misc{ontosemantics2024,
  title={OntoSemantics: Self-Improving Ontology-RAG Systems for Biomedical Relationship Extraction},
  author={Matthew Drago},
  year={2025},
  url={https://github.com/mdrago98/ontosemantics}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [BioRED Challenge](https://academic.oup.com/bib/article/doi/10.1093/bib/bbac282/6645993) for the evaluation dataset
- [Pronto](https://github.com/althonos/pronto) for ontology processing
- [Ollama](https://ollama.ai/) for local LLM inference
- Open Biomedical Ontologies Foundry for ontology standards

## Contact

- **Author**: Matthew Drago
- **Blog**: [Here](https://www.matthewdrago.com/blog/ontosemantics)

---

*Building the future of AI-powered biomedical research, one relationship at a time.*