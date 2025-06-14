# MetaIntelligence: The Intelligence of Intelligent Systems

## 🌟 Breaking Through "The Illusion of Thinking" - A Self-Evolving AI Platform

**MetaIntelligence** is a revolutionary AI integration system designed as "the intelligence of intelligent systems". Moving beyond mere LLM extension, it functions as a truly intelligent entity with self-awareness, self-improvement, and self-evolution capabilities. This platform implements solutions to overcome fundamental limitations identified in Apple Research's groundbreaking paper ["The Illusion of Thinking"](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf).

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Apple_Research-orange.svg)](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf)
[![Status](https://img.shields.io/badge/Status-Research_Implementation_v2.1-brightgreen.svg)](README.md)

---

## 🚀 Core Features & Modes

MetaIntelligence introduces powerful new features including Retrieval-Augmented Generation (RAG), quantum-inspired reasoning, and self-improving complexity analysis, all orchestrated under a meta-cognitive framework.

### Key Concepts

- **True Self-Awareness**: The system understands and improves its own thought processes.
- **Dynamic Architecture**: Optimizes its internal structure during runtime.
- **Value Evolution**: Learns and evolves its value judgment criteria from experience.
- **Emergent Problem Discovery**: Proactively identifies hidden problems unnoticed by humans.
- **Transcendent Wisdom Generation**: Synthesizes ultimate wisdom by integrating multiple intelligent systems.
- **Consciousness Evolution**: Gradual advancement and transcendence of consciousness levels.

### Research-Based Reasoning Modes

Our system offers a comprehensive suite of reasoning modes designed to tackle problems of any complexity level, now enhanced with advanced strategies.

| Mode | Complexity Target | Primary Benefit | Use Case |
|:-----|:------------------|:----------------|:---------|
| `efficient` | Low | **Overthinking Prevention** | Quick questions, basic tasks |
| `balanced` | Medium | Optimal Reasoning Quality | Standard analysis, explanations |
| `decomposed` | High | **Collapse Prevention & Speed** | Complex problem-solving, system design |
| `adaptive` | Auto-detected | **Dynamic Optimization** | Questions of unknown or mixed complexity |
| `parallel` | All | **Best-of-Breed Quality** | Mission-critical tasks, maximum quality |
| `quantum_inspired` | All | Holistic, Synthesized Insight | Brainstorming, philosophical questions, strategy |
| `edge` | Low | Lightweight & Fast | Low-resource devices, quick checks |
| `speculative_thought` | All | **Exploratory, Rapid Prototyping** | Early-stage ideation, multiple perspectives |
| `self_discover` | All | **Autonomous Strategy Construction** | Novel or ill-defined problems |
| `paper_optimized` | All | Complete Research Integration | Maximum research benefit, benchmarking |


### Advanced Features

- **Automatic Complexity Analysis**: Real-time, multi-language problem complexity assessment.
- **Self-Improving Analysis**: Learns from past interactions to make smarter future decisions.
- **Dynamic Strategy Selection**: Optimal reasoning approach chosen per problem.
- **Retrieval-Augmented Generation (RAG)**: Augments prompts with external knowledge from **Wikipedia** or local files/URLs.
- **Overthinking & Collapse Prevention**: Core mechanisms to maintain efficiency and reasoning quality.
- **Multi-Provider Support**: OpenAI, Claude, Gemini, Ollama, HuggingFace, Llama.cpp.
- **Structured Thought Process**: Outputs detailed reasoning steps for full transparency.

---

## 🛠️ Installation & Quick Start

### Requirements

- Python 3.10+
- `pip` package manager
- At least one LLM provider API key or a local Ollama/Llama.cpp setup
- `ffmpeg` (for audio processing, optional)

### Setup

```bash
# Clone the repository
git clone https://github.com/littlebuddha-dev/MetaIntelligence.git
cd MetaIntelligence

# Install all required dependencies
pip install -r requirements.txt

# Install spaCy models for advanced NLP-based complexity analysis (optional but recommended)
python -m spacy download en_core_web_sm
python -m spacy download ja_core_news_sm

# Create and configure your environment file
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, CLAUDE_API_KEY, GEMINI_API_KEY, HF_TOKEN, etc.)
# For Ollama, ensure OLLAMA_API_BASE_URL is correct.

# For local models with Ollama:
# Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh
# Start Ollama server: ollama serve
# Pull recommended models:
# ollama pull gemma3:latest      # Recommended efficient model
# ollama pull deepseek-r1        # Reasoning-specialized (if available)
# ollama pull llama3.1:latest    # General purpose
# Verify Ollama models: ollama list

# For local models with Llama.cpp:
# Ensure llama-cpp-python[server] is installed from requirements.txt
# Download a .gguf model, e.g., Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
# Place it in a 'models' directory or configure LLAMACPP_DEFAULT_MODEL_PATH in .env
# Start the Llama.cpp server: python -m llama_cpp.server --model <path_to_your_model.gguf>

# Test your setup
python quick_test_v2.py
```

### Immediate Usage (CLI)

The `fetch_llm_v2.py` script serves as the main command-line interface.

```bash
# Basic efficiency test (overthinking prevention)
python fetch_llm_v2.py ollama "What is the capital of Japan?" --mode efficient --model gemma3:latest

# Medium complexity balanced reasoning
python fetch_llm_v2.py claude "Explain the main causes of climate change." --mode balanced

# High complexity decomposed reasoning (collapse prevention)
python fetch_llm_v2.py openai "Design a sustainable urban transportation system." --mode decomposed

# Adaptive mode for unknown complexity
python fetch_llm_v2.py gemini "Analyze the economic impact of AI on the job market." --mode adaptive

# Quantum-inspired synthesis for deep insights
python fetch_llm_v2.py openai "What is the nature of consciousness?" --mode quantum_inspired

# Speculative thought for diverse initial ideas
python fetch_llm_v2.py ollama "Generate three innovative business ideas for a remote work future." --mode speculative_thought

# RAG-powered query using Wikipedia
python fetch_llm_v2.py openai "What were the key findings of the LIGO experiment?" --mode balanced --wikipedia

# RAG-powered query using a local knowledge base (e.g., a PDF document)
# First, create a knowledge base (e.g., load a PDF into a file called my_research.pdf)
# Then, use it with the --rag flag
# python fetch_llm_v2.py openai "Summarize the key points of the report on renewable energy." --mode balanced --rag --knowledge-base my_research.pdf
```

## 🎮 Usage Examples

### Academic Research with RAG

```bash
# Augment a query with the latest information from Wikipedia
python fetch_llm_v2.py claude \
"Based on recent findings, what are the main challenges in quantum computing?" \
--mode balanced --wikipedia
```

### Business Strategy Synthesis

```bash
# Use quantum-inspired mode to generate a multi-faceted strategy
python fetch_llm_v2.py openai \
"Develop a holistic market entry strategy for an AI startup in Europe, considering optimistic, pessimistic, and ethical viewpoints." \
--mode quantum_inspired
```

### Technical Problem Solving

```bash
# Design a complex system using the decomposition strategy
python fetch_llm_v2.py ollama \
"Design a high-availability, scalable microservices architecture for a social media app." \
--mode decomposed --model deepseek-r1
```

### Master System Usage (Python API)

The core of MetaIntelligence's advanced capabilities lies in its Master Integration System, which orchestrates various internal AI systems for ultimate problem-solving.

```python
# /examples/master_system_api_usage.py
# Title: Master System API Usage Example
# Role: Demonstrates how to use the MetaIntelligence Master Integration System directly in Python.

import asyncio
from llm_api.providers import get_provider
from llm_api.master_system.integration_orchestrator import MasterIntegrationOrchestrator, IntegrationConfig

async def use_master_system():
    # Initialize a primary provider (e.g., Ollama or OpenAI)
    provider = get_provider("ollama", enhanced=True) # Ensure 'enhanced=True' for V2 capabilities
    
    # Configure the integration system to enable all advanced features
    config = IntegrationConfig(enable_all_systems=True)
    orchestrator = MasterIntegrationOrchestrator(provider, config)
    
    # Initialize the integrated system (activates meta-cognition, dynamic architecture, etc.)
    print("🌟 Initializing MetaIntelligence Master System...")
    init_result = await orchestrator.initialize_integrated_system()
    print("✅ Initialization complete!")
    print(f"Integration Harmony Score: {init_result.get('integration_harmony', 'N/A'):.2f}")
    
    # Solve an ultimate, integrated problem
    problem_statement = "What is the optimal balance between artificial intelligence and human flourishing in the future?"
    print(f"\n🎯 Solving ultimate problem: {problem_statement[:80]}...")
    
    solution = await orchestrator.solve_ultimate_integrated_problem(
        problem_statement,
        context={"cli_demo": False, "complexity": "transcendent"},
        use_full_integration=True
    )
    
    print("\n✨ Ultimate Problem Solved!")
    print(f"Transcendent Solution:\n{solution.get('integrated_solution', '')[:500]}...")
    print(f"\nTranscendence Achieved: {solution.get('transcendence_achieved', False)}")
    print(f"Self-Evolution Triggered: {solution.get('self_evolution_triggered', False)}")
    print(f"Wisdom Distilled: {solution.get('wisdom_distillation', '')}")
    
    # Example of evolving consciousness
    print("\n🧬 Evolving Integrated Consciousness...")
    consciousness_evolution_result = await orchestrator.evolve_integrated_consciousness()
    print(f"Consciousness Evolution Successful: {consciousness_evolution_result.get('consciousness_evolution_successful', False)}")
    print(f"New Collective Consciousness Level: {consciousness_evolution_result.get('new_collective_level', 'N/A'):.3f}")

if __name__ == "__main__":
    asyncio.run(use_master_system())
```

## 🔬 Architecture & Implementation

The MetaIntelligence (formerly CogniQuantum V2) architecture integrates new components for RAG, self-learning, and advanced reasoning pipelines under a unified master system.

### Core Components

The system is built upon a hierarchical structure:

```
MetaIntelligence Master System
├── Meta-Cognition Engine (メタ認知エンジン)
├── Dynamic Architecture System (動的アーキテクチャ)
├── SuperIntelligence Orchestrator (超知能統合)
├── Value Evolution Engine (価値進化システム)
├── Problem Discovery Engine (問題発見システム)
└── Quantum Reasoning Core (量子推論コア)
```

For a detailed breakdown of the project structure and each component, refer to `docs/architecture.md`.

### Research Implementation Pipeline (V2 Logic)

1. **Input**: User prompt is received.
2. **RAG (Optional)**: RAGManager extracts a search query, retrieves context from Wikipedia or local files, and augments the prompt.
3. **Complexity Analysis**: AdaptiveComplexityAnalyzer assesses the complexity, potentially using suggestions from the ComplexityLearner.
4. **Pipeline Selection**: The system chooses a reasoning pipeline (EnhancedReasoningEngine, QuantumReasoningEngine, ParallelPipeline, SpeculativePipeline) based on the selected mode.
5. **Execution**: The chosen engine processes the prompt, applying regime-specific strategies (e.g., decomposition, overthinking prevention).
6. **Self-Correction (Adaptive Mode)**: The system can re-evaluate an insufficient solution and escalate the complexity regime for another attempt.
7. **Learning**: If the complexity was adjusted, ComplexityLearner records the successful outcome to improve future analyses.
8. **Output**: A final, structured solution with a detailed thought process is returned.

## 🤝 Supported Providers

MetaIntelligence supports a wide range of LLM providers, both standard and enhanced with V2 research-based optimizations.

| Provider | Standard | Enhanced V2 | Special Features |
|:---------|:---------|:------------|:----------------|
| OpenAI | ✅ | ✅ | GPT-4o optimization, vision support |
| Claude | ✅ | ✅ | Reasoning specialization, large context |
| Gemini | ✅ | ✅ | Multimodal, speed optimization |
| Ollama | ✅ | ✅ | Local models, privacy, cost-free, concurrency-safe |
| HuggingFace | ✅ | ✅ | Open-source diversity, experimentation |
| Llama.cpp | ✅ | ✅ | High-performance local inference |

## 🧪 Testing & Validation

A comprehensive test suite is included to ensure system stability and performance.

```bash
# Quick system health check
python quick_test_v2.py

# Run a comprehensive test across all available V2 providers and modes
python test_all_v2_providers.py
```

## 📚 Further Documentation

- **Project Structure**: `docs/architecture.md`
- **Installation Guide**: `docs/installation_guide.md`
- **Meta-AI System Concepts**: `docs/concepts/meta_ai_system_concept.md`
- **Evolution Roadmap**: `docs/roadmap.md`
- **CLI Command Reference**: `docs/cli_guide.md`
- **API Reference**: `docs/api_reference.md`
- **Usage Examples**: `docs/usage_examples.md`
- **Troubleshooting Guide**: See `python fetch_llm_v2.py --troubleshooting` or `python quick_test_v2.py --troubleshooting`

## 📜 License

MIT License - see the [LICENSE](LICENSE) file for details.

This is an independent research implementation based on the publicly available Apple Research paper and is not an official Apple product.

---

> "The true intelligence is to know what you know, to know what you don't know, and most importantly, to know how to keep learning."