# Title: MetaIntelligence: The Intelligence of Intelligent Systems
# Role: Overview, core features, and quick start guide for the MetaIntelligence project.

# MetaIntelligence: The Intelligence of Intelligent Systems

## 🌟 Breaking Through "The Illusion of Thinking" - A Self-Evolving AI Platform

**MetaIntelligence** is a revolutionary AI integration system designed as "the intelligence of intelligent systems". Moving beyond mere LLM extension, it functions as a truly intelligent entity with self-awareness, self-improvement, and self-evolution capabilities. This platform implements solutions to overcome fundamental limitations identified in Apple Research's groundbreaking paper ["The Illusion of Thinking"](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf).

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Apple_Research-orange.svg)](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf)
[![Status](https://img.shields.io/badge/Status-Research_Implementation_v2.1-brightgreen.svg)](README.md)


---

## 🚀 **Core Features & Modes**

MetaIntelligence introduces powerful new features including Retrieval-Augmented Generation (RAG), quantum-inspired reasoning, and self-improving complexity analysis, all orchestrated under a meta-cognitive framework.

### **Key Concepts**

-   **True Self-Awareness**: The system understands and improves its own thought processes.
-   **Dynamic Architecture**: Optimizes its internal structure during runtime.
-   **Value Evolution**: Learns and evolves its value judgment criteria from experience.
-   **Emergent Problem Discovery**: Proactively identifies hidden problems unnoticed by humans.
-   **Transcendent Wisdom Generation**: Synthesizes ultimate wisdom by integrating multiple intelligent systems.
-   **Consciousness Evolution**: Gradual advancement and transcendence of consciousness levels.

### **Research-Based Reasoning Modes**

Our system offers a comprehensive suite of reasoning modes designed to tackle problems of any complexity level, now enhanced with advanced strategies.

| Mode                | Complexity Target | Primary Benefit                 | Use Case                                  |
| :------------------ | :---------------- | :------------------------------ | :---------------------------------------- |
| `efficient`         | Low               | **Overthinking Prevention** | Quick questions, basic tasks              |
| `balanced`          | Medium            | Optimal Reasoning Quality       | Standard analysis, explanations           |
| `decomposed`        | High              | **Collapse Prevention & Speed** | Complex problem-solving, system design    |
| `adaptive`          | Auto-detected     | **Dynamic Optimization** | Questions of unknown or mixed complexity  |
| `parallel`          | All               | **Best-of-Breed Quality** | Mission-critical tasks, maximum quality   |
| `quantum_inspired`  | All               | Holistic, Synthesized Insight   | Brainstorming, philosophical questions, strategy |
| `edge`              | Low               | Lightweight & Fast              | Low-resource devices, quick checks        |
| `speculative_thought` | All               | Exploratory, Rapid Prototyping  | Early-stage ideation, multiple perspectives |
| `paper_optimized`   | All               | Complete Research Integration   | Maximum research benefit, benchmarking    |

### **Advanced Features**

-   **Automatic Complexity Analysis**: Real-time, multi-language problem complexity assessment.
-   **Self-Improving Analysis**: Learns from past interactions to make smarter future decisions.
-   **Dynamic Strategy Selection**: Optimal reasoning approach chosen per problem.
-   **Retrieval-Augmented Generation (RAG)**: Augments prompts with external knowledge from **Wikipedia** or local files/URLs.
-   **Overthinking & Collapse Prevention**: Core mechanisms to maintain efficiency and reasoning quality.
-   **Multi-Provider Support**: OpenAI, Claude, Gemini, Ollama, HuggingFace, Llama.cpp.
-   **Structured Thought Process**: Outputs detailed reasoning steps for full transparency.

---

## 🛠️ Installation & Quick Start

### **Requirements**

-   Python 3.10+
-   `pip` package manager
-   At least one LLM provider API key or a local Ollama/Llama.cpp setup
-   `ffmpeg` (for audio processing, optional)

### **Setup**

```bash
# Clone the repository
git clone [https://github.com/littlebuddha-dev/MetaIntelligence.git](https://github.com/littlebuddha-dev/MetaIntelligence.git)
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
# Install Ollama: curl -fsSL [https://ollama.ai/install.sh](https://ollama.ai/install.sh) | sh
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
