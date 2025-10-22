# Reachy Mini Agentic AI

**PhD Research Project: Agentic AI with Deep Reinforcement Learning & Human-in-the-Loop**

This repository contains a multi-phase project for developing an advanced agentic AI system for the Reachy Mini robot, with real-time multimodal capabilities (vision, speech, movement) and eventual self-improvement through Deep Reinforcement Learning with human feedback.

## 🎯 Project Phases

### ✅ Phase 1: Multimodal Agentic AI (COMPLETED)

**Real-time vision, speech, and natural language interaction with OpenAI Agents SDK**

- Multi-agent system (Coordinator, Robot Control, Vision)
- Ollama + LiteLLM integration (Gemma 3 27B)
- WhisperX speech-to-text (70x realtime)
- Piper TTS with streaming
- FastAPI with WebSocket for real-time communication
- 10+ robot control and vision tools

👉 **[See PHASE1_README.md for complete documentation](PHASE1_README.md)**

### 🔨 Phase 2: Self-Coding Agent (In Development)

- Autonomous tool generation
- Code testing and validation
- Safety guardrails
- Dynamic tool registration

### 🚀 Phase 3: DRL with Human-in-the-Loop (Planned)

- Natural language feedback
- Demonstration learning
- Reward model training
- PPO-based policy optimization

📋 **[See PROJECT_SPEC.md for full project specification](PROJECT_SPEC.md)**

---

## 🚀 Quick Start - Phase 1 API

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python3 -m piper.download_voices en_US-lessac-medium
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your paths and settings
```

### 3. Start the Reachy Daemon

```bash
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
```

### 4. Start the API Server

```bash
python -m src.main
```

### 5. Test the System

```bash
python test_api.py
```

**API Documentation:** http://localhost:8000/docs

---

## 📚 Early Demos (Before Phase 1)

Interactive demos for Reachy Mini robot in MuJoCo simulation, including basic LLM integration with Ollama.

## Quick Start

### 1. Start the Simulation

```bash
# Start daemon with MuJoCo simulation
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
```

### 2. Run Basic Demos

```bash
# Test basic movements
python test_simulation.py

# Antenna expressions
python demo_antennas.py

# Combined behaviors (nodding, shaking, etc.)
python demo_combined.py

# Camera feed
python demo_camera.py

# Choreographed movements
python demo_choreography.py
```

## LLM Integration (with Ollama)

### Setup

```bash
# Start Ollama (in separate terminal)
ollama serve

# Pull a model
ollama pull gemma3:27b
```

### LLM Demos

```bash
# Quick test (automated, ~2 min)
python llm_quick_test.py

# Automated conversation examples (~3-4 min)
python llm_text_interaction.py

# Interactive real-time chat
python llm_interactive_chat.py

# Vision integration (requires: ollama pull llava)
python llm_vision_interaction.py
```

### Gemma 3 Vision Demos (Multimodal)

Gemma 3 can see and understand images! Give Reachy the power of vision.

```bash
# Quick test - one photo, one description (~1-2 min)
python gemma_vision_simple.py

# Interactive - ask anything about what Reachy sees!
python gemma_vision_interactive.py

# Full demo - comprehensive showcase (~15-20 min)
python gemma_vision_demo.py
```

See **[GEMMA_VISION_README.md](GEMMA_VISION_README.md)** for detailed vision guide.

## 📖 Documentation

### Phase 1 (Current)
- **[PHASE1_README.md](PHASE1_README.md)** - Complete Phase 1 documentation
- **[PROJECT_SPEC.md](PROJECT_SPEC.md)** - Full 3-phase project specification

### Early Development Docs
- **[DEMO_GUIDE.md](DEMO_GUIDE.md)** - Complete SDK and REST API reference
- **[LLM_INTEGRATION_GUIDE.md](LLM_INTEGRATION_GUIDE.md)** - Detailed LLM integration guide
- **[GEMMA_VISION_README.md](GEMMA_VISION_README.md)** - Gemma 3 vision guide

## 📂 File Structure

```
reachy-mini/
├── README.md                      # This file
├── PHASE1_README.md               # Phase 1 complete documentation
├── PROJECT_SPEC.md                # Full 3-phase project spec
├── config.yaml                    # Main configuration
├── .env.example                   # Environment variables template
├── requirements.txt               # Python dependencies
├── test_api.py                    # Phase 1 API test script
│
├── src/                           # Phase 1 source code
│   ├── agents/                    # Agent definitions
│   │   ├── coordinator.py         # Main coordinator agent
│   │   ├── robot_agent.py         # Robot control specialist
│   │   ├── vision_agent.py        # Vision analysis specialist
│   │   └── runner.py              # Agent execution runner
│   ├── tools/                     # Function tools
│   │   ├── robot_tools.py         # Robot control tools
│   │   └── vision_tools.py        # Vision analysis tools
│   ├── api/                       # FastAPI application
│   │   ├── main.py                # API server
│   │   └── routes/                # API endpoints
│   ├── multimodal/                # Speech & audio
│   │   ├── speech_to_text.py      # WhisperX integration
│   │   └── text_to_speech.py      # Piper TTS integration
│   ├── utils/                     # Utilities
│   │   ├── config.py              # Configuration management
│   │   └── session.py             # Session management
│   └── main.py                    # Entry point
│
├── docs/                          # Technical documentation
│   ├── openai-agent-sdk.md        # OpenAI Agents SDK reference
│   ├── whisperX.md                # WhisperX documentation
│   ├── piper.md                   # Piper TTS documentation
│   └── ...                        # Other tech docs
│
└── [Early demos]                  # Initial exploration demos
    ├── test_simulation.py
    ├── demo_*.py
    ├── llm_*.py
    └── gemma_vision_*.py
```

## Features Demonstrated

### Basic Robot Control
- ✓ Head movements (pitch, yaw, roll, translation)
- ✓ Antenna expressions for emotions
- ✓ Camera feed access
- ✓ Coordinated choreography
- ✓ REST API usage

### LLM Integration
- ✓ Text-based conversation
- ✓ Emotion detection and expression
- ✓ Context-aware responses
- ✓ Vision-language integration (LLaVA)
- ✓ Real-time interaction

### Gemma 3 Vision (Multimodal)
- ✓ Scene understanding and description
- ✓ Object identification and counting
- ✓ Spatial relationship analysis
- ✓ Color and texture recognition
- ✓ Interactive visual Q&A
- ✓ Multi-angle scene analysis
- ✓ Creative interpretation

## Requirements

- Python 3.10-3.13
- reachy-mini package
- MuJoCo (for simulation)
- Ollama (for LLM features)
- requests library

## Resources

- [Reachy Mini Official Repo](https://github.com/pollen-robotics/reachy_mini)
- [Reachy Mini Website](https://www.pollen-robotics.com/reachy-mini/)
- [Ollama](https://ollama.ai)
- [MuJoCo](https://mujoco.org)

## License

These demo scripts are provided as examples for the Reachy Mini robot platform.

---

**Made with** [Claude Code](https://claude.com/claude-code) 🤖
