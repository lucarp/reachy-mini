# Reachy Mini Simulation & LLM Integration

Interactive demos for Reachy Mini robot in MuJoCo simulation, including LLM integration with Ollama.

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

## Documentation

- **[DEMO_GUIDE.md](DEMO_GUIDE.md)** - Complete SDK and REST API reference
- **[LLM_INTEGRATION_GUIDE.md](LLM_INTEGRATION_GUIDE.md)** - Detailed LLM integration guide

## File Structure

```
.
├── README.md                      # This file
├── DEMO_GUIDE.md                  # Complete SDK reference
├── LLM_INTEGRATION_GUIDE.md       # LLM integration guide
├── GEMMA_VISION_README.md         # Gemma 3 vision guide
│
├── test_simulation.py             # Basic movement test
├── demo_antennas.py               # Antenna expressions
├── demo_combined.py               # Combined head + antenna behaviors
├── demo_camera.py                 # Camera feed demo
├── demo_choreography.py           # Advanced choreographed movements
│
├── llm_quick_test.py              # Quick LLM integration test
├── llm_text_interaction.py        # Automated LLM conversation examples
├── llm_interactive_chat.py        # Real-time interactive chat
├── llm_vision_interaction.py      # Vision + LLM integration (LLaVA)
│
├── gemma_vision_simple.py         # Quick Gemma vision test
├── gemma_vision_interactive.py    # Interactive Q&A with Gemma vision
└── gemma_vision_demo.py           # Comprehensive Gemma vision demo
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
