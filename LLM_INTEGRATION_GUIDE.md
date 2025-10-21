# LLM Integration Guide for Reachy Mini

This guide shows how to integrate Large Language Models (LLMs) with Reachy Mini using Ollama running locally.

## Prerequisites

### 1. Install and Start Ollama

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai

# Start Ollama server
ollama serve

# In another terminal, pull a model
ollama pull gemma3:27b      # Good balance of speed and quality
# or
ollama pull deepseek-r1:8b  # Faster, smaller
# or
ollama pull llava           # For vision capabilities
```

### 2. Verify Ollama is Running

```bash
curl http://localhost:11434/api/tags
```

### 3. Start Reachy Mini Daemon

```bash
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
```

## Demo Scripts

### 1. Quick Test (`llm_quick_test.py`)

**Purpose**: Verify LLM + Robot integration is working
**Interaction**: Automated (no user input needed)
**Duration**: ~2 minutes

```bash
python llm_quick_test.py
```

This script:
- Tests Ollama connection
- Sends test queries to LLM
- Demonstrates basic emotion expressions
- Shows LLM responses mapped to robot movements

### 2. Text Interaction (`llm_text_interaction.py`)

**Purpose**: Pre-scripted conversation examples with emotional analysis
**Interaction**: Automated demonstrations
**Duration**: ~3-4 minutes

```bash
python llm_text_interaction.py
```

Features:
- LLM generates responses to predefined prompts
- Second LLM call analyzes emotion in response
- Robot expresses detected emotion through movement
- Shows 6 different conversation scenarios

### 3. Interactive Chat (`llm_interactive_chat.py`)

**Purpose**: Real-time conversation with the robot
**Interaction**: Type messages in terminal
**Duration**: As long as you want!

```bash
python llm_interactive_chat.py
```

Features:
- Maintains conversation context
- Detects emotions from keywords
- Expressive responses through movement
- Type 'quit' or 'exit' to end
- Shows conversation statistics

Example conversation:
```
👤 You: Hello! How are you?
🤖 Reachy [happy]: Hi! I'm doing great, thanks for asking!

👤 You: What can you see around you?
🤖 Reachy [curious]: I have a camera, and I can see a table with some objects on it!

👤 You: That's amazing!
🤖 Reachy [excited]: Thanks! I love exploring and learning new things!
```

### 4. Vision Interaction (`llm_vision_interaction.py`)

**Purpose**: Robot uses camera + vision LLM to describe what it sees
**Requires**: Vision model (e.g., llava)
**Interaction**: Automated
**Duration**: ~5-10 minutes (vision models are slower)

```bash
# First install a vision model
ollama pull llava

# Then run the demo
python llm_vision_interaction.py
```

Features:
- Captures images from robot's camera
- Uses vision LLM to describe the scene
- Looks at scene from different angles
- Counts and identifies objects
- Answers questions about what it sees

## Architecture Overview

### Basic Flow

```
User Input → LLM (Ollama) → Response → Emotion Detection → Robot Expression
```

### Components

1. **Ollama Client**: Handles API calls to local Ollama server
2. **Emotion Detector**: Analyzes text to determine emotion and intensity
3. **Expression Engine**: Maps emotions to robot movements
4. **Robot Controller**: Executes movements via Reachy Mini SDK

### Emotion → Movement Mapping

| Emotion   | Antenna Position      | Head Movement           | Duration |
|-----------|-----------------------|-------------------------|----------|
| Happy     | Wide open (-2, 2)     | Slight tilt up          | 1.0s     |
| Sad       | Drooping (0.5, -0.5)  | Look down               | 1.5s     |
| Curious   | One up (-1.5, 0)      | Tilt + look aside       | 1.0s     |
| Excited   | Wide open (-2, 2)     | Quick wiggle (2-3x)     | 0.3s ea  |
| Thinking  | Slight (-0.5, 0.5)    | Look away + down        | 1.5s     |
| Neutral   | Center (0, 0)         | Center position         | 1.0s     |

## Code Examples

### Simple LLM Query

```python
import requests

def ask_ollama(prompt, model="gemma3:27b"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
    )
    return response.json()["response"]

answer = ask_ollama("What is a robot?")
print(answer)
```

### Express Emotion with Robot

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as robot:
    # Happy expression
    robot.set_target_antenna_joint_positions([-1.5, 1.5])
    pose = create_head_pose(pitch=10, degrees=True)
    robot.goto_target(head=pose, duration=1.0)
```

### Complete Interaction

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import requests

def chat(message, robot, model="gemma3:27b"):
    # Get LLM response
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": message, "stream": False}
    ).json()["response"]

    # Simple emotion detection
    if "!" in response or "amazing" in response.lower():
        emotion = "excited"
    elif "?" in response:
        emotion = "curious"
    else:
        emotion = "happy"

    # Express emotion
    if emotion == "excited":
        robot.set_target_antenna_joint_positions([-2.0, 2.0])
    elif emotion == "curious":
        robot.set_target_antenna_joint_positions([-1.5, 0.0])

    return response

with ReachyMini() as robot:
    response = chat("Hello robot!", robot)
    print(response)
```

## Advanced: Vision Integration

### Analyze Image with LLM

```python
import base64
import requests

def analyze_image(image_path, prompt, model="llava"):
    # Encode image
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')

    # Query vision model
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
    )
    return response.json()["response"]

# Use with robot camera
from reachy_mini import ReachyMini
import cv2

with ReachyMini() as robot:
    # Capture image
    frame = robot.media.camera.read()
    cv2.imwrite("/tmp/view.jpg", frame)

    # Analyze
    description = analyze_image("/tmp/view.jpg", "What do you see?")
    print(description)
```

## Performance Tips

### Model Selection

- **Fast response** (~1-2s): `deepseek-r1:8b`, `gemma:7b`
- **Better quality** (~3-5s): `gemma3:27b`, `llama3`
- **Vision** (~30-60s): `llava`, `llava:13b`

### Optimization

1. **Preload models**: First query is slower, subsequent ones are faster
2. **Keep prompts concise**: Shorter prompts = faster responses
3. **Limit context**: Only send last 3-6 conversation turns
4. **Parallel processing**: Generate response while robot moves

```python
import threading

def get_response_async(prompt, callback):
    """Get LLM response in background."""
    response = ask_ollama(prompt)
    callback(response)

# Robot can move while LLM is thinking
thread = threading.Thread(target=get_response_async, args=(prompt, handle_response))
thread.start()
robot.express("thinking")  # Robot shows it's thinking
thread.join()  # Wait for response
```

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Check what port it's using
lsof -i :11434
```

### Slow Responses

- Use smaller models (8b instead of 27b)
- Reduce context length
- Ensure Ollama has enough RAM
- Check CPU usage during generation

### Robot Not Moving

- Verify daemon is running: `curl http://localhost:8000/api/state/full`
- Check for errors in daemon terminal
- Ensure MuJoCo viewer is open (for simulation)

### Vision Model Issues

- Vision models are MUCH slower (30-60 seconds per query)
- Require more RAM (8GB+ recommended)
- Image quality matters - ensure good lighting
- Try smaller vision models: `llava:7b` vs `llava:13b`

## Ideas for Extension

### 1. Voice Integration
- Add speech-to-text (Whisper)
- Add text-to-speech (piper, coqui)
- Create voice-controlled robot

### 2. Multimodal Responses
- Combine vision + conversation
- Robot describes what it sees during chat
- Answer questions about environment

### 3. Personality Tuning
- Customize system prompts
- Add personality traits (curious, shy, excited)
- Create different "modes"

### 4. Learning & Memory
- Save conversations to disk
- Build long-term memory
- Learn user preferences

### 5. Task Execution
- Parse commands from LLM
- Execute robot actions
- Chain multiple movements

Example:
```python
# User: "Look around and tell me what you see"
# 1. Parse intent: look_around + describe
# 2. Execute: scan environment
# 3. Capture images from multiple angles
# 4. Analyze with vision LLM
# 5. Summarize findings
# 6. Respond expressively
```

## Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama Python Library](https://github.com/ollama/ollama-python)
- [Reachy Mini Docs](https://github.com/pollen-robotics/reachy_mini)
- [Reachy Conversation Demo](https://github.com/pollen-robotics/reachy_mini_conversation_demo)

## Next Steps

1. Run `llm_quick_test.py` to verify everything works
2. Try `llm_interactive_chat.py` for real conversations
3. Experiment with different models and prompts
4. Build your own AI-powered robot applications!
5. Share what you create! 🤖
