# Reachy Mini Agentic AI - Phase 1

> **Real-time multimodal agentic AI system for the Reachy Mini robot**

Phase 1 implements the core foundation with vision, speech, and natural language interaction using the OpenAI Agents SDK.

## 🎯 Features

### ✅ Completed

- **Multi-Agent System** using OpenAI Agents SDK
  - Coordinator agent for conversation management
  - Robot control specialist for physical movements
  - Vision analyst for scene understanding

- **Multimodal Capabilities**
  - **Vision**: Gemma 3 27B for scene analysis via Ollama
  - **Speech-to-Text**: WhisperX with 70x realtime speed, VAD preprocessing
  - **Text-to-Speech**: Piper TTS with streaming support

- **Robot Control Tools** (10+ tools)
  - Head movement (pitch, yaw, roll)
  - Antenna expressions
  - Emotion expression (happy, sad, curious, surprised, neutral)
  - Object tracking (look_at_object)

- **Vision Tools**
  - Photo capture
  - Scene analysis with custom questions
  - Object detection
  - View description

- **FastAPI Backend**
  - REST API for all robot functions
  - WebSocket for real-time bidirectional communication
  - Session management with SQLite
  - CORS enabled for web clients

## 🏗️ Architecture

```
src/
├── agents/          # Agent definitions (coordinator, robot, vision)
├── tools/           # Function tools for robot control and vision
├── api/             # FastAPI application with routes
│   └── routes/      # Chat, robot, vision, session endpoints
├── multimodal/      # WhisperX and Piper TTS integration
└── utils/           # Config management and session storage
```

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Piper Voice Model

```bash
python3 -m piper.download_voices en_US-lessac-medium
```

### 3. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:27b

# Piper Voice Path (update after downloading)
PIPER_VOICE_PATH=/path/to/en_US-lessac-medium.onnx

# WhisperX Configuration
WHISPERX_MODEL=base
WHISPERX_DEVICE=cpu  # or 'cuda'
WHISPERX_COMPUTE_TYPE=int8  # or 'float16' for GPU

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### 4. Ensure Ollama is Running

```bash
# Make sure Ollama is running with Gemma 3 27B
ollama list | grep gemma3
```

## 🚀 Usage

### Start the API Server

```bash
python -m src.main
```

The API will be available at `http://localhost:8000`

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔌 API Endpoints

### Chat Endpoints

#### POST `/api/chat/message`

Send a text message to the agent.

```bash
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "message": "What do you see in front of you?"
  }'
```

Response:
```json
{
  "status": "success",
  "response": "I can see a wooden desk with a laptop...",
  "agent": "VisionAnalyst",
  "session_id": "user123",
  "message_count": 3
}
```

#### POST `/api/chat/stream`

Stream agent response (Server-Sent Events).

```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "message": "Tell me what you see"
  }'
```

#### WebSocket `/api/chat/ws`

Real-time bidirectional communication.

```javascript
const ws = new WebSocket('ws://localhost:8000/api/chat/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'message',
    session_id: 'user123',
    message: 'Hello Reachy!'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'chunk') {
    console.log(data.content);
  }
};
```

### Robot Control Endpoints

#### POST `/api/robot/move_head`

```json
{
  "pitch": 10.0,
  "yaw": 20.0,
  "roll": 0.0,
  "duration": 2.0
}
```

#### POST `/api/robot/set_antennas`

```json
{
  "left": "up",
  "right": "up",
  "duration": 1.0
}
```

#### POST `/api/robot/look_at`

```json
{
  "direction": "left",
  "intensity": "medium"
}
```

#### POST `/api/robot/express_emotion`

```json
{
  "emotion": "happy"
}
```

#### GET `/api/robot/current_pose`

Get current head and antenna positions.

### Vision Endpoints

#### POST `/api/vision/take_photo`

```json
{
  "save_path": "./photos/image.jpg"
}
```

#### POST `/api/vision/analyze_scene`

```json
{
  "question": "Are there any people in this image?"
}
```

#### POST `/api/vision/detect_objects`

Detect and list objects in view.

#### POST `/api/vision/describe_view`

```json
{
  "focus": "colors"
}
```

### Session Endpoints

#### GET `/api/sessions/list`

List all sessions.

#### GET `/api/sessions/{session_id}/history`

Get conversation history.

#### POST `/api/sessions/clear`

```json
{
  "session_id": "user123"
}
```

#### DELETE `/api/sessions/{session_id}`

Delete a session.

## 🧪 Testing

### Quick Test

```bash
# Test chat endpoint
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "message": "Move your head to the left"
  }'
```

### Test Vision

```bash
curl -X POST http://localhost:8000/api/vision/analyze_scene \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Describe what you see"
  }'
```

### Test Robot Control

```bash
curl -X POST http://localhost:8000/api/robot/express_emotion \
  -H "Content-Type: application/json" \
  -d '{
    "emotion": "happy"
  }'
```

## 🎭 Agent System

### Coordinator Agent

Main conversational agent that:
- Understands user requests
- Delegates to specialist agents
- Maintains conversation context
- Provides natural responses

### Robot Control Agent

Specialist for physical control:
- Safe movement execution
- Emotion expression
- Object tracking
- Position reporting

### Vision Agent

Specialist for visual understanding:
- Scene analysis
- Object detection
- Visual Q&A
- Image description

## 🔧 Configuration

All configuration is in `config.yaml`:

```yaml
# LLM Configuration
llm:
  provider: ollama
  model: gemma3:27b
  temperature: 0.7

# Speech-to-Text (WhisperX)
speech_to_text:
  model: base
  batch_size: 1  # For low latency
  vad_filter: true

# Text-to-Speech (Piper)
text_to_speech:
  voice: en_US-lessac-medium
  streaming: true

# Robot Safety
robot:
  safety:
    max_head_speed: 50
    workspace_limits:
      pitch: [-45, 45]
      yaw: [-90, 90]
```

## 📊 Session Management

Sessions are stored in SQLite (`./data/sessions.db`):

- Automatic session creation
- Conversation history persistence
- Configurable history limits
- Automatic cleanup of old sessions

## 🔐 Safety Features

- Workspace limits for head movements
- Movement speed limits
- Collision checking
- Safe default positions

## 🐛 Troubleshooting

### Robot Not Found

Ensure the Reachy daemon is running:

```bash
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
```

### Ollama Connection Error

```bash
# Check Ollama is running
curl http://localhost:11434/api/version

# Check model is available
ollama list | grep gemma3
```

### WhisperX Model Not Found

```bash
# Models are downloaded automatically on first use
# Or manually download:
python -c "import whisperx; whisperx.load_model('base')"
```

### Piper Voice Not Found

```bash
# Download the voice
python3 -m piper.download_voices en_US-lessac-medium

# Update PIPER_VOICE_PATH in .env
```

## 📈 Performance Targets

- **Voice Latency**: <500ms (STT + LLM + TTS)
- **Vision Analysis**: ~2-5s (depends on scene complexity)
- **Robot Movement**: Real-time execution
- **Concurrent Sessions**: 10+ simultaneous users

## 🚦 Next Steps (Phase 2)

- Self-coding agent capabilities
- Tool generation and testing
- Safety validation system
- Dynamic tool registration

## 📝 License

This project is part of a PhD research on agentic AI with Deep Reinforcement Learning.

---

**Built with:**
- [OpenAI Agents SDK](https://github.com/openai/agents-sdk)
- [LiteLLM](https://github.com/BerriAI/litellm)
- [WhisperX](https://github.com/m-bain/whisperX)
- [Piper TTS](https://github.com/rhasspy/piper)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Reachy Mini SDK](https://docs.pollen-robotics.com/)
