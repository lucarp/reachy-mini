# Reachy Mini Agentic AI: Project Specification

**Version**: 1.0
**Last Updated**: October 2025
**Author**: Lucas Rodrigues Pereira
**Project Type**: PhD Research - Agentic AI with Deep Reinforcement Learning

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Vision & Goals](#project-vision--goals)
3. [System Architecture](#system-architecture)
4. [Phase 1: Basic Agentic AI (v1)](#phase-1-basic-agentic-ai-v1)
5. [Phase 2: Self-Coding Agent (v2)](#phase-2-self-coding-agent-v2)
6. [Phase 3: DRL with Human-in-the-Loop (v3)](#phase-3-drl-with-human-in-the-loop-v3)
7. [Technical Stack](#technical-stack)
8. [API Specifications](#api-specifications)
9. [Development Roadmap](#development-roadmap)
10. [Implementation Guidelines](#implementation-guidelines)
11. [Testing Strategy](#testing-strategy)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Challenges & Mitigation](#challenges--mitigation)
14. [References & Resources](#references--resources)

---

## Executive Summary

This project aims to develop an advanced agentic AI system for the Reachy Mini robot, capable of multimodal perception (vision, speech), autonomous tool creation, and continuous improvement through Deep Reinforcement Learning with Human-in-the-Loop feedback.

### Key Objectives

1. **Phase 1**: Build a real-time multimodal agent using OpenAI Agents SDK
2. **Phase 2**: Enable self-coding capabilities for autonomous tool generation
3. **Phase 3**: Implement DRL with human feedback for continuous improvement

### Innovation

- First implementation combining OpenAI Agents SDK with embodied robotics
- Novel self-coding framework with safety-first tool validation
- Human-in-the-Loop DRL using natural language feedback and demonstrations
- Real-time multimodal interaction (<500ms latency) on consumer hardware

---

## Project Vision & Goals

### Vision Statement

Create an autonomous, continuously learning robotic agent that can:
- Perceive and understand its environment through vision and speech
- Execute complex tasks through tool use and multi-agent coordination
- Extend its own capabilities by generating and testing new tools
- Improve through human feedback in natural language and demonstrations

### Research Questions

1. Can an LLM-based agent effectively control embodied robotics through tool composition?
2. How can we safely enable autonomous tool generation in production systems?
3. What is the most effective way to convert natural language feedback into reward signals for DRL?
4. Can human demonstrations be efficiently integrated into policy learning?
5. What are the optimal tradeoffs between real-time performance and model capability?

### Success Criteria

**Phase 1 Success**:
- [ ] Real-time voice interaction (<500ms latency)
- [ ] Vision integration with object detection and scene understanding
- [ ] Successful execution of 10+ predefined tools for robot control
- [ ] Remote API access with <100ms response time
- [ ] 90%+ success rate on basic task completion

**Phase 2 Success**:
- [ ] Agent generates valid Python tools 80%+ of the time
- [ ] Self-generated tools pass safety validation 95%+ of the time
- [ ] Successfully incorporates 20+ self-generated tools
- [ ] Zero critical safety failures in tool execution

**Phase 3 Success**:
- [ ] Natural language feedback → reward conversion accuracy >85%
- [ ] Policy improvement demonstrated over 100+ episodes
- [ ] Human satisfaction score >4.0/5.0 after training
- [ ] Agent generalizes to novel tasks not in training set

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  Voice Input  │  Web Interface  │  Mobile App  │  CLI          │
│  (Whisper)    │  (React/Next)   │  (Flutter)   │  (Python)     │
└────────┬──────────────┬──────────────┬──────────────┬───────────┘
         │              │              │              │
         └──────────────┴──────────────┴──────────────┘
                         │
                    ┌────▼────┐
                    │ FastAPI │
                    │   API   │
                    └────┬────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │ Session │    │ Tracing │    │  Auth   │
    │ Manager │    │  Layer  │    │ & Auth  │
    └────┬────┘    └────┬────┘    └─────────┘
         │              │
         └──────┬───────┘
                │
         ┌──────▼──────────────────────────────────────────┐
         │        OpenAI Agents SDK Core                   │
         ├─────────────────────────────────────────────────┤
         │                                                  │
         │  ┌──────────────┐  ┌──────────────┐            │
         │  │ Main Agent   │  │ Specialist   │            │
         │  │ (Coordinator)│◄─┤ Agents       │            │
         │  └──────┬───────┘  └──────────────┘            │
         │         │                                       │
         │  ┌──────▼────────────────────────┐             │
         │  │      Tool Registry            │             │
         │  ├───────────────────────────────┤             │
         │  │ • Robot Control Tools         │             │
         │  │ • Vision Tools                │             │
         │  │ • Web Search Tools            │             │
         │  │ • Self-Generated Tools [v2]   │             │
         │  └──────┬────────────────────────┘             │
         │         │                                       │
         └─────────┼───────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    ┌────▼────┐         ┌───▼────┐
    │ Ollama  │         │ Reachy │
    │  LLM    │         │  Mini  │
    │ Backend │         │ Daemon │
    └────┬────┘         └───┬────┘
         │                  │
    ┌────▼────┐         ┌───▼────┐
    │ Gemma3  │         │ MuJoCo │
    │  27B    │         │  Sim   │
    └─────────┘         └────────┘

         [Phase 3: DRL Layer]

    ┌────────────────────────────────────┐
    │      DRL Training Pipeline         │
    ├────────────────────────────────────┤
    │ • Experience Buffer                │
    │ • Reward Model (NL → Reward)       │
    │ • Policy Network                   │
    │ • Demonstration Database           │
    │ • Human Feedback Interface         │
    └────────────────────────────────────┘
```

### Component Breakdown

#### 1. **API Layer (FastAPI)**
- **Purpose**: RESTful API for remote interaction
- **Endpoints**:
  - `/chat` - Text-based conversation
  - `/voice` - Voice input/output
  - `/vision` - Image analysis requests
  - `/tools` - Tool management and execution
  - `/session` - Session management
  - `/feedback` - Human feedback collection [Phase 3]
- **Features**:
  - WebSocket support for real-time streaming
  - Authentication and rate limiting
  - Request/response logging
  - CORS for web clients

#### 2. **OpenAI Agents SDK Core**
- **Agents**:
  - `MainAgent`: Orchestrator, handles high-level planning
  - `RobotControlAgent`: Specialist for robot movement and manipulation
  - `VisionAgent`: Specialist for visual perception and analysis
  - `WebSearchAgent`: Internet search and information retrieval
  - `CodeAgent`: Tool generation and testing [Phase 2]
- **Handoffs**: Agent delegation based on task requirements
- **Guardrails**: Input/output validation, safety checks
- **Sessions**: Conversation history and context management

#### 3. **Tool System**
- **Predefined Tools** (Phase 1):
  - `move_head(pitch, yaw, roll)` - Head positioning
  - `set_antennas(left, right)` - Antenna control
  - `take_photo()` - Capture image
  - `analyze_scene(prompt)` - Vision analysis
  - `speak(text)` - Text-to-speech
  - `search_web(query)` - Web search
  - `get_robot_state()` - Current state query
  - `express_emotion(emotion)` - Emotional expression
- **Generated Tools** (Phase 2):
  - Dynamic tool creation
  - Validation and testing
  - Version control
  - Activation/deactivation

#### 4. **Multimodal Interface**
- **Vision Pipeline**:
  - Camera capture from Reachy
  - Image preprocessing
  - Gemma 3 multimodal analysis
  - Object detection/tracking
  - Scene understanding
- **Speech Pipeline**:
  - **Input**: Whisper (tiny/base for <500ms)
  - **Output**: Piper TTS or similar fast engine
  - **VAD**: Voice Activity Detection for turn-taking
  - **Streaming**: Real-time audio processing

#### 5. **LLM Backend (Ollama)**
- **Model**: Gemma3:27b (or configurable)
- **API**: OpenAI-compatible interface
- **Features**:
  - Function calling support
  - Streaming responses
  - Context caching
  - Multi-turn conversations

#### 6. **DRL Training System** (Phase 3)
- **Components**:
  - Experience replay buffer
  - Reward model (NL feedback → scalar reward)
  - Policy network (PPO/SAC)
  - Demonstration database
  - Value function approximator
- **Training Loop**:
  - Agent interaction → feedback → reward → policy update
  - Periodic evaluation episodes
  - Checkpoint management

---

## Phase 1: Basic Agentic AI (v1)

### Objectives

Build a functional multimodal agent that can:
1. See and understand its environment using Gemma 3 vision
2. Listen and speak with <500ms latency
3. Execute tasks using predefined tools
4. Be controlled remotely via API
5. Coordinate multiple specialist agents

### Architecture Details

#### OpenAI Agents SDK Integration

```python
# Conceptual structure

from openai_agents import Agent, Handoff, Session, GuardRail

# Define main coordinator agent
main_agent = Agent(
    name="ReachyCoordinator",
    instructions="""You are Reachy, an intelligent robot assistant.
    You can see, hear, and control your robotic body.
    Delegate to specialist agents for specific tasks.
    Always explain what you're doing and why.""",
    tools=[
        get_robot_state,
        express_emotion,
        web_search,
    ],
    handoffs=[
        Handoff(target="RobotControlAgent", condition="robot movement needed"),
        Handoff(target="VisionAgent", condition="visual analysis needed"),
    ]
)

# Specialist: Robot Control
robot_agent = Agent(
    name="RobotControlAgent",
    instructions="""Control Reachy's physical movements.
    Plan safe, smooth motions. Validate positions before execution.
    Return to coordinator when task is complete.""",
    tools=[
        move_head,
        set_antennas,
        goto_pose,
    ]
)

# Specialist: Vision
vision_agent = Agent(
    name="VisionAgent",
    instructions="""Analyze visual information using Gemma 3.
    Take photos, describe scenes, identify objects, answer visual questions.
    Provide detailed observations back to coordinator.""",
    tools=[
        take_photo,
        analyze_scene,
        detect_objects,
        describe_image,
    ]
)

# Session management
session = Session(
    agents=[main_agent, robot_agent, vision_agent],
    guardrails=[
        GuardRail(validate_robot_safety),
        GuardRail(validate_llm_output),
    ]
)
```

#### Real-Time Voice Pipeline

```python
# Voice Input (Whisper)
class VoiceInputProcessor:
    def __init__(self):
        self.model = whisper.load_model("base")  # Fast, decent quality
        self.vad = VADDetector()  # Voice Activity Detection

    async def stream_transcription(self, audio_stream):
        """Real-time transcription with <500ms latency"""
        async for chunk in audio_stream:
            if self.vad.is_speech(chunk):
                text = self.model.transcribe(chunk, fp16=True)
                yield text["text"]

# Voice Output (Piper TTS)
class VoiceOutputProcessor:
    def __init__(self):
        self.tts = PiperTTS(model="en_US-lessac-medium")  # Fast synthesis

    async def synthesize_stream(self, text):
        """Stream audio generation for low latency"""
        async for audio_chunk in self.tts.generate_stream(text):
            yield audio_chunk
```

#### Tool Implementation Example

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# Tool: Move head
async def move_head(pitch: float, yaw: float, roll: float, duration: float = 2.0):
    """
    Move Reachy's head to specified orientation.

    Args:
        pitch: Pitch angle in degrees (-45 to 45)
        yaw: Yaw angle in degrees (-45 to 45)
        roll: Roll angle in degrees (-30 to 30)
        duration: Movement duration in seconds

    Returns:
        dict: Status and final position
    """
    # Validation
    if not (-45 <= pitch <= 45):
        return {"error": "Pitch out of safe range"}
    if not (-45 <= yaw <= 45):
        return {"error": "Yaw out of safe range"}
    if not (-30 <= roll <= 30):
        return {"error": "Roll out of safe range"}

    # Execute
    with ReachyMini() as reachy:
        pose = create_head_pose(pitch=pitch, yaw=yaw, roll=roll, degrees=True)
        reachy.goto_target(head=pose, duration=duration)

        # Verify
        final_pose = reachy.get_current_head_pose()

    return {
        "status": "success",
        "final_position": {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll
        }
    }

# Register tool with agent
main_agent.register_tool(move_head)
```

#### FastAPI Implementation

```python
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

app = FastAPI(title="Reachy Mini Agentic API")

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    agent_used: str
    tools_called: list[str]

# Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Text-based chat with the agent"""
    session = session_manager.get_or_create(request.session_id)

    result = await session.run(request.message)

    return ChatResponse(
        response=result.message,
        session_id=session.id,
        agent_used=result.agent_name,
        tools_called=result.tools_used
    )

@app.websocket("/ws/voice")
async def voice_interaction(websocket: WebSocket):
    """Real-time voice interaction"""
    await websocket.accept()

    voice_in = VoiceInputProcessor()
    voice_out = VoiceOutputProcessor()

    async for audio_chunk in websocket.iter_bytes():
        # Transcribe
        text = await voice_in.transcribe(audio_chunk)

        # Process with agent
        session = session_manager.get(websocket.session_id)
        response = await session.run(text)

        # Synthesize and send
        async for audio_out in voice_out.synthesize_stream(response.message):
            await websocket.send_bytes(audio_out)

@app.post("/tools/execute")
async def execute_tool(tool_name: str, params: dict):
    """Execute a specific tool directly"""
    if tool_name not in tool_registry:
        raise HTTPException(404, f"Tool {tool_name} not found")

    result = await tool_registry[tool_name](**params)
    return result

@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history"""
    session = session_manager.get(session_id)
    return session.get_history()
```

### Phase 1 Implementation Checklist

**Week 1-2: Foundation**
- [ ] Set up FastAPI project structure
- [ ] Install and configure Ollama with Gemma3
- [ ] Implement basic tool system
- [ ] Create session management
- [ ] Set up OpenAI Agents SDK

**Week 3-4: Core Agent**
- [ ] Implement main coordinator agent
- [ ] Create robot control specialist agent
- [ ] Create vision specialist agent
- [ ] Implement handoff logic
- [ ] Add guardrails for safety

**Week 5-6: Multimodal Interface**
- [ ] Integrate Whisper for STT
- [ ] Integrate Piper for TTS
- [ ] Implement VAD for turn-taking
- [ ] Optimize for <500ms latency
- [ ] Test real-time performance

**Week 7-8: Integration & Testing**
- [ ] Connect all components
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Documentation
- [ ] Demo preparation

---

## Phase 2: Self-Coding Agent (v2)

### Objectives

Enable the agent to:
1. Generate new Python tools to extend capabilities
2. Test generated tools in sandboxed environment
3. Validate safety and functionality
4. Incorporate successful tools into tool registry
5. Version and manage self-generated tools

### Architecture Details

#### Tool Generation Pipeline

```
User Request
    │
    ▼
[MainAgent identifies capability gap]
    │
    ▼
[Handoff to CodeAgent]
    │
    ▼
[CodeAgent generates tool code]
    │
    ▼
[Static Analysis & Safety Check]
    │
    ├─[FAIL]──► Log & Report to User
    │
    ▼[PASS]
[Sandbox Testing]
    │
    ├─[FAIL]──► Log & Suggest Fix
    │
    ▼[PASS]
[Human Review (optional)]
    │
    ▼
[Add to Tool Registry]
    │
    ▼
[Tool Available for Use]
```

#### Code Agent Implementation

```python
code_agent = Agent(
    name="CodeAgent",
    instructions="""You are a code generation specialist.

    Your job is to create new tools for the robot when needed.

    RULES:
    1. Generate only pure Python functions
    2. Use type hints for all parameters
    3. Include comprehensive docstrings
    4. Add input validation
    5. Handle errors gracefully
    6. Never use exec(), eval(), or __import__()
    7. Only import from whitelist: [reachy_mini, numpy, requests, cv2]
    8. Return structured dict with status and data

    TEMPLATE:
    ```python
    async def tool_name(param1: type1, param2: type2) -> dict:
        \"\"\"
        Brief description.

        Args:
            param1: Description
            param2: Description

        Returns:
            dict: {"status": "success/error", "data": ...}
        \"\"\"
        # Validation
        if validation_fails:
            return {"status": "error", "message": "..."}

        # Implementation
        try:
            result = do_something()
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    ```

    Generate code, explain what it does, and why it's safe.
    """,
    tools=[
        generate_tool_code,
        explain_code,
        suggest_improvements,
    ]
)
```

#### Safety Validation System

```python
class ToolValidator:
    """Validates generated tools for safety and correctness"""

    # Whitelist of allowed imports
    ALLOWED_IMPORTS = {
        'reachy_mini', 'reachy_mini.utils',
        'numpy', 'np',
        'requests',
        'cv2',
        'asyncio',
        'time',
        'math',
        'typing',
    }

    # Blacklist of dangerous operations
    FORBIDDEN_PATTERNS = [
        r'\bexec\b', r'\beval\b', r'\b__import__\b',
        r'\bsubprocess\b', r'\bos\.system\b',
        r'\bshutil\.rmtree\b', r'\bopen\(.+[\'"]w',
        r'\bsocket\b', r'\bpickle\.loads\b',
    ]

    def validate_code(self, code: str) -> tuple[bool, str]:
        """
        Validate generated code for safety.

        Returns:
            (is_valid, error_message)
        """
        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.ALLOWED_IMPORTS:
                        return False, f"Forbidden import: {alias.name}"

            if isinstance(node, ast.ImportFrom):
                if node.module not in self.ALLOWED_IMPORTS:
                    return False, f"Forbidden import: {node.module}"

        # Check for forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code):
                return False, f"Forbidden operation detected: {pattern}"

        # Check function signature
        func_def = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break

        if not func_def:
            return False, "No function definition found"

        # Validate type hints
        if not func_def.returns:
            return False, "Function must have return type annotation"

        # All checks passed
        return True, "Validation successful"
```

#### Sandbox Testing Environment

```python
class ToolTester:
    """Test generated tools in isolated environment"""

    def __init__(self):
        self.test_timeout = 10  # seconds
        self.max_memory = 512 * 1024 * 1024  # 512MB

    async def test_tool(self, code: str, test_cases: list[dict]) -> dict:
        """
        Test tool with provided test cases.

        Args:
            code: Generated tool code
            test_cases: List of {"input": {...}, "expected": {...}}

        Returns:
            {"passed": bool, "results": [...], "errors": [...]}
        """
        # Create isolated namespace
        namespace = {
            'ReachyMini': MockReachyMini,  # Mock for testing
            'numpy': numpy,
            'cv2': cv2,
            'requests': MockRequests,  # Controlled HTTP
        }

        # Execute code in namespace
        try:
            exec(code, namespace)
        except Exception as e:
            return {
                "passed": False,
                "errors": [f"Execution error: {e}"]
            }

        # Find the function
        func_name = self._extract_function_name(code)
        if func_name not in namespace:
            return {
                "passed": False,
                "errors": [f"Function {func_name} not found"]
            }

        func = namespace[func_name]

        # Run test cases
        results = []
        errors = []

        for i, test in enumerate(test_cases):
            try:
                # Run with timeout and memory limit
                result = await asyncio.wait_for(
                    func(**test['input']),
                    timeout=self.test_timeout
                )

                # Check result
                if self._matches_expected(result, test.get('expected')):
                    results.append({
                        "test": i,
                        "status": "pass",
                        "result": result
                    })
                else:
                    results.append({
                        "test": i,
                        "status": "fail",
                        "expected": test['expected'],
                        "got": result
                    })

            except asyncio.TimeoutError:
                errors.append(f"Test {i}: Timeout after {self.test_timeout}s")
            except Exception as e:
                errors.append(f"Test {i}: {type(e).__name__}: {e}")

        passed = len(errors) == 0 and all(r['status'] == 'pass' for r in results)

        return {
            "passed": passed,
            "results": results,
            "errors": errors
        }
```

#### Tool Registry and Versioning

```python
class ToolRegistry:
    """Manage predefined and generated tools"""

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.tools = {}
        self.versions = {}

    def register_tool(self,
                     name: str,
                     code: str,
                     metadata: dict,
                     source: str = "generated") -> bool:
        """
        Register a new tool or version.

        Args:
            name: Tool name
            code: Tool code
            metadata: Author, description, tests, etc.
            source: "predefined" or "generated"

        Returns:
            bool: Success
        """
        # Version management
        if name in self.tools:
            version = len(self.versions.get(name, [])) + 1
        else:
            version = 1

        # Save to disk
        tool_file = self.storage_path / source / name / f"v{version}.py"
        tool_file.parent.mkdir(parents=True, exist_ok=True)
        tool_file.write_text(code)

        # Save metadata
        meta_file = tool_file.with_suffix('.json')
        meta_file.write_text(json.dumps(metadata, indent=2))

        # Load into registry
        namespace = {}
        exec(code, namespace)
        func = namespace[name]

        self.tools[name] = func
        self.versions.setdefault(name, []).append(version)

        return True

    def rollback_tool(self, name: str, version: int):
        """Rollback to previous version"""
        tool_file = self.storage_path / "generated" / name / f"v{version}.py"
        if not tool_file.exists():
            raise ValueError(f"Version {version} not found for {name}")

        code = tool_file.read_text()
        namespace = {}
        exec(code, namespace)

        self.tools[name] = namespace[name]
```

### Self-Coding Workflow Example

```
User: "I need you to take photos every 5 seconds for 1 minute"

MainAgent: "I don't have a tool for periodic photo capture. Let me create one."
    │
    ▼
[Handoff to CodeAgent]

CodeAgent generates:
```python
async def periodic_photo_capture(interval: int, duration: int) -> dict:
    """
    Capture photos at regular intervals.

    Args:
        interval: Seconds between photos
        duration: Total duration in seconds

    Returns:
        dict: {"status": "success", "photos": [...], "count": int}
    """
    if interval <= 0 or duration <= 0:
        return {"status": "error", "message": "Invalid parameters"}

    from reachy_mini import ReachyMini
    import time
    import cv2

    photos = []
    start_time = time.time()

    with ReachyMini() as reachy:
        while time.time() - start_time < duration:
            frame = reachy.media.camera.read()
            if frame is not None:
                filename = f"/tmp/photo_{len(photos)}.jpg"
                cv2.imwrite(filename, frame)
                photos.append(filename)

            time.sleep(interval)

    return {
        "status": "success",
        "photos": photos,
        "count": len(photos)
    }
```

[Validator checks: ✓ Safe imports, ✓ No forbidden ops, ✓ Type hints]

[Tester runs with test cases:
  - Test 1: interval=1, duration=5 → Expect ~5 photos
  - Test 2: interval=0, duration=10 → Expect error
  - Test 3: interval=2, duration=6 → Expect ~3 photos
]

[All tests pass ✓]

[Tool added to registry as "periodic_photo_capture" v1]

MainAgent: "I've created a new tool! Now executing your request..."

[Uses periodic_photo_capture(5, 60)]

MainAgent: "Done! I captured 12 photos over 60 seconds."
```

### Phase 2 Implementation Checklist

**Week 1-2: Code Generation**
- [ ] Implement CodeAgent with templates
- [ ] Create code generation prompts
- [ ] Build code extraction and parsing
- [ ] Test code generation quality

**Week 3-4: Validation & Safety**
- [ ] Implement AST-based validator
- [ ] Create import whitelist/blacklist
- [ ] Add pattern matching for dangerous ops
- [ ] Build resource limit enforcement

**Week 5-6: Testing System**
- [ ] Create sandboxed execution environment
- [ ] Implement mock objects for testing
- [ ] Build test case generator
- [ ] Add timeout and memory limits

**Week 7-8: Integration**
- [ ] Create tool registry with versioning
- [ ] Implement rollback mechanism
- [ ] Add human review interface
- [ ] End-to-end testing
- [ ] Documentation

---

## Phase 3: DRL with Human-in-the-Loop (v3)

### Objectives

Enable continuous improvement through:
1. Natural language feedback → reward signal conversion
2. Human demonstrations for imitation learning
3. Policy training using DRL (PPO or SAC)
4. Evaluation and benchmarking
5. Safe exploration with human oversight

### Theoretical Foundation

#### Reinforcement Learning Formulation

- **State Space (S)**:
  - Robot state: joint positions, velocities, camera image, audio buffer
  - Task context: current goal, conversation history, tool availability
  - Environment: detected objects, scene description

- **Action Space (A)**:
  - Tool selection and parameters
  - Natural language responses
  - Agent handoffs

- **Reward Function (R)**:
  - Natural language feedback → scalar reward (via reward model)
  - Task completion indicators
  - Safety violations (negative reward)
  - Efficiency metrics (time, resource usage)

- **Policy (π)**:
  - Parameterized by LLM fine-tuning
  - Maps states to action distributions
  - Learned through PPO or SAC

#### Human-in-the-Loop Mechanisms

**1. Natural Language Feedback → Reward**

```
Human provides feedback during or after task:
  "That was perfect!" → +1.0
  "Good, but a bit slow" → +0.5
  "That's not quite right" → -0.3
  "No! Stop!" → -1.0

Reward Model:
  - Fine-tuned LLM (smaller model, e.g., Gemma 4B)
  - Input: (state, action, outcome, human_feedback)
  - Output: Scalar reward in [-1, 1]

Training:
  - Collect (feedback, reward) pairs from humans
  - Fine-tune model to predict reward from feedback
  - Use in RL training loop
```

**2. Demonstration Collection**

```
Human demonstrates correct behavior:
  1. Human takes control (via interface)
  2. System records: (state, action, outcome)
  3. Store in demonstration database

Use in training:
  - Behavioral cloning (pre-training)
  - GAIL (Generative Adversarial Imitation Learning)
  - DAgger (Dataset Aggregation)
```

### Architecture Details

#### DRL Training Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                   Training Loop                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Agent Interaction                                    │
│     ├─ Execute policy π_θ(a|s)                          │
│     ├─ Collect trajectory τ = (s₀, a₀, s₁, a₁, ...)    │
│     └─ Store in experience buffer                       │
│                                                          │
│  2. Human Feedback (async)                               │
│     ├─ Human observes agent behavior                     │
│     ├─ Provides NL feedback or demonstration            │
│     └─ Reward model converts to scalar reward           │
│                                                          │
│  3. Reward Assignment                                    │
│     ├─ r_task: Task completion reward                   │
│     ├─ r_human: From reward model                       │
│     ├─ r_safety: Safety violations (negative)           │
│     └─ r_total = α*r_task + β*r_human + γ*r_safety     │
│                                                          │
│  4. Policy Update (PPO)                                  │
│     ├─ Compute advantages: Â = Q(s,a) - V(s)            │
│     ├─ Update policy: θ ← θ + ∇_θ L_CLIP                │
│     └─ Update value function: V_φ(s)                    │
│                                                          │
│  5. Evaluation                                           │
│     ├─ Test on held-out tasks                           │
│     ├─ Measure success rate, efficiency                 │
│     └─ Checkpoint if improved                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### Implementation: Reward Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class RewardModel:
    """Convert natural language feedback to reward signals"""

    def __init__(self, model_name: str = "google/gemma-4b"):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1  # Regression for reward prediction
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode_interaction(self,
                          state: dict,
                          action: dict,
                          outcome: dict,
                          feedback: str) -> str:
        """
        Encode interaction for reward model.

        Format:
        State: {robot position, scene description, task goal}
        Action: {tool used, parameters}
        Outcome: {result, time taken, errors}
        Feedback: "{human feedback text}"
        """
        prompt = f"""
        Task: {state['task_description']}
        Robot State: {state['robot_state']}
        Scene: {state['scene_description']}

        Action Taken: {action['tool']}({action['params']})
        Outcome: {outcome['result']}
        Time: {outcome['time']}s

        Human Feedback: "{feedback}"

        Based on this feedback, rate the agent's performance:
        """
        return prompt

    def predict_reward(self,
                      state: dict,
                      action: dict,
                      outcome: dict,
                      feedback: str) -> float:
        """
        Predict reward from feedback.

        Returns:
            float: Reward in [-1, 1]
        """
        prompt = self.encode_interaction(state, action, outcome, feedback)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            reward = torch.tanh(outputs.logits[0, 0])  # Squash to [-1, 1]

        return reward.item()

    def train_from_examples(self, examples: list[dict]):
        """
        Fine-tune reward model from human feedback examples.

        Examples format:
        [
            {
                "state": {...},
                "action": {...},
                "outcome": {...},
                "feedback": "That was great!",
                "reward": 0.9  # Human-assigned ground truth
            },
            ...
        ]
        """
        # Prepare dataset
        prompts = []
        rewards = []

        for ex in examples:
            prompt = self.encode_interaction(
                ex['state'], ex['action'], ex['outcome'], ex['feedback']
            )
            prompts.append(prompt)
            rewards.append(ex['reward'])

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        labels = torch.tensor(rewards).unsqueeze(1)

        # Training loop
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        self.model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = torch.nn.functional.mse_loss(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

#### Implementation: Policy Training (PPO)

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical
from collections import deque
import numpy as np

class PolicyNetwork(nn.Module):
    """Policy network for action selection"""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

class ValueNetwork(nn.Module):
    """Value function approximator"""

    def __init__(self, state_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.network(state)

class PPOTrainer:
    """Proximal Policy Optimization trainer"""

    def __init__(self,
                 policy_net: PolicyNetwork,
                 value_net: ValueNetwork,
                 reward_model: RewardModel,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2):
        self.policy_net = policy_net
        self.value_net = value_net
        self.reward_model = reward_model

        self.policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon  # PPO clip parameter

        self.experience_buffer = deque(maxlen=10000)

    def collect_trajectory(self, env, agent, max_steps: int = 100):
        """
        Collect trajectory from agent-environment interaction.

        Returns:
            list of (state, action, reward, next_state, done)
        """
        trajectory = []
        state = env.reset()

        for step in range(max_steps):
            # Get action from current policy
            state_tensor = torch.FloatTensor(state)
            action_probs = self.policy_net(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()

            # Execute action
            next_state, task_reward, done, info = env.step(action.item())

            # Get human feedback (if available)
            human_feedback = env.get_human_feedback()  # May be None

            # Compute reward
            if human_feedback:
                human_reward = self.reward_model.predict_reward(
                    state=info['state_dict'],
                    action=info['action_dict'],
                    outcome=info['outcome'],
                    feedback=human_feedback
                )
            else:
                human_reward = 0.0

            total_reward = task_reward + 0.5 * human_reward  # Weight combination

            trajectory.append({
                'state': state,
                'action': action.item(),
                'action_prob': action_probs[action].item(),
                'reward': total_reward,
                'next_state': next_state,
                'done': done,
            })

            if done:
                break

            state = next_state

        return trajectory

    def compute_advantages(self, trajectory):
        """Compute GAE (Generalized Advantage Estimation)"""
        rewards = [t['reward'] for t in trajectory]
        states = [t['state'] for t in trajectory]

        # Compute values
        values = []
        for state in states:
            state_tensor = torch.FloatTensor(state)
            value = self.value_net(state_tensor)
            values.append(value.item())

        # Compute advantages using GAE
        advantages = []
        gae = 0

        for t in reversed(range(len(trajectory))):
            if t == len(trajectory) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * 0.95 * gae  # 0.95 is GAE lambda
            advantages.insert(0, gae)

        return advantages, values

    def update_policy(self, trajectories, epochs: int = 10):
        """Update policy using PPO"""
        # Prepare batch
        states = []
        actions = []
        old_action_probs = []
        advantages = []
        returns = []

        for traj in trajectories:
            adv, vals = self.compute_advantages(traj)

            for t, (step, advantage) in enumerate(zip(traj, adv)):
                states.append(step['state'])
                actions.append(step['action'])
                old_action_probs.append(step['action_prob'])
                advantages.append(advantage)

                # Compute return
                ret = sum(self.gamma ** i * traj[t + i]['reward']
                         for i in range(len(traj) - t))
                returns.append(ret)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_action_probs = torch.FloatTensor(old_action_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for epoch in range(epochs):
            # Get current action probabilities
            action_probs = self.policy_net(states)
            dist = Categorical(action_probs)
            new_action_probs = dist.probs.gather(1, actions.unsqueeze(1)).squeeze()

            # Compute ratio
            ratio = new_action_probs / (old_action_probs + 1e-8)

            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            # Update value function
            values = self.value_net(states).squeeze()
            value_loss = nn.functional.mse_loss(values, returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }
```

#### Human Feedback Interface

```python
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

class FeedbackRequest(BaseModel):
    episode_id: str
    timestamp: float
    feedback_text: str
    rating: float  # Optional explicit rating -1 to 1

class DemonstrationRequest(BaseModel):
    task_description: str
    actions: list[dict]  # Sequence of actions

# FastAPI endpoints
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit natural language feedback for current episode"""
    # Store feedback
    feedback_db.add({
        'episode_id': request.episode_id,
        'timestamp': request.timestamp,
        'feedback': request.feedback_text,
        'rating': request.rating
    })

    # If reward model is trained, compute reward
    if reward_model.is_trained():
        reward = reward_model.predict_reward(
            state=current_state,
            action=last_action,
            outcome=last_outcome,
            feedback=request.feedback_text
        )

        # Update experience buffer with reward
        experience_buffer.update_reward(request.episode_id, reward)

    return {"status": "received", "reward": reward if reward_model.is_trained() else None}

@app.post("/demonstration")
async def submit_demonstration(request: DemonstrationRequest):
    """Submit expert demonstration"""
    demonstration_db.add({
        'task': request.task_description,
        'actions': request.actions,
        'timestamp': time.time()
    })

    return {"status": "stored", "demo_id": demo_id}

@app.websocket("/ws/live_feedback")
async def live_feedback(websocket: WebSocket):
    """Real-time feedback during agent execution"""
    await websocket.accept()

    async for message in websocket.iter_text():
        feedback = json.loads(message)

        # Process feedback immediately
        if feedback['type'] == 'stop':
            # Emergency stop
            agent.stop()
            reward = -1.0
        elif feedback['type'] == 'good':
            reward = 0.5
        elif feedback['type'] == 'great':
            reward = 1.0
        elif feedback['type'] == 'bad':
            reward = -0.5

        # Update experience
        experience_buffer.add_reward(reward)

        await websocket.send_json({"status": "processed", "reward": reward})
```

#### Training Loop Integration

```python
async def training_loop(num_episodes: int = 1000):
    """Main DRL training loop with human-in-the-loop"""

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode} ===")

        # 1. Collect trajectory
        trajectory = ppo_trainer.collect_trajectory(
            env=reachy_env,
            agent=main_agent,
            max_steps=100
        )

        # 2. Wait for human feedback (optional, with timeout)
        print("Waiting for human feedback (30s timeout)...")
        feedback = await asyncio.wait_for(
            feedback_queue.get(),
            timeout=30.0
        )

        if feedback:
            # Convert feedback to rewards
            for step in trajectory:
                reward = reward_model.predict_reward(
                    state=step['state'],
                    action=step['action'],
                    outcome=step['outcome'],
                    feedback=feedback['text']
                )
                step['reward'] += 0.5 * reward  # Add to existing reward

        # 3. Store trajectory
        ppo_trainer.experience_buffer.append(trajectory)

        # 4. Update policy (every 10 episodes)
        if episode % 10 == 0 and len(ppo_trainer.experience_buffer) >= 10:
            print("Updating policy...")
            losses = ppo_trainer.update_policy(
                list(ppo_trainer.experience_buffer),
                epochs=10
            )
            print(f"Policy loss: {losses['policy_loss']:.4f}")
            print(f"Value loss: {losses['value_loss']:.4f}")

        # 5. Evaluate (every 50 episodes)
        if episode % 50 == 0:
            print("Evaluating...")
            eval_metrics = evaluate_policy(
                policy_net=ppo_trainer.policy_net,
                env=reachy_env,
                num_episodes=10
            )
            print(f"Success rate: {eval_metrics['success_rate']:.2%}")
            print(f"Avg reward: {eval_metrics['avg_reward']:.3f}")

            # Save checkpoint if improved
            if eval_metrics['success_rate'] > best_success_rate:
                best_success_rate = eval_metrics['success_rate']
                torch.save({
                    'policy_net': ppo_trainer.policy_net.state_dict(),
                    'value_net': ppo_trainer.value_net.state_dict(),
                    'episode': episode,
                    'metrics': eval_metrics,
                }, f"checkpoints/best_policy_ep{episode}.pt")
```

### Phase 3 Implementation Checklist

**Week 1-2: Reward Model**
- [ ] Collect initial feedback dataset
- [ ] Train reward model baseline
- [ ] Implement feedback interface
- [ ] Test reward prediction accuracy

**Week 3-4: Demonstration System**
- [ ] Build demonstration recording interface
- [ ] Implement behavioral cloning baseline
- [ ] Create demonstration database
- [ ] Test imitation performance

**Week 5-6: Policy Training**
- [ ] Implement PPO trainer
- [ ] Define state/action spaces
- [ ] Create training environment wrapper
- [ ] Test policy updates

**Week 7-8: Integration & Evaluation**
- [ ] Connect all components
- [ ] Run end-to-end training
- [ ] Evaluate on benchmark tasks
- [ ] Analysis and documentation

---

## Technical Stack

### Core Components

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Agent Framework** | OpenAI Agents SDK | Latest | Multi-agent coordination |
| **LLM Backend** | Ollama | Latest | Local inference |
| **LLM Model** | Gemma 3 27B | Latest | Reasoning and generation |
| **Vision Model** | Gemma 3 27B | Latest | Multimodal understanding |
| **API Framework** | FastAPI | 0.104+ | REST API and WebSockets |
| **Speech-to-Text** | Whisper | Tiny/Base | <500ms transcription |
| **Text-to-Speech** | Piper TTS | Latest | Fast synthesis |
| **DRL Framework** | PyTorch | 2.0+ | Policy training |
| **Robot SDK** | reachy-mini | 1.0+ | Robot control |
| **Simulation** | MuJoCo | Latest | Development and testing |

### Development Tools

| Tool | Purpose |
|------|---------|
| **Python 3.11+** | Primary language |
| **Poetry** | Dependency management |
| **Docker** | Containerization |
| **pytest** | Testing framework |
| **black** | Code formatting |
| **mypy** | Type checking |
| **ruff** | Linting |
| **pre-commit** | Git hooks |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Database** | PostgreSQL | Feedback and demonstration storage |
| **Cache** | Redis | Session management, caching |
| **Queue** | Celery + RabbitMQ | Async task processing |
| **Monitoring** | Prometheus + Grafana | Metrics and visualization |
| **Logging** | ELK Stack | Centralized logging |
| **Tracing** | OpenAI Agents SDK built-in | Agent flow visualization |

---

## API Specifications

### REST API Endpoints

#### Chat & Conversation

```yaml
POST /api/v1/chat
Description: Send text message to agent
Request:
  message: string
  session_id: string (optional)
Response:
  response: string
  session_id: string
  agent_used: string
  tools_called: list[string]
  trace_id: string

GET /api/v1/session/{session_id}
Description: Get session information
Response:
  session_id: string
  created_at: timestamp
  message_count: int
  active_agent: string

GET /api/v1/session/{session_id}/history
Description: Get conversation history
Response:
  messages: list[{
    role: string,
    content: string,
    timestamp: timestamp,
    agent: string
  }]
```

#### Voice Interaction

```yaml
WS /api/v1/ws/voice
Description: Real-time voice interaction
Send: Binary audio chunks (16kHz, 16-bit, mono)
Receive:
  type: "transcription" | "response" | "audio"
  data: string | bytes

POST /api/v1/voice/stt
Description: Speech-to-text (batch)
Request: Binary audio file
Response:
  transcription: string
  confidence: float
  duration: float

POST /api/v1/voice/tts
Description: Text-to-speech
Request:
  text: string
  voice: string (optional)
Response: Binary audio file
```

#### Vision & Perception

```yaml
POST /api/v1/vision/analyze
Description: Analyze image
Request:
  image: base64 or multipart
  prompt: string (optional)
Response:
  description: string
  objects: list[{name: string, confidence: float}]
  scene_type: string

POST /api/v1/vision/capture
Description: Capture image from robot camera
Response:
  image: base64
  timestamp: timestamp
```

#### Tools

```yaml
GET /api/v1/tools
Description: List available tools
Response:
  tools: list[{
    name: string,
    description: string,
    parameters: dict,
    source: "predefined" | "generated"
  }]

POST /api/v1/tools/execute
Description: Execute specific tool
Request:
  tool_name: string
  parameters: dict
Response:
  status: "success" | "error"
  result: any
  execution_time: float

POST /api/v1/tools/generate [Phase 2]
Description: Generate new tool
Request:
  description: string
  test_cases: list[dict]
Response:
  tool_name: string
  code: string
  validation_result: dict
  test_results: dict
```

#### Feedback & Training [Phase 3]

```yaml
POST /api/v1/feedback
Description: Submit feedback for episode
Request:
  episode_id: string
  feedback_text: string
  rating: float (optional)
Response:
  status: string
  reward: float

POST /api/v1/demonstration
Description: Submit expert demonstration
Request:
  task_description: string
  actions: list[dict]
Response:
  demo_id: string
  status: string

GET /api/v1/training/status
Description: Get training progress
Response:
  episode: int
  success_rate: float
  avg_reward: float
  best_checkpoint: string
```

---

## Development Roadmap

### Phase 1: Months 1-3

| Month | Milestone | Deliverables |
|-------|-----------|-------------|
| **Month 1** | Foundation | • Project structure<br>• FastAPI skeleton<br>• Ollama integration<br>• Basic tools (5+) |
| **Month 2** | Agent System | • OpenAI Agents SDK integration<br>• Multi-agent architecture<br>• Tool system complete<br>• Voice pipeline (STT + TTS) |
| **Month 3** | Integration & Testing | • End-to-end workflows<br>• Performance optimization<br>• Documentation<br>• **v1 Demo** |

### Phase 2: Months 4-6

| Month | Milestone | Deliverables |
|-------|-----------|-------------|
| **Month 4** | Code Generation | • CodeAgent implementation<br>• Tool generation pipeline<br>• Safety validation |
| **Month 5** | Testing & Safety | • Sandboxed execution<br>• Test framework<br>• Tool registry with versioning |
| **Month 6** | Integration | • Self-coding workflows<br>• Human review interface<br>• **v2 Demo** |

### Phase 3: Months 7-12

| Month | Milestone | Deliverables |
|-------|-----------|-------------|
| **Month 7-8** | Reward Model | • Feedback collection system<br>• Reward model training<br>• Feedback interface |
| **Month 9-10** | Policy Training | • PPO implementation<br>• Environment wrapper<br>• Training pipeline |
| **Month 11** | Integration | • End-to-end DRL loop<br>• Demonstration system |
| **Month 12** | Evaluation & Publication | • Benchmark results<br>• **v3 Demo**<br>• Research paper |

---

## Implementation Guidelines

### Project Structure

```
reachy-mini/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── main_agent.py
│   │   ├── robot_control_agent.py
│   │   ├── vision_agent.py
│   │   ├── code_agent.py         [Phase 2]
│   │   └── base.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── predefined/
│   │   │   ├── robot_control.py
│   │   │   ├── vision.py
│   │   │   └── web_search.py
│   │   ├── generated/            [Phase 2]
│   │   ├── registry.py
│   │   └── validator.py          [Phase 2]
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   ├── voice.py
│   │   │   ├── vision.py
│   │   │   ├── tools.py
│   │   │   └── feedback.py       [Phase 3]
│   │   └── middleware/
│   ├── multimodal/
│   │   ├── __init__.py
│   │   ├── stt.py               # Whisper
│   │   ├── tts.py               # Piper
│   │   └── vision.py            # Gemma vision
│   ├── drl/                      [Phase 3]
│   │   ├── __init__.py
│   │   ├── reward_model.py
│   │   ├── policy_network.py
│   │   ├── ppo_trainer.py
│   │   ├── environment.py
│   │   └── evaluation.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── feedback.py
│   │   └── demonstrations.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── config.py
├── tests/
│   ├── test_agents/
│   ├── test_tools/
│   ├── test_api/
│   └── test_drl/
├── scripts/
│   ├── train_reward_model.py    [Phase 3]
│   ├── train_policy.py          [Phase 3]
│   └── evaluate.py
├── configs/
│   ├── agents.yaml
│   ├── tools.yaml
│   └── training.yaml             [Phase 3]
├── docs/
│   ├── API.md
│   ├── ARCHITECTURE.md
│   └── DEVELOPMENT.md
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── pyproject.toml
├── README.md
└── PROJECT_SPEC.md              # This document
```

### Coding Standards

1. **Type Hints**: All functions must have type hints
2. **Docstrings**: Google style docstrings for all public functions
3. **Testing**: 80%+ code coverage
4. **Formatting**: Black with line length 100
5. **Linting**: Ruff for code quality
6. **Async**: Use async/await for I/O operations
7. **Error Handling**: Explicit error handling, no silent failures

### Example Code Style

```python
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

async def execute_tool(
    tool_name: str,
    parameters: Dict[str, Any],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a tool with given parameters.

    Args:
        tool_name: Name of the tool to execute
        parameters: Tool parameters
        session_id: Optional session identifier

    Returns:
        Execution result dictionary with status and data

    Raises:
        ToolNotFoundError: If tool doesn't exist
        ValidationError: If parameters are invalid
    """
    logger.info(f"Executing tool: {tool_name} for session {session_id}")

    # Validate
    if tool_name not in tool_registry:
        logger.error(f"Tool not found: {tool_name}")
        raise ToolNotFoundError(f"Tool {tool_name} not found")

    # Execute
    try:
        result = await tool_registry[tool_name](**parameters)
        logger.info(f"Tool {tool_name} executed successfully")
        return {"status": "success", "data": result}
    except Exception as e:
        logger.exception(f"Tool {tool_name} failed: {e}")
        return {"status": "error", "message": str(e)}
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_tools/test_robot_control.py

import pytest
from src.tools.predefined.robot_control import move_head

@pytest.mark.asyncio
async def test_move_head_valid_params():
    """Test head movement with valid parameters"""
    result = await move_head(pitch=10, yaw=20, roll=5, duration=2.0)

    assert result["status"] == "success"
    assert "final_position" in result
    assert result["final_position"]["pitch"] == 10

@pytest.mark.asyncio
async def test_move_head_invalid_pitch():
    """Test head movement with out-of-range pitch"""
    result = await move_head(pitch=100, yaw=0, roll=0)

    assert result["status"] == "error"
    assert "out of safe range" in result["message"]
```

### Integration Tests

```python
# tests/test_api/test_chat_endpoint.py

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_chat_endpoint():
    """Test basic chat interaction"""
    response = client.post(
        "/api/v1/chat",
        json={"message": "Move your head left"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "session_id" in data
    assert "move_head" in data["tools_called"]
```

### End-to-End Tests

```python
# tests/test_e2e/test_multimodal_workflow.py

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_vision_and_speak_workflow():
    """Test complete workflow: take photo, analyze, speak result"""

    # Initialize session
    session = await create_session()

    # Send request
    result = await session.run("Take a photo and describe what you see")

    # Verify workflow
    assert result.status == "success"
    assert "take_photo" in result.tools_used
    assert "analyze_scene" in result.tools_used
    assert "speak" in result.tools_used
    assert len(result.message) > 0
```

### Performance Tests

```python
# tests/test_performance/test_latency.py

import pytest
import time

@pytest.mark.performance
@pytest.mark.asyncio
async def test_voice_pipeline_latency():
    """Verify voice pipeline meets <500ms requirement"""

    audio_chunk = load_test_audio()

    start = time.time()
    transcription = await stt_processor.transcribe(audio_chunk)
    latency = (time.time() - start) * 1000  # ms

    assert latency < 500, f"STT latency {latency}ms exceeds 500ms requirement"
```

---

## Evaluation Metrics

### Phase 1 Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Voice latency (STT) | <500ms | Time from audio → text |
| Voice latency (TTS) | <500ms | Time from text → audio |
| API response time | <100ms | 95th percentile |
| Tool success rate | >90% | Successful executions / total |
| Agent handoff accuracy | >95% | Correct specialist / total handoffs |
| Session uptime | >99% | Time operational / total time |

### Phase 2 Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Code generation success | >80% | Valid tools / generation attempts |
| Safety validation pass rate | >95% | Safe tools / validated tools |
| Test pass rate | >90% | Tools passing tests / tested tools |
| Tool incorporation rate | >70% | Incorporated / generated tools |
| Zero critical failures | 0 | Safety violations in production |

### Phase 3 Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Reward model accuracy | >85% | Predicted ≈ human rating |
| Policy improvement | +20% | Success rate increase |
| Sample efficiency | <1000 episodes | Episodes to convergence |
| Generalization | >70% | Success on novel tasks |
| Human satisfaction | >4.0/5.0 | Post-training survey |
| Safety violations | <1% | Unsafe actions / total actions |

### Benchmark Tasks

**Phase 1**: Basic Competence
1. Look at object and describe it
2. Find and identify specific object
3. Answer questions about scene
4. Execute multi-step command ("Look left, take photo, describe")
5. Handle interruptions and corrections

**Phase 2**: Self-Extension
6. Generate tool for new capability
7. Fix failing tool based on error
8. Combine multiple tools for complex task
9. Version management (rollback, upgrade)
10. Safe rejection of dangerous requests

**Phase 3**: Learning & Adaptation
11. Improve task execution based on feedback
12. Learn from demonstrations
13. Transfer knowledge to similar tasks
14. Handle distribution shift
15. Maintain safety under exploration

---

## Challenges & Mitigation

### Challenge 1: Real-Time Voice (<500ms)

**Problem**: Whisper models can be slow, especially larger ones

**Mitigation**:
- Use tiny/base Whisper models (sacrifice some accuracy)
- Implement VAD to reduce processing time
- Use streaming transcription where possible
- Consider Whisper.cpp for faster inference
- Hardware acceleration (GPU/Metal)

### Challenge 2: LLM Latency for Tool Calls

**Problem**: Gemma 27B can take 3-5 seconds for complex reasoning

**Mitigation**:
- Cache frequent tool combinations
- Use smaller model for simple tasks
- Implement request batching
- Optimize prompts for conciseness
- Consider fine-tuned smaller models

### Challenge 3: Safety in Self-Coding

**Problem**: Generated code could be malicious or break system

**Mitigation**:
- Multi-layer validation (AST, pattern matching, testing)
- Sandboxed execution environment
- Whitelist approach for imports and operations
- Human-in-the-loop for critical tools
- Rollback mechanism for failures
- Resource limits (CPU, memory, time)

### Challenge 4: Reward Signal from NL Feedback

**Problem**: Natural language is ambiguous, hard to convert to scalar reward

**Mitigation**:
- Collect diverse feedback examples
- Fine-tune dedicated reward model
- Combine multiple signals (NL + explicit ratings)
- Use uncertainty estimates
- Active learning for ambiguous cases

### Challenge 5: Sample Efficiency in DRL

**Problem**: RL typically requires many samples, expensive with real robot

**Mitigation**:
- Simulation-first training (MuJoCo)
- Behavioral cloning for initialization
- Off-policy algorithms (SAC)
- Human demonstrations to seed buffer
- Curriculum learning
- Sim-to-real transfer techniques

### Challenge 6: Distribution Shift

**Problem**: Policy trained on certain tasks may fail on new ones

**Mitigation**:
- Diverse training tasks
- Domain randomization in simulation
- Continual learning approaches
- Meta-learning for quick adaptation
- Safety fallbacks for OOD detection

---

## References & Resources

### OpenAI Agents SDK
- [Official Documentation](https://github.com/openai/openai-agents-sdk)
- [Swarm Framework (predecessor)](https://github.com/openai/swarm)

### LLM & Inference
- [Ollama](https://ollama.ai)
- [Gemma 3 Documentation](https://ai.google.dev/gemma)
- [vLLM](https://github.com/vllm-project/vllm)

### Robotics
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
- [MuJoCo](https://mujoco.org)

### Speech
- [Whisper](https://github.com/openai/whisper)
- [Piper TTS](https://github.com/rhasspy/piper)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)

### Reinforcement Learning
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

### Human-in-the-Loop RL
- [RLHF Papers](https://arxiv.org/abs/2203.02155)
- [TAMER Framework](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/AAAI08-knox.pdf)
- [COACH](https://arxiv.org/abs/1809.10618)

### Code Generation & Safety
- [CodeQL](https://codeql.github.com/)
- [Bandit Security Linter](https://github.com/PyCQA/bandit)
- [RestrictedPython](https://github.com/zopefoundation/RestrictedPython)

---

## Appendix: Example Workflows

### Example 1: Basic Voice Interaction

```
User: [speaks] "Hey Reachy, look at the table and tell me what you see"

System:
1. VAD detects speech start
2. Whisper transcribes in <500ms
3. MainAgent receives: "look at the table and tell me what you see"
4. MainAgent identifies: Vision task + Speech output
5. Handoff to VisionAgent
6. VisionAgent:
   - Calls take_photo()
   - Calls analyze_scene("describe the table")
   - Returns: "I see a wooden table with an apple, croissant, and rubber duck"
7. Return to MainAgent
8. MainAgent calls speak("I see a wooden table...")
9. Piper TTS generates audio in <500ms
10. Audio played to user

Total latency: ~2 seconds (acceptable for complex task)
```

### Example 2: Tool Generation (Phase 2)

```
User: "I need you to monitor the table and alert me if something new appears"

MainAgent: "I don't have a monitoring tool. Let me create one."

1. Handoff to CodeAgent
2. CodeAgent generates:

async def monitor_scene_changes(interval: int = 5, duration: int = 60) -> dict:
    """Monitor scene for changes and alert on new objects"""
    from reachy_mini import ReachyMini
    import cv2
    import time

    with ReachyMini() as reachy:
        # Get baseline
        baseline = reachy.media.camera.read()
        baseline_description = await analyze_scene("list objects")
        baseline_objects = set(baseline_description.split(', '))

        alerts = []
        start_time = time.time()

        while time.time() - start_time < duration:
            time.sleep(interval)

            # Check current scene
            current = reachy.media.camera.read()
            current_description = await analyze_scene("list objects")
            current_objects = set(current_description.split(', '))

            # Detect new objects
            new_objects = current_objects - baseline_objects
            if new_objects:
                alerts.append({
                    'time': time.time() - start_time,
                    'new_objects': list(new_objects)
                })

        return {"status": "success", "alerts": alerts}

3. Validation: ✓ Passes safety checks
4. Testing: ✓ Passes with mock objects
5. Registered as "monitor_scene_changes" v1
6. MainAgent: "Tool created! Now monitoring..."
7. Executes: monitor_scene_changes(interval=5, duration=60)
```

### Example 3: Learning from Feedback (Phase 3)

```
Episode 1:
Task: "Move head to look at the apple"
Agent action: move_head(pitch=45, yaw=0, roll=0)
Outcome: Head moves but overshoots, apple not centered
Human feedback: "Too far down, needs to be less tilted"
Reward model converts: feedback → reward = -0.3
Policy update: Reduce pitch for "look at apple" task

Episode 2:
Task: "Move head to look at the apple"
Agent action: move_head(pitch=30, yaw=0, roll=0)
Outcome: Apple centered in view
Human feedback: "Perfect!"
Reward model converts: feedback → reward = +1.0
Policy update: Reinforce this behavior

After 50 episodes:
Agent learns optimal pitch angle for apple height
Generalizes to other objects at similar height
Success rate: 85% → 95%
```

---

**End of Project Specification Document**

*This is a living document and will be updated as the project progresses.*
