# Gemma 3 Vision Integration with Reachy Mini

Gemma 3 is a multimodal LLM from Google that can process both text and images with a 128K context window and support for over 140 languages. This makes it perfect for giving Reachy Mini the ability to see and understand its environment!

## What is Gemma 3?

Gemma 3 is a family of lightweight models built on Gemini technology:
- **Multimodal**: Processes text AND images
- **128K context window**: Can analyze detailed scenes
- **140+ languages**: Multilingual understanding
- **Multiple sizes**: 270M, 1B, 4B, 12B, and 27B parameters
- **Efficient**: Runs locally on consumer hardware

## Available Demos

### 1. Simple Test (`gemma_vision_simple.py`)
**Best for**: Quick verification that vision works
**Time**: ~1-2 minutes
**Interaction**: Automated

Takes one photo and gets Gemma to describe it.

```bash
python gemma_vision_simple.py
```

**What it does**:
- Positions Reachy to look at the table
- Captures one image
- Asks Gemma to describe it
- Shows the result

**Expected output**:
```
🤖 GEMMA'S RESPONSE:
================================================================

The image shows a wooden table with several objects on it. There's
an apple, a croissant, and what appears to be a rubber duck toy.
The table surface is light brown wood with visible grain texture.

================================================================
```

### 2. Interactive Q&A (`gemma_vision_interactive.py`)
**Best for**: Exploring vision capabilities
**Time**: As long as you want!
**Interaction**: Type questions in real-time

Ask Reachy anything about what it sees!

```bash
python gemma_vision_interactive.py
```

**Example session**:
```
👤 You: What objects can you see?
🤖 Reachy: I can see an apple, a croissant, and a yellow rubber duck
           on a wooden table.

👤 You: What color is the apple?
🤖 Reachy: The apple is red with some yellow/green tones.

👤 You: Count all the items
🤖 Reachy: There are 3 distinct items: 1 apple, 1 croissant,
           and 1 rubber duck toy.

👤 You: new photo
📷 Taking photo...
✓ Photo saved

👤 You: Describe the table texture
🤖 Reachy: The table has a light brown wooden surface with visible
           grain patterns running horizontally...
```

**Commands**:
- Type any question about the scene
- `new photo` - Take a fresh picture
- `quit` - Exit

### 3. Full Demo (`gemma_vision_demo.py`)
**Best for**: Comprehensive demonstration
**Time**: ~15-20 minutes
**Interaction**: Automated showcase

Complete tour of Gemma's vision capabilities from multiple angles.

```bash
python gemma_vision_demo.py
```

**Demonstrates**:
1. **Scene 1**: Overview from table view
2. **Scene 2**: Objects from the left
3. **Scene 3**: Comparative view from right
4. **Scene 4**: Close-up detail analysis
5. **Scene 5**: Object counting
6. **Scene 6**: Creative interpretation
7. **Final**: Comprehensive multi-question analysis

**Saves 6 images** to `/tmp/reachy_scene*.jpg` for review.

## Performance Notes

### Speed
- **gemma3:27b**: ~30-60 seconds per image query
- **gemma3:12b**: ~15-30 seconds (if you have it)
- **gemma3:4b**: ~10-20 seconds (if you have it)

First query is slowest (model loading), subsequent queries are faster.

### Quality vs Speed
- **27B model**: Best quality, most detailed descriptions
- **12B model**: Good balance
- **4B model**: Faster but less detailed

### Memory Usage
- **gemma3:27b**: ~17GB disk space, ~12GB RAM during inference
- Ensure you have sufficient RAM available

## What Gemma Can Do

### Object Recognition
```python
"What objects can you see?"
→ Lists all visible items with details
```

### Spatial Understanding
```python
"Where is the apple relative to the duck?"
→ Describes spatial relationships
```

### Color Analysis
```python
"What colors dominate this scene?"
→ Identifies and describes colors
```

### Texture and Material
```python
"Describe the materials and textures"
→ Analyzes surface properties
```

### Counting
```python
"Count all the items on the table"
→ Enumerates objects accurately
```

### Comparative Analysis
```python
"Compare this to a typical dining table"
→ Makes contextual comparisons
```

### Creative Interpretation
```python
"Write a short story about this scene"
→ Creative and imaginative responses
```

### Multi-Question Analysis
```python
"Answer these: 1) Main object? 2) Colors? 3) Purpose?"
→ Handles complex multi-part queries
```

## Code Examples

### Basic Vision Query

```python
import base64
import requests
from reachy_mini import ReachyMini

with ReachyMini() as reachy:
    # Take photo
    frame = reachy.media.camera.read()

    # Save it
    import cv2
    cv2.imwrite("/tmp/view.jpg", frame)

    # Encode for Gemma
    with open("/tmp/view.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    # Ask Gemma
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:27b",
            "prompt": "What do you see?",
            "images": [img_b64],
            "stream": False,
        },
        timeout=120
    )

    answer = response.json()["response"]
    print(f"Gemma sees: {answer}")
```

### Custom Questions

```python
questions = [
    "Describe the scene in detail",
    "What is the most prominent object?",
    "What colors are present?",
    "Estimate the size of objects",
    "What is the lighting like?",
    "Describe any text visible",
    "What emotions does this scene evoke?",
]

for question in questions:
    answer = ask_gemma(image_path, question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### Multi-Angle Analysis

```python
from reachy_mini.utils import create_head_pose

angles = [
    ("center", 0, -20),
    ("left", 30, -20),
    ("right", -30, -20),
    ("close", 0, -30),
]

for name, yaw, pitch in angles:
    # Position robot
    pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
    reachy.goto_target(head=pose, duration=2.0)
    time.sleep(2.5)

    # Capture
    frame = reachy.media.camera.read()
    cv2.imwrite(f"/tmp/{name}.jpg", frame)

    # Analyze
    answer = ask_gemma(f"/tmp/{name}.jpg", "What's unique about this view?")
    print(f"{name}: {answer}")
```

## Troubleshooting

### "Model not found" Error

```bash
# Pull Gemma 3
ollama pull gemma3:27b

# Verify it's installed
ollama list
```

### Slow Responses

- **Normal**: First query takes 60+ seconds (loading model)
- **Subsequent**: ~30-45 seconds
- **Try smaller model**: `ollama pull gemma3:12b`

### Timeout Errors

```python
# Increase timeout
response = requests.post(..., timeout=180)  # 3 minutes
```

### Out of Memory

- Close other applications
- Use smaller model (12B or 4B)
- Ensure 12GB+ RAM available

### Poor Quality Descriptions

- Ensure good lighting in simulation
- Try different camera angles
- Be more specific in your questions
- Use the 27B model for best quality

## Comparison: Gemma vs LLaVA

| Feature | Gemma 3 | LLaVA |
|---------|---------|-------|
| **Speed** | Fast (30-60s) | Slower (60-120s) |
| **Quality** | Excellent | Excellent |
| **Size** | 27B | 13B typical |
| **Languages** | 140+ | English-focused |
| **Context** | 128K tokens | 4K-8K tokens |
| **Best for** | Detailed analysis, multilingual | General vision tasks |

Both are excellent! Gemma 3 is faster and more multilingual, while LLaVA is more specialized for vision.

## Tips for Best Results

### Photography
1. **Good angle**: Look slightly down at scene (-20° pitch)
2. **Centered**: Keep subject in frame center
3. **Lighting**: Simulation lighting is good by default
4. **Distance**: Not too close (objects should be fully visible)

### Questions
1. **Be specific**: "What color is the apple?" vs "colors?"
2. **One thing at a time**: Break complex queries into parts
3. **Use context**: "Compare to X" gives better answers
4. **Follow up**: Ask clarifying questions based on answers

### Performance
1. **Warmup**: First query is slow, subsequent ones faster
2. **Batch questions**: Ask multiple things in one prompt
3. **Reuse images**: Don't take new photo for each question
4. **Close other apps**: Free up RAM for Gemma

## Advanced Use Cases

### Object Tracking
Take photos at intervals and track object movement over time.

### Scene Comparison
Compare "before and after" scenes to detect changes.

### Visual Question Answering
Build a dataset of scene questions and answers for training.

### Accessibility
Describe environment for visually impaired users.

### Inventory Management
Count and catalog objects on shelves or tables.

### Quality Control
Inspect objects for defects or irregularities.

### Educational Tool
Teach computer vision concepts with real-time examples.

## Next Steps

1. **Start simple**: Run `gemma_vision_simple.py`
2. **Go interactive**: Try `gemma_vision_interactive.py`
3. **Full demo**: Experience `gemma_vision_demo.py`
4. **Build your own**: Use code examples to create custom apps
5. **Experiment**: Try different models, questions, and angles

## Resources

- [Gemma Documentation](https://ai.google.dev/gemma)
- [Ollama Gemma Guide](https://ollama.com/library/gemma3)
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)

---

**Happy exploring! 🤖👁️**
