# OpenAI Agents Handoff Hierarchy Demo 🚀

This demo showcases the **fixed handoff hierarchy** where handed-off agents appear as child spans under their parent agents in Traceloop, instead of separate root spans.

## 🎯 What You'll See

**BEFORE (Broken):**
```
Data Router (root)
Analytics Agent (root)  ❌ Wrong - should be child!
```

**AFTER (Fixed):**
```
Data Router (root)
├─ Analytics Agent (child) ✅ Perfect hierarchy!
│  ├─ analyze_data.tool
│  └─ generate_report.tool
```

## 🚀 Quick Start

### 1. Set Environment Variables

Create a `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
```

Then edit `.env` with your API keys:
```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
TRACELOOP_API_KEY=your-traceloop-api-key-here  # Optional but recommended
```

Or set environment variables directly:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export TRACELOOP_API_KEY="your-traceloop-api-key"  # Optional but recommended
```

### 2. Install Dependencies
```bash
pip install openai-agents traceloop-sdk
```

### 3. Run the Demo
```bash
# Simple demo (recommended)
python simple_handoff_demo.py

# Or full featured demo
python sample_handoff_app.py
```

### 4. View Results in Traceloop
- Go to https://app.traceloop.com/
- Look for traces from "handoff-demo" app
- See the beautiful parent-child hierarchy! 🎉

## 📊 Expected Trace Structure

You should see a **single unified trace** with this hierarchy:

```
🔗 Trace: handoff-demo
├─ 🤖 Data Router (root span)
   ├─ 🤖 Analytics Agent (child span) ✅
      ├─ 🔧 analyze_data.tool (grandchild)
      └─ 🔧 generate_report.tool (grandchild)
```

## 🎉 Success Indicators

✅ **Single trace** (not multiple separate traces)  
✅ **Analytics Agent is child** of Data Router  
✅ **Tools are nested** under Analytics Agent  
✅ **Proper parent-child relationships** throughout  

## 🔧 How It Works

The demo uses the **fixed OpenTelemetry instrumentation** that:

1. **Detects handoffs** via the agents framework tracing system
2. **Registers handoff contexts** with parent span information  
3. **Maintains context** across multiple tool executions
4. **Creates child spans** in the same OpenTelemetry trace

## 📝 Sample Scenarios

The demo includes realistic scenarios:
- **User Behavior Analysis**: Analyze website engagement patterns
- **E-commerce Analytics**: Customer purchase pattern analysis  
- **Marketing Campaign Analysis**: Campaign performance optimization

## 🐛 Troubleshooting

**No traces in Traceloop?**
- Check your `TRACELOOP_API_KEY` is set correctly
- Verify you're looking at the right project/environment
- Wait 30 seconds for trace ingestion

**Still seeing multiple root spans?**
- This means the fix isn't working - check OpenTelemetry instrumentation
- Look for debug logs showing "Found handoff context"

**API errors?**
- Verify `OPENAI_API_KEY` is valid and has credits
- Check network connectivity

## 🎯 Key Benefits

This fixed hierarchy enables:
- **Better debugging**: Clear parent-child relationships
- **Performance analysis**: See handoff overhead vs tool execution
- **Workflow understanding**: Visual representation of agent interactions
- **Error tracking**: Failures properly attributed to responsible agents

## 💡 Next Steps

Once you see the working hierarchy:
1. **Integrate into your application** using the same patterns
2. **Add custom attributes** for better filtering in Traceloop  
3. **Set up alerts** for failed handoffs or long-running workflows
4. **Create dashboards** showing agent performance metrics

Happy tracing! 🎉