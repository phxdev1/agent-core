#!/usr/bin/env python3
"""
ReAct Orchestrator Agent using LangChain
Coordinates tool usage with reasoning and acting loops
"""

import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, BaseTool
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI

# Import our systems
from knowledge.web_search_mcp import get_web_search
from knowledge.fact_verification_direct import get_direct_fact_system
from core.unified_memory_system import get_unified_memory, StepType
from utils.config_loader import config
from utils.redis_logger import get_redis_logger

logger = get_redis_logger(__name__)


class ToolCallbackHandler(BaseCallbackHandler):
    """Callback handler to log tool usage"""
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Log when a tool is called"""
        tool_name = serialized.get("name", "unknown")
        logger.info(f"Calling tool: {tool_name} with input: {input_str[:100]}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Log tool output"""
        logger.info(f"Tool returned: {output[:200]}")
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> Any:
        """Log agent actions"""
        logger.info(f"Agent action: {action.tool} - {action.tool_input}")


class MCPToolWrapper(BaseTool):
    """Wrapper to convert MCP tools to LangChain tools"""
    
    def __init__(self, name: str, description: str, func: callable):
        super().__init__()
        self.name = name
        self.description = description
        self.func = func
        
    def _run(self, query: str) -> str:
        """Synchronous run (for compatibility)"""
        return asyncio.run(self._arun(query))
    
    async def _arun(self, query: str) -> str:
        """Asynchronous run"""
        try:
            result = await self.func(query)
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2)
            return str(result)
        except Exception as e:
            logger.error(f"Tool {self.name} error: {e}")
            return f"Error: {str(e)}"


class ReActOrchestrator:
    """
    ReAct Orchestrator using LangChain
    Coordinates multiple tools with reasoning and acting
    """
    
    def __init__(self):
        """Initialize the orchestrator with all tools"""
        self.api_key = config.get('openrouter_api_key')
        self.model = config.get('model', 'mistralai/mistral-medium-3.1')
        
        # Initialize memory system
        self.memory_system = get_unified_memory()
        self.memory_system.create_session("orchestrator")
        
        # Initialize tool systems
        self.web_search = None
        self.fact_checker = None
        
        # Setup LangChain components
        self.llm = self._setup_llm()
        self.tools = self._setup_tools()
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=10,  # Keep last 10 exchanges
            return_messages=True
        )
        self.agent = self._setup_agent()
        self.executor = self._setup_executor()
        
        logger.info("ReAct Orchestrator initialized with LangChain")
    
    def _setup_llm(self):
        """Setup the language model using OpenRouter via OpenAI interface"""
        # Use OpenAI client with OpenRouter endpoint
        return ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3,  # Lower for more focused reasoning
            max_tokens=2048
        )
    
    def _setup_tools(self) -> List[Tool]:
        """Setup all available tools"""
        tools = []
        
        # Web Search Tool
        async def web_search_func(query: str) -> str:
            """Search the web for information"""
            if self.web_search is None:
                self.web_search = await get_web_search()
            results = await self.web_search.web_search(query, num_results=3)
            return json.dumps(results, indent=2)
        
        tools.append(MCPToolWrapper(
            name="web_search",
            description="Search the web for current information. Input should be a search query.",
            func=web_search_func
        ))
        
        # News Search Tool
        async def news_search_func(query: str) -> str:
            """Search for recent news"""
            if self.web_search is None:
                self.web_search = await get_web_search()
            results = await self.web_search.news_search(query, time_range="week")
            return json.dumps(results, indent=2)
        
        tools.append(MCPToolWrapper(
            name="news_search",
            description="Search for recent news articles. Input should be a news topic.",
            func=news_search_func
        ))
        
        # Fact Checking Tool
        async def fact_check_func(claim: str) -> str:
            """Verify a factual claim"""
            if self.fact_checker is None:
                self.fact_checker = await get_direct_fact_system()
            result = await self.fact_checker.verify_fact(claim)
            return f"Verified: {result.verified}, Confidence: {result.confidence:.0%}, Evidence: {result.evidence[:2]}"
        
        tools.append(MCPToolWrapper(
            name="fact_check",
            description="Verify if a factual claim is true. Input should be a statement to verify.",
            func=fact_check_func
        ))
        
        # Wikipedia Tool
        async def wikipedia_func(topic: str) -> str:
            """Get information from Wikipedia"""
            if self.fact_checker is None:
                self.fact_checker = await get_direct_fact_system()
            info = await self.fact_checker.get_accurate_info(topic)
            return json.dumps(info, indent=2)
        
        tools.append(MCPToolWrapper(
            name="wikipedia",
            description="Get accurate information from Wikipedia about a topic.",
            func=wikipedia_func
        ))
        
        # Calculator Tool
        tools.append(Tool(
            name="calculator",
            description="Perform mathematical calculations. Input should be a mathematical expression.",
            func=lambda expr: str(eval(expr, {"__builtins__": {}}, {}))
        ))
        
        # Current Date/Time Tool
        tools.append(Tool(
            name="current_datetime",
            description="Get the current date and time.",
            func=lambda _: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        
        return tools
    
    def _setup_agent(self):
        """Setup the ReAct agent"""
        # ReAct prompt template
        react_prompt = PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad", "chat_history"],
            template="""You are a helpful AI assistant with access to various tools. Use the ReAct framework to solve problems:
- Thought: Reason about what to do
- Action: Choose a tool to use
- Action Input: Provide input for the tool
- Observation: See the tool's output
- ... (repeat as needed)
- Thought: I now know the final answer
- Final Answer: Provide the complete answer

Available tools:
{tools}

Chat History:
{chat_history}

Question: {input}

Thought: Let me think about how to approach this question.
{agent_scratchpad}"""
        )
        
        # Create the ReAct agent
        return create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )
    
    def _setup_executor(self):
        """Setup the agent executor"""
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,  # Show reasoning steps
            max_iterations=5,  # Limit iterations to prevent loops
            handle_parsing_errors=True,
            callbacks=[ToolCallbackHandler()]
        )
    
    async def process_message(self, message: str) -> str:
        """
        Process a message using the ReAct framework
        
        Args:
            message: User message to process
            
        Returns:
            Agent response after reasoning and tool usage
        """
        try:
            # Log to memory system
            self.memory_system.add_conversation_step(
                step_type=StepType.USER_INPUT,
                content={"message": message}
            )
            
            # Detect if we need specific tools based on message
            message_lower = message.lower()
            
            # Add hints to help the agent choose tools
            if any(keyword in message_lower for keyword in ["search", "find", "look up", "google"]):
                message = f"{message} (Hint: You may want to use web_search)"
            elif any(keyword in message_lower for keyword in ["news", "latest", "recent", "today"]):
                message = f"{message} (Hint: You may want to use news_search)"
            elif any(keyword in message_lower for keyword in ["verify", "fact check", "is it true", "correct"]):
                message = f"{message} (Hint: You may want to use fact_check)"
            elif any(keyword in message_lower for keyword in ["calculate", "compute", "math", "solve"]):
                message = f"{message} (Hint: You may want to use calculator)"
            
            # Run the agent executor
            result = await self.executor.ainvoke({"input": message})
            
            # Extract the response
            response = result.get("output", "I couldn't process that request.")
            
            # Log to memory system
            self.memory_system.add_conversation_step(
                step_type=StepType.AGENT_RESPONSE,
                content={"response": response}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return f"I encountered an error: {str(e)}"
    
    async def plan_and_execute(self, task: str) -> Dict[str, Any]:
        """
        Plan and execute a complex task
        
        Args:
            task: Complex task to break down and execute
            
        Returns:
            Execution results with plan and outcomes
        """
        # Create a planning prompt
        planning_prompt = f"""
        Break down this task into steps that can be executed with available tools:
        Task: {task}
        
        Available tools: {', '.join([t.name for t in self.tools])}
        
        Create a numbered list of steps.
        """
        
        # Get the plan
        plan_result = await self.executor.ainvoke({"input": planning_prompt})
        plan = plan_result.get("output", "")
        
        # Execute the task
        execution_prompt = f"""
        Execute this task step by step using the available tools:
        {task}
        
        Plan:
        {plan}
        
        Execute each step and provide the final result.
        """
        
        execution_result = await self.executor.ainvoke({"input": execution_prompt})
        
        return {
            "task": task,
            "plan": plan,
            "execution": execution_result.get("output", ""),
            "tools_used": [t.name for t in self.tools if t.name in str(execution_result)]
        }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history"""
        messages = self.memory.chat_memory.messages
        history = []
        for msg in messages:
            if hasattr(msg, 'content'):
                role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                history.append({"role": role, "content": msg.content})
        return history
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Orchestrator memory cleared")


# Singleton instance
_orchestrator = None

async def get_orchestrator() -> ReActOrchestrator:
    """Get or create the orchestrator singleton"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ReActOrchestrator()
    return _orchestrator