# backend/clarify.py
"""
Query clarification module for the Deep Researcher application.

This module handles the refinement and clarification of research queries.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from backend.config import OPENAI_API_KEY, BASE_MODEL, MODEL_TEMPERATURE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format to just show the message
)
logger = logging.getLogger(__name__)

class QueryClarifier:
    """Handles query refinement and clarification."""

    def __init__(self):
        """Initialize the query clarifier."""
        try:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set")

            self.llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                model=BASE_MODEL,
                temperature=MODEL_TEMPERATURE
            )
            self.chat_history = []
            logger.info(f"Initialized QueryClarifier with model: {self.llm.model_name} and temperature: {self.llm.temperature}")
        except Exception as e:
            logger.error(f"Error initializing QueryClarifier: {str(e)}")
            raise

    def _get_full_context_for_llm(self) -> str:
        """Constructs a string representation of the conversation history."""
        context = []
        for msg in self.chat_history:
            if isinstance(msg, HumanMessage):
                context.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                if not msg.content.strip().upper().startswith(("YES", "NO")):
                     context.append(f"Assistant: {msg.content}")
        return "\n".join(context)

    def _check_query_sufficiency(self) -> Tuple[bool, str]:
        """
        Check if the query, based on the current chat history, is sufficient.
        """
        system_message = """You are a research assistant. Your task is to evaluate if the user's request, based on the conversation history provided, is sufficiently clear and detailed for deep research.
Consider the latest user input in the context of the whole conversation.
Respond with exactly 'YES' if the request is sufficient, or 'NO' if it needs more clarification.
After YES/NO, provide a brief explanation for your decision."""

        messages = [SystemMessage(content=system_message)]
        messages.extend(self.chat_history)

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm
        response = chain.invoke({})

        response_content = response.content.strip()
        is_sufficient = response_content.upper().startswith("YES")
        return is_sufficient, response_content

    def _get_clarifying_questions(self) -> str:
        """
        Get clarifying questions based on the current chat history.
        """
        system_message = """You are a research assistant. Based on the conversation history, the user's research request is still not clear enough.
Ask specific, concise questions that would help clarify the query to you for better research.
Avoid asking questions you (the Assistant) have already asked in the conversation history provided.
Format your response only as a numbered list of questions, like:
1. Question 1?
2. Question 2?"""

        previous_ai_questions = []
        for msg in self.chat_history:
            if isinstance(msg, AIMessage) and "?" in msg.content:
                 lines = msg.content.split('\n')
                 for line in lines:
                     if line.strip().endswith('?'):
                         previous_ai_questions.append(line.strip())

        messages = [SystemMessage(content=system_message)]
        messages.extend(self.chat_history)

        if previous_ai_questions:
             messages.append(SystemMessage(content=f"Avoid re-asking questions similar to these: {'; '.join(previous_ai_questions)}"))

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm
        response = chain.invoke({})
        return response.content

    def _enhance_query(self) -> str:
        """
        Enhance the query based on the sufficient chat history.
        """
        system_message = """You are a research assistant. Based on the entire conversation history, the user's request is now clear.
Synthesize the information provided throughout the conversation into a single, comprehensive, concise and detailed research query."""

        messages = [SystemMessage(content=system_message)]
        messages.extend(self.chat_history)

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm
        response = chain.invoke({})
        return response.content

    def clarify_query(self, user_input: str) -> Tuple[Optional[str], bool, List[str]]:
        """
        Processes the latest user input, checks sufficiency, and either enhances the query or asks clarifying questions.
        """
        if not user_input:
            raise ValueError("User input cannot be empty")

        self.chat_history.append(HumanMessage(content=user_input))

        try:
            is_sufficient, explanation = self._check_query_sufficiency()
            self.chat_history.append(AIMessage(content=explanation))

            if is_sufficient:
                enhanced_query = self._enhance_query()
                self.chat_history.append(AIMessage(content=enhanced_query))
                return enhanced_query, True, []
            else:
                questions_str = self._get_clarifying_questions()
                self.chat_history.append(AIMessage(content=questions_str))
                questions_list = [q.strip() for q in questions_str.split('\n') if q.strip() and q.strip()[0].isdigit()]
                return None, False, questions_list

        except Exception as e:
            logger.error(f"Error in clarify_query: {str(e)}")
            raise

    def get_chat_history(self) -> List:
        """Get the current chat history."""
        return self.chat_history

    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []

    def interactive_clarify(self, initial_query: str, max_turns: int = 3) -> Optional[str]:
        """
        Interactively clarify a research query through conversation.
        """
        current_input = initial_query
        turn_count = 0
        
        while turn_count < max_turns:
            turn_count += 1
            
            enhanced_query, is_complete, questions = self.clarify_query(current_input)
            
            if is_complete:
                return enhanced_query
                
            if questions:
                print("\nPlease provide more details about:")
                for i, q in enumerate(questions, 1):
                    print(f"{i}. {q}")
            else:
                print("\nCould you please provide more details about your research query?")
                
            user_response = input("\nYour response (or 'quit' to exit): ").strip()
            
            if user_response.lower() == 'quit':
                return None
                
            if not user_response:
                print("Response cannot be empty. Please try again.")
                turn_count -= 1
                continue
                
            current_input = user_response
            
        return None

# --- Updated Example Usage ---
if __name__ == "__main__":
    try:
        clarifier = QueryClarifier()
        print("--- Starting Interactive Query Clarification ---")
        
        # Example initial query
        initial_query = "Tell me about quantum computing"
        print(f"Initial Query: {initial_query}")
        
        # Run interactive clarification
        final_query = clarifier.interactive_clarify(initial_query)
        
        if final_query:
            print("\n--- Clarification Complete ---")
            print(f"Final Research Query: {final_query}")
        else:
            print("\n--- Clarification Incomplete or Cancelled ---")
            
    except Exception as e:
        print(f"\n--- An Error Occurred ---")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()