from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pymongo import MongoClient
import uuid
import datetime
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client.llm_chat_db

# LLM API configuration
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_default_url = "http://127.0.0.1:5001/generate"
LLM_API_URL = os.getenv("LLM_API_URL", LLM_default_url)

@app.route('/api/chat/message', methods=['POST'])
def handle_message():
    """Handle incoming chat messages, process with LLM, and store in MongoDB"""
    data = request.json
    message = data.get('message')
    conversation_id = data.get('conversationId')
    
    # Get settings from the request if provided
    settings = data.get('settings', {})
    temperature = settings.get('temperature', 0.5)
    max_tokens = settings.get('maxTokens', 2000)
    top_p = settings.get('topP', 1.0)
    top_k = settings.get('topK', 5)
    context_length = settings.get('contextLength', 6)
    system_prompt = settings.get('systemPrompt', "You're a helpful assistant.")
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    # Create a new conversation if no conversation_id is provided
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        title = generate_title(message)
        
        # Create a new conversation in MongoDB
        db.conversations.insert_one({
            "conversationId": conversation_id,
            "title": title,
            "createdAt": datetime.datetime.utcnow(),
            "lastMessageTimestamp": datetime.datetime.utcnow()
        })
    
    # Store user message in MongoDB
    user_message_data = {
        "conversationId": conversation_id,
        "role": "user",
        "content": message,
        "timestamp": datetime.datetime.utcnow()
    }
    
    db.messages.insert_one(user_message_data)
    
    # Get context for LLM (previous messages)
    conversation_history = []
    if context_length > 0:
        messages = list(db.messages.find(
            {"conversationId": conversation_id},
            {"_id": 0, "role": 1, "content": 1}
        ).sort("timestamp", -1).limit(context_length))
        
        # Reverse to get chronological order
        conversation_history = list(reversed(messages))
    
    # Prepare request to LLM API
    llm_request_data = {
        "messages": conversation_history + [{"role": "user", "content": message}],
        "settings" : {
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k
        }
    }
    
    # Call LLM API
    try:
        headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
        llm_response = requests.post(LLM_API_URL, json=llm_request_data, headers=headers)
        llm_response.raise_for_status()
        
        # Extract the LLM's response text
        response_data = llm_response.json()
        llm_message = response_data.get("response", "I couldn't generate a response at this time.")
        
        # Store LLM response in MongoDB
        timestamp = datetime.datetime.utcnow()
        bot_message_data = {
            "conversationId": conversation_id,
            "role": "assistant",
            "content": llm_message,
            "timestamp": timestamp
        }
        
        db.messages.insert_one(bot_message_data)
        
        # Update conversation last message timestamp
        db.conversations.update_one(
            {"conversationId": conversation_id},
            {"$set": {"lastMessageTimestamp": timestamp}}
        )
        
        # Return response to frontend
        return jsonify({
            "conversationId": conversation_id,
            "response": llm_message,
            "timestamp": timestamp.isoformat()
        })
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM API: {e}")
        return jsonify({"error": "Failed to get response from LLM service"}), 500

@app.route('/api/chat/conversations', methods=['GET'])
def get_conversations():
    """Retrieve a list of all conversations"""
    # Get conversations sorted by lastMessageTimestamp (newest first)
    conversations = list(db.conversations.find(
        {},
        {"_id": 0, "conversationId": 1, "title": 1, "lastMessageTimestamp": 1}
    ).sort("lastMessageTimestamp", -1))
    
    # Format timestamps to ISO format for JSON serialization
    for conv in conversations:
        if "lastMessageTimestamp" in conv:
            conv["lastMessageTimestamp"] = conv["lastMessageTimestamp"].isoformat()
    
    return jsonify(conversations)

@app.route('/api/chat/conversations/<conversation_id>/rename', methods=['PATCH'])
def rename_conversation(conversation_id):
    """Rename an existing conversation"""
    data = request.json
    new_title = data.get('newTitle')
    
    if not new_title:
        return jsonify({"error": "No new title provided"}), 400
    
    result = db.conversations.update_one(
        {"conversationId": conversation_id},
        {"$set": {"title": new_title}}
    )
    
    if result.matched_count == 0:
        return jsonify({"error": "Conversation not found"}), 404
    
    return jsonify({"success": True, "conversationId": conversation_id, "title": new_title})

@app.route('/api/chat/conversations/<conversation_id>/delete', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a conversation and its messages"""
    # Delete the conversation
    conversation_result = db.conversations.delete_one({"conversationId": conversation_id})
    
    if conversation_result.deleted_count == 0:
        return jsonify({"error": "Conversation not found"}), 404
    
    # Delete all messages associated with the conversation
    db.messages.delete_many({"conversationId": conversation_id})
    
    return jsonify({"success": True})

@app.route('/api/chat/conversations/<conversation_id>/messages', methods=['GET'])
def get_conversation_messages(conversation_id):
    """Get all messages for a specific conversation"""
    messages = list(db.messages.find(
        {"conversationId": conversation_id},
        {"_id": 0, "role": 1, "content": 1, "timestamp": 1}
    ).sort("timestamp", 1))
    
    # Format timestamps for JSON serialization
    for msg in messages:
        if "timestamp" in msg:
            msg["timestamp"] = msg["timestamp"].isoformat()
    
    return jsonify(messages)

def generate_title(first_message):
    """Generate a title based on the first message of a conversation"""
    # In a real implementation, you might want to:
    # 1. Use the LLM to generate a title
    # 2. Extract keywords from the message
    # 3. Take the first few words
    
    # Simple implementation - truncate first message
    if len(first_message) > 30:
        return first_message[:30] + "..."
    return first_message

if __name__ == '__main__':
    app.run(debug=True, port=5000)