import requests
import time
import json

# CONFIGURATION
VISION_BACKEND_URL = "http://127.0.0.1:8000"  # Change to your backend address if needed
VISION_COMMAND_FILE = "vision_command.json"  # File to listen for Vision's commands

def read_command():
    try:
        with open(VISION_COMMAND_FILE, 'r+') as file:
            return json.load(file)
    except FileNotFoundError:
        return None

def write_response(response):
    with open("vision_response.json", 'w+') as file:
        json.dump(response, file, indent=4)

def process_command(command):
    action = command.get("action")
    if action == "forecast":
        return requests.get(f"{VISION_BACKEND_URL}/forecast").json()
    elif action == "run_amplifier":
        payload = command.get("payload")
        return requests.post(f"{VISION_BACKEND_URL}/run_amplifier", json=payload).json()
    else:
        return {"error": "Unknown command"}

def main_loop():
    print("🚀 Vision API Client is running...")
    while True:
        command = read_command()
        if command:
            print(f"Received command: {command}")
            response = process_command(command)
            print(f"Response: {response}")
            write_response(response)
            # Clear command after processing
            with open(VISION_COMMAND_FILE, 'w') as file:
                json.dump({}, file)
        time.sleep(1)  # Check every second

if __name__ == "__main__":
    main_loop()
