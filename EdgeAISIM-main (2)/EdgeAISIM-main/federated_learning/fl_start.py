import subprocess
import time
import signal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def start_process(command):
    """Start a subprocess and return the process."""
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    try:
        logging.info("🚀 Starting the Federated Learning Server...")
        server_process = start_process(["python", "federated_learning/fl_server.py"])
        time.sleep(5)  # Allow the server to initialize

        logging.info("🔄 Starting Federated Learning Clients...")
        client1_process = start_process(["python", "federated_learning/fl_client.py"])
        client2_process = start_process(["python", "federated_learning/fl_client.py"])

        # Wait for the processes to complete
        server_process.wait()
        client1_process.wait()
        client2_process.wait()

    except Exception as e:
        logging.error(f"❌ Error occurred: {e}", exc_info=True)

    finally:
        logging.info("🛑 Stopping all processes...")
        for process in [server_process, client1_process, client2_process]:
            if process.poll() is None:  # If process is still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

if __name__ == "__main__":
    main()
