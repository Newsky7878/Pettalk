from flask import Flask, request

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_audio():
    """ 处理 ESP32 上传的音频 """
    try:
        with open("received_audio.wav", "wb") as f:
            f.write(request.data)
        return "Upload Successful", 200
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
