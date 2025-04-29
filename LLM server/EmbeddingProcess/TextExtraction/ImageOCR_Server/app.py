from flask import Flask, request, jsonify
from PIL import Image
from deepseek_vl import inference
from ModelManager import ModelManager

app = Flask(__name__)
manager = ModelManager(timeout=600)  # 10 minutes timeout

@app.route("/process", methods=["POST"])
def process_image():
    if 'image' not in request.files or 'message' not in request.form:
        return jsonify({'error': 'Image file and message are required'}), 400

    image_file = request.files['image']
    message = request.form['message']
    image_name = image_file.filename

    try:
        image = Image.open(image_file.stream).convert("RGB")
        vl_chat_processor, vl_gpt, tokenizer = manager.get_model()

        response = inference(
            message=message,
            images_names=[image_name],
            pil_images=[image],
            vl_chat_processor=vl_chat_processor,
            vl_gpt=vl_gpt,
            tokenizer=tokenizer
        )

        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/unload", methods=["POST"])
def unload_model():
    manager.force_unload()
    return jsonify({'status': 'Model unloaded manually.'})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
