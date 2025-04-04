from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from accelerate import Accelerator
from transformers import BitsAndBytesConfig
from accelerate.utils import get_max_memory
from PIL import Image
from flask import Flask, request, jsonify
import base64
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import logging
from io import BytesIO


def load_model_with_memory_split(model_name="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", 
                               max_gpu_memory="10GiB"):
 
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Get the maximum memory available
    max_memory = get_max_memory()
    
    # Modify the GPU memory constraint
    if torch.cuda.is_available():
        gpu_id = 0  # Assuming using first GPU
        max_memory[gpu_id] = max_gpu_memory
        # Explicitly set remaining memory to CPU
        max_memory["cpu"] = "11GiB"  # Adjust based on your available RAM
    
    print(f"Memory map configuration: {max_memory}")
    
    # Load the model with specific device mapping

    #
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
        offload_folder="offload",  # Folder for CPU offloading
        offload_state_dict=True    # Enable state dict offloading
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)
    
    return model, processor

# passed in base64 encoded image
# uses default prompt 
# returns output from model
def model_inference(b64data):
    global loaded
    global model, processor
    if(loaded==False):
        # You can adjust max_gpu_memory based on your GPU
        model_name="meta-llama/Llama-3.2-11B-Vision-Instruct"
        model_name="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
        model, processor = load_model_with_memory_split(model_name=model_name, max_gpu_memory="13GiB")
        model.tie_weights()
        print("Model loaded successfully")
        print(f"Model device map: {model.hf_device_map}")
        loaded =True
    b64data = str.encode(b64data) # string to b64
    b64decoded = base64.b64decode(b64data) #b64encoded to b64 decoded
    img = Image.open(BytesIO(b64decoded))
    
    text= """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
**Instructions:**

Please analyze the given image and perform the following tasks:

1. **Description**:
   - Provide a concise yet comprehensive description of the image.
   - Focus on major subjects, objects, and significant details.
   - Ignore minor features and do not describe emotions or feelings.
   - The description will be saved to support future searches.

2. **Entity Extraction (NER)**:
   - Identify and list all significant entities mentioned in your description.
   - For each entity, specify the entity type using the following categories:
     - **Person**: Individual people or groups of people.
     - **Organization**: Companies, institutions, agencies, etc.
     - **Location**: Geographical places like cities, countries, landmarks.
     - **Product**: Physical items, devices, software, etc.
     - **Event**: Occasions or happenings, such as meetings, concerts.
     - **Concept**: Abstract ideas, subjects, or fields of study.
     - **Artwork**: Titles of books, movies, paintings, etc.
     - **Other**: Any significant entity not covered by the above categories.
   - Present the entities in a JSON format as shown below.

**Output Format:**

```
Description:
[Your image description here]

Entities:
{
  "entities": [
    {"text": "Entity1", "type": "EntityType1"},
    {"text": "Entity2", "type": "EntityType2"},
    {"text": "Entity3", "type": "EntityType3"}
  ]
}
```

**Example:**

*Description:*
A group of medical researchers are analyzing data on their computers in a laboratory. The logo of the World Health Organization is displayed on a screen. Test tubes and lab equipment are visible on the tables.

*Entities:*
```json
{
  "entities": [
    {"text": "medical researchers", "type": "Person"},
    {"text": "data", "type": "Concept"},
    {"text": "computers", "type": "Product"},
    {"text": "laboratory", "type": "Location"},
    {"text": "World Health Organization", "type": "Organization"},
    {"text": "screen", "type": "Product"},
    {"text": "test tubes", "type": "Product"},
    {"text": "lab equipment", "type": "Product"},
    {"text": "tables", "type": "Product"}
  ]
}
```

**Additional Notes:**

- **Accuracy:** Ensure that all entities are directly observable from the image description and avoid any assumptions beyond the visible content.
- **Formatting:** Make sure the JSON is correctly formatted and valid.
- **Clarity:** Do not include any additional commentary or information beyond what is requested.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


    text= """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a medical transcription employee and
your job is to identify data in medical images.  You do not prescribe or dispense medications.  You reliably gather information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
This is an image of a medical prescription.  Return any of the following data:
Name of person
Name of doctor
Name of medication
Dosage
Condition

reconsider your results to improve accuracy.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    with torch.no_grad():
        inputs = processor(images=[img, img],  return_tensors="pt")
        text_embedding = processor(text=text, return_tensors='pt')
        inputs.update(text_embedding)
        # Move inputs to the same device as their corresponding model parts
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.25)
        return processor.decode(outputs[0], skip_special_tokens=True)

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    print('starting upload')
    try:
        print('here')
        # Check if the post request has the image field
        if 'image' not in request.json:
            return jsonify({'error': 'No image data provided'}), 400
        print('htere')
        
        base64_data = request.json['image']
        print("the type I got is ", type(base64_data))
        the_text = model_inference(base64_data)

        try:
            # Remove potential data URL prefix
            if 'base64,' in base64_data:
                base64_data = base64_data.split('base64,')[1]
            
            # Decode the base64 string
            image_data = base64.b64decode(base64_data)
        except Exception as e:
            print(e)
            logger.error(f"Base64 decoding error: {str(e)}")
            return jsonify({'error': 'Invalid base64 data'}), 400

        # Generate a unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'image_{timestamp}.jpg'  # Default to .jpg, adjust as needed
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))

        # Save the image
        with open(filepath, 'wb') as f:
            f.write(image_data)

        logger.info(f"Successfully saved image: {filepath}")

        return jsonify({
            'message': 'Image uploaded successfully',
            'filename': filename,
            'text' : the_text
        }), 200

    except Exception as e:
        print("whoops!")
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    loaded = False
    model = None
    processor = None
    # Configure maximum content length (optional)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
    
    # Run the server
    app.run(host='0.0.0.0', port=8896, debug=False)

# example
# curl -X POST -H "Content-Type: application/json" -d '{"your_json_data": "here"}' <url>
# curl -X POST -H "Content-Type: application/json" -d '{"image":"aGVsbG8="}' http://sdc:5000/upload


"""  Motivation 
A local business has a video camera at the entrance to their store aimed outward toward the street and parking lot with a good view of the adjacent public park.    This is a popular location with pedestrian foot traffic and light suburban vehicle traffic.  The video feed is made available to a highly capable LLM  Vision Instruct model.    Video frames are captured once per second 24/7.    Each frame is passed to the vision model with the prompt "Describe in detail what you see in this image."    The text results for each frame are saved for future aggregation.

One application would be "outlier detection", but this presents an interesting problem.   What is an outlier?    Suppose the system is primed with a few hours of images and there was very mundane activity.   Suppose then a large tractor trailer drives through the scene.   this will be the first time the model has seen that type of vehicle.   Is that an outlier?    Certainly a motor vehicle accident or a house fire would be an outlier.   

Can you suggest a strategy such that the aggregate text  can incrementally  queried for outliers or interesting events using LLM prompts?   """
