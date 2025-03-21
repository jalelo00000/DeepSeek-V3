import random
import string
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# 🔑 إنشاء مفتاح API عشوائي قصير (5 أحرف)
API_KEY = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
print(f"🔑 API Key: {API_KEY}")  # طباعة المفتاح لاستخدامه لاحقًا

# تحميل النموذج من المستودع
model_name = "deepseek-ai/deepseek-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 📌 تعريف نموذج البيانات للطلب
class RequestData(BaseModel):
    prompt: str

# 📌 صفحة HTML مع JavaScript مدمج
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>واجهة الذكاء الاصطناعي</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; padding: 20px; }}
            input, button {{ padding: 10px; margin: 10px; width: 80%; max-width: 400px; }}
            #response {{ margin-top: 20px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>جرب نموذج الذكاء الاصطناعي</h2>
        <input type="text" id="userInput" placeholder="اكتب نصًا هنا..." />
        <button onclick="sendRequest()">إرسال</button>
        <p id="response"></p>

        <script>
            function sendRequest() {{
                let userInput = document.getElementById("userInput").value;
                fetch("/generate", {{
                    method: "POST",
                    headers: {{
                        "Content-Type": "application/json",
                        "api_key": "{API_KEY}"  // استخدم مفتاح API الذي تم توليده
                    }},
                    body: JSON.stringify({{ prompt: userInput }})
                }})
                .then(response => response.json())
                .then(data => {{
                    document.getElementById("response").innerText = "الإجابة: " + data.response;
                }})
                .catch(error => console.error("خطأ:", error));
            }}
        </script>
    </body>
    </html>
    """
    return html_content

# 📌 نقطة API لتوليد النصوص
@app.post("/generate")
async def generate_text(request: RequestData, api_key: str = Header(None)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="🚫 مفتاح API غير صحيح")

    # تحويل الإدخال إلى نموذج الذكاء الاصطناعي
    inputs = tokenizer(request.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"response": response_text}
