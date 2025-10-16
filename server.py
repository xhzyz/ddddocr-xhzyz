import asyncio
import base64
import logging
from io import BytesIO

import cv2
import ddddocr
import httpx
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- App and Logging Configuration ---
# 配置日志记录
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

app = FastAPI(
    title="Async CAPTCHA Solver API",
    description="An asynchronous API for solving various types of CAPTCHAs using ddddocr.",
    version="2.0.0"
)


# --- Pydantic Models for Request Bodies ---
# 使用Pydantic模型进行数据校验和自动文档生成
class SlideCaptchaRequest(BaseModel):
    slidingImage: str = Field(..., description="Base64 encoded or URL of the sliding block image.")
    backImage: str = Field(..., description="Base64 encoded or URL of the background image.")
    simpleTarget: bool = Field(True, description="Whether to use the simple target method.")

class ClassificationRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded or URL of the image for OCR classification.")

class DetectionRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded or URL of the image for object detection.")

class CalculateRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded or URL of the calculation CAPTCHA image.")

class CropRequest(BaseModel):
    image: str = Field(..., description="URL of the image to be cropped.")
    y_coordinate: int = Field(..., description="The y-coordinate to split the image.")

class SelectRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded or URL of the image for point selection.")


# --- Asynchronous Image Fetching ---
# 异步获取图片字节流
async def get_image_bytes(image_data: str, client: httpx.AsyncClient) -> bytes:
    """Asynchronously gets image bytes from a URL or decodes a base64 string."""
    if image_data.startswith(('http://', 'https://')):
        try:
            response = await client.get(image_data)
            response.raise_for_status()
            return response.content
        except httpx.RequestError as e:
            logging.error(f"HTTP request error fetching image: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {e}")
    else:
        try:
            # 补全可能缺失的padding
            padding = '=' * (4 - len(image_data) % 4)
            return base64.b64decode(image_data + padding)
        except (ValueError, TypeError) as e:
            logging.error(f"Base64 decode error: {e}")
            raise HTTPException(status_code=400, detail="Invalid base64 image data.")

def image_to_base64(image: Image.Image, format='PNG') -> str:
    """Converts a PIL Image to a base64 encoded string."""
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def preprocess_image(img: Image.Image, is_arithmetic=False) -> Image.Image:
    img = img.convert("L")
    img_array = np.array(img)

    if is_arithmetic:
        # 提升对比度
        img_array = cv2.convertScaleAbs(img_array, alpha=1.5, beta=0)
        # CLAHE 局部对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_array = clahe.apply(img_array)
        # 轻微模糊去噪
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        # 自适应阈值（字符白，背景黑）
        img_array = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )
        # 形态学修复断笔
        kernel = np.ones((3, 3), np.uint8)
        img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel, iterations=1)
        kernel_small = np.ones((2, 2), np.uint8)
        img_array = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel_small, iterations=1)
    else:
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(img_array)

def classify_text(img: bytes) -> bytes:
    img = Image.open(BytesIO(img)) if isinstance(img, bytes) else img
    processed_img = preprocess_image(img, is_arithmetic=False)
    img_bytes = BytesIO()
    processed_img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()
    return img_bytes

# --- CAPTCHA Solver Class ---
class CAPTCHA:
    def __init__(self):
        self.ocr = ddddocr.DdddOcr(show_ad=False)
        self.det = ddddocr.DdddOcr(det=True, show_ad=False)

    async def _run_in_thread(self, func, *args):
        """Runs a synchronous, CPU-bound function in a separate thread."""
        try:
            # asyncio.to_thread 在 Python 3.9+ 中可用
            return await asyncio.to_thread(func, *args)
        except Exception as e:
            logging.error(f"Error in threaded execution of {func.__name__}: {e}")
            # 返回None或根据需要抛出特定异常
            return None
            
    # --- CAPTCHA Solving Methods ---

    async def capcode(self, sliding_bytes: bytes, back_bytes: bytes, simple_target: bool):
        res = await self._run_in_thread(self.ocr.slide_match, sliding_bytes, back_bytes, simple_target)
        return res['target'][0] if res else None

    async def slideComparison(self, sliding_bytes: bytes, back_bytes: bytes):
        res = await self._run_in_thread(self.ocr.slide_comparison, sliding_bytes, back_bytes)
        return res['target'][0] if res else None

    async def classification(self, image_bytes: bytes):
        return await self._run_in_thread(self.ocr.classification, image_bytes)

    async def detection(self, image_bytes: bytes):
        return await self._run_in_thread(self.det.detection, image_bytes)
    
    async def calculate(self, image_bytes: bytes):
        expression = await self.classification(image_bytes)
        if not expression:
            return None
        # 使用更安全的方式清理和计算表达式
        cleaned_expression = ''.join(filter(lambda char: char in '0123456789+-*/().', expression.split('=')[0]))
        try:
            # 注意: eval() 有安全风险。在生产环境中，请考虑使用更安全的解析器，如 ast.literal_eval 或第三方库。
            # 这里我们信任ddddocr的输出是良性的。
            return eval(cleaned_expression)
        except (SyntaxError, NameError, ZeroDivisionError) as e:
            logging.error(f"Failed to evaluate expression '{cleaned_expression}': {e}")
            return None

    async def crop(self, image_bytes: bytes, y_coordinate: int):
        try:
            image = Image.open(BytesIO(image_bytes))
            if y_coordinate <= 0 or y_coordinate * 2 >= image.height:
                raise ValueError("y_coordinate is out of bounds for cropping.")

            upper_half = image.crop((0, 0, image.width, y_coordinate))
            # middle_half = image.crop((0, y_coordinate, image.width, y_coordinate*2)) # 原代码中未使用
            lower_half = image.crop((0, y_coordinate*2, image.width, image.height))

            slidingImage_b64 = image_to_base64(upper_half)
            backImage_b64 = image_to_base64(lower_half)
            
            return {'slidingImage': slidingImage_b64, 'backImage': backImage_b64}
        except Exception as e:
            logging.error(f"Image cropping error: {e}")
            return None

    async def select(self, image_bytes: bytes):
        bboxes = await self.detection(image_bytes)
        if not bboxes:
            return []

        im = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        results = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            cropped_image = im[y1:y2, x1:x2]
            
            _, buffer = cv2.imencode('.png', cropped_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 由于 self.classification 也是异步的，需要 await
            text_result = await self.classification(base64.b64decode(image_base64))
            if text_result:
                results.append({text_result: bbox})
        
        return results

# --- API Endpoints ---

captcha_solver = CAPTCHA()

@app.get("/")
async def root():
    return {"message": "API is running successfully!"}

@app.post("/capcode")
async def handle_capcode(data: SlideCaptchaRequest):
    async with httpx.AsyncClient() as client:
        sliding_bytes = await get_image_bytes(data.slidingImage, client)
        back_bytes = await get_image_bytes(data.backImage, client)
    
    result = await captcha_solver.capcode(sliding_bytes, back_bytes, data.simpleTarget)
    if result is None:
        raise HTTPException(status_code=500, detail="Error during capcode processing.")
    return {"result": result}

@app.post("/slideComparison")
async def handle_slide_comparison(data: SlideCaptchaRequest):
    async with httpx.AsyncClient() as client:
        sliding_bytes = await get_image_bytes(data.slidingImage, client)
        back_bytes = await get_image_bytes(data.backImage, client)
    
    result = await captcha_solver.slideComparison(sliding_bytes, back_bytes)
    if result is None:
        raise HTTPException(status_code=500, detail="Error during slide comparison processing.")
    return {"result": result}

@app.post("/classification")
async def handle_classification(data: ClassificationRequest):
    async with httpx.AsyncClient() as client:
        image_bytes = await get_image_bytes(data.image, client)
    
    result = await captcha_solver.classification(image_bytes)
    if result is None:
        raise HTTPException(status_code=500, detail="Error during classification processing.")
    return {"result": result}

@app.post("/detection")
async def handle_detection(data: DetectionRequest):
    async with httpx.AsyncClient() as client:
        image_bytes = await get_image_bytes(data.image, client)
    
    result = await captcha_solver.detection(image_bytes)
    if result is None:
        raise HTTPException(status_code=500, detail="Error during detection processing.")
    return {"result": result}

@app.post("/calculate")
async def handle_calculate(data: CalculateRequest):
    async with httpx.AsyncClient() as client:
        image_bytes = await get_image_bytes(data.image, client)

    result = await captcha_solver.calculate(image_bytes)
    if result is None:
        raise HTTPException(status_code=500, detail="Error during calculation processing.")
    return {"result": result}

@app.post("/crop")
async def handle_crop(data: CropRequest):
    async with httpx.AsyncClient() as client:
        image_bytes = await get_image_bytes(data.image, client)

    result = await captcha_solver.crop(image_bytes, data.y_coordinate)
    if result is None:
        raise HTTPException(status_code=500, detail="Error during image cropping.")
    return result

@app.post("/select")
async def handle_select(data: SelectRequest):
    async with httpx.AsyncClient() as client:
        image_bytes = await get_image_bytes(data.image, client)
    
    result = await captcha_solver.select(image_bytes)
    if result is None: # select方法在出错时可能返回None
        raise HTTPException(status_code=500, detail="Error during select processing.")
    return {"result": result}


@app.post("/classification/beta1")
async def handle_classification_beta(data: ClassificationRequest):
    async with httpx.AsyncClient() as client:
        image_bytes = await get_image_bytes(data.image, client)
    image_bytes = classify_text(image_bytes)
    result = await captcha_solver.classification(image_bytes)
    if result is None:
        raise HTTPException(status_code=500, detail="Error during classification processing.")
    return {"result": result}

# --- How to Run ---
# To run this application, save it as main.py and use an ASGI server like Uvicorn.
# Command: uvicorn main:app --host 0.0.0.0 --port 7777 --reload
if __name__ == "__main__":
    import uvicorn
    # 允许在脚本直接运行时启动服务，主要用于调试
    uvicorn.run(app, host="0.0.0.0", port=7777)