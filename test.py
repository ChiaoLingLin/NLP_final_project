import json
import os
import sys
import argparse
import time
from google import genai # 确保您的环境中安装了 google-genai SDK
# from google.generativeai.errors import APIError # 导入 APIError 用于更精确的错误处理

# --- 1. 餐厅领域类别定义 ---
restaurant_entity_labels = ['RESTAURANT', 'FOOD', 'DRINKS', 'AMBIENCE', 'SERVICE', 'LOCATION']
restaurant_attribute_labels = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE_OPTIONS', 'MISCELLANEOUS']

def combine_lists(list1, list2):
    combined_list = list1 + list2
    category_dict = {category: i for i, category in enumerate(combined_list)}
    return category_dict, combined_list

restaurant_category_dict, restaurant_category_list = combine_lists(restaurant_entity_labels, restaurant_attribute_labels)
CATEGORY_LIST = restaurant_category_list 

# --- 2. 配置和 Argparse ---

def parser_getting():
    parser = argparse.ArgumentParser(description='DimABSA inference for Task 3 in Chinese.')
    # 更改默认输入文件名为您的数据文件
    parser.add_argument('--data_path', type=str, default="./dataset/", help="Base path for data files.")
    parser.add_argument('--infer_data', type=str, default="zho_restaurant_dev_task3.jsonl", help="Inference data file name.")
    parser.add_argument('--output_path', type=str, default="./tasks/")
    # 新增参数，用于限制推论数量，默认为 None (无限制)
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of samples to process for testing.")
    
    args = parser.parse_args()
    return args

# 映射输出文件名
OUT_PUT_FILE_NAME_MAP = {
    'res_zho': "pred_zho_restaurant.jsonl", # 针对餐厅领域的输出文件名
}
# 设置一个全局变量或在 run_inference 中传递文件名
OUTPUT_FILENAME = OUT_PUT_FILE_NAME_MAP['res_zho']


# --- 3. 辅助函数 ---

def load_inference_data(args):
    """
    读取指定路径的 JSONL 文件。
    """
    # 确保文件路径正确地使用了 args.infer_data
    file_path = os.path.join(args.data_path, args.infer_data)
    inference_datasets = []
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    # 修正：您的 JSONL 文件是每行一个完整的 JSON 对象，不需要 find('{')
                    data = json.loads(line.strip())

                except json.JSONDecodeError:
                    print(f"Skipping malformed line (JSON Decode Error): {line.strip()}", file=sys.stderr)
                    continue

                data_id = data.get('ID')
                text = data.get('Text')
                if data_id and text:
                    inference_datasets.append({'ID': data_id, 'Text': text})
    except FileNotFoundError:
        # 如果文件未找到，尝试在当前目录下查找 (针对您提供的文件)
        try:
             with open(args.infer_data, 'r', encoding='utf-8') as f:
                print(f"Loading data from local file: {args.infer_data}...")
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        print(f"Skipping malformed line (JSON Decode Error): {line.strip()}", file=sys.stderr)
                        continue

                    data_id = data.get('ID')
                    text = data.get('Text')
                    if data_id and text:
                        inference_datasets.append({'ID': data_id, 'Text': text})
        except FileNotFoundError:
            print(f"Error: Input file {file_path} or {args.infer_data} not found.", file=sys.stderr)
            sys.exit(1)
        
    print(f"Loaded {len(inference_datasets)} samples.")
    return inference_datasets

def format_quadruplet_output(data_id, quadruplets):
    """
    格式化 Task 3 的输出为指定的 JSONL 格式 (Quadruplet)，
    确保 VA 字段符合 "Valence#Arousal" 字符串格式。
    """
    dump_data_quadra = {
        "ID": data_id,
        "Quadruplet": []
    }
    
    for quad in quadruplets:
        meta_quadra = {}
        meta_quadra["Aspect"] = quad.get("Aspect", "")
        meta_quadra["Category"] = quad.get("Category", "")
        meta_quadra["Opinion"] = quad.get("Opinion", "")
        
        # 确保 Valence 和 Arousal 是浮点数并格式化
        try:
            valence = float(quad.get("Valence", 0.0))
            arousal = float(quad.get("Arousal", 0.0))
            # 格式化为两位小数的字符串
            meta_quadra["VA"] = f"{valence:.2f}#{arousal:.2f}" 
        except (ValueError, TypeError):
            # 处理非数字情况
            meta_quadra["VA"] = "0.00#0.00"

        dump_data_quadra['Quadruplet'].append(meta_quadra)

    return dump_data_quadra

def construct_gemini_prompt(text, categories):
    """
    构建用于 Gemini 推论的 Prompt。
    """
    categories_str = ", ".join(categories)
    
    prompt = f"""
    您是一位情感分析專家，請從提供的中文餐廳評論中提取所有相關的「情感四元組」(Quadruplet)。
    四元組包含：**方面詞(Aspect)**、**意見詞(Opinion)**、**方面類別(Category)** 和 **情價-喚醒度(Valence-Arousal)**。

    **Task 3 定義：** 提取 <方面詞, 意見詞, 方面類別, 情價-喚醒度>

    **評論文本 (Text):** "{text}"

    **可用的方面類別 (Category List):** {categories_str}
    
    **情價-喚醒度 (Valence-Arousal) 說明:**
    - 情價 (Valence): 評分範圍為 1.0 (非常負面) 到 9.0 (非常正面)。
    - 喚醒度 (Arousal): 評分範圍為 1.0 (非常平靜) 到 9.0 (非常激動)。

    **請遵守以下規則：**
    1. 您的回應必須且只能是一個 **JSON 列表**，不包含任何額外解釋或文本。
    2. 如果文本中沒有找到任何四元組，請返回一個空列表：`[]`。
    3. Category 必須是 'Category List' 中的一個。
    4. Valence 和 Arousal 必須是 1.0 到 9.0 之間的**數字類型**。

    **範例(golden arousal example)**
    {{"ID": "R0623:S009", "Text": "咖哩一上桌超級香阿，香料味道很豐富且溫和。", "Quadruplet": [{{"Aspect": "咖哩", "Category": "FOOD#QUALITY", "Opinion": "超級香", "Valence": 8.00, "Arousal": 8.25}}, {{"Aspect": "香料味道", "Category": "FOOD#QUALITY", "Opinion": "很豐富", "Valence": 7.75, "Arousal": 7.75}}, {{"Aspect": "香料味道", "Category": "FOOD#QUALITY", "Opinion": "溫和", "Valence": 6.75, "Arousal": 6.25}}]}}
    {{"ID": "R3012:S001", "Text": "主餐豬排份量非常多CP值很高。", "Quadruplet": [{{"Aspect": "豬排份量", "Category": "FOOD#STYLE_OPTIONS", "Opinion": "非常多", "Valence": 6.83, "Arousal": 6.33}}, {{"Aspect": "CP值", "Category": "FOOD#PRICES", "Opinion": "很高", "Valence": 6.83, "Arousal": 6.17}}]}}
    {{"ID": "R1970:S037", "Text": "最後就是很不錯的餐廳，雖然價格偏高了點但味道上都還蠻不錯。", "Quadruplet": [{{"Aspect": "餐廳", "Category": "RESTAURANT#GENERAL", "Opinion": "很不錯", "Valence": 6.50, "Arousal": 6.00}}, {{"Aspect": "價格", "Category": "RESTAURANT#PRICES", "Opinion": "偏高", "Valence": 4.50, "Arousal": 4.00}}, {{"Aspect": "味道", "Category": "FOOD#QUALITY", "Opinion": "蠻不錯", "Valence": 6.00, "Arousal": 5.50}}]}}
    {{"ID": "R1673:S007", "Text": "餐廳略顯服務不夠周到。", "Quadruplet": [{{"Aspect": "服務", "Category": "SERVICE#GENERAL", "Opinion": "不夠周到", "Valence": 4.50, "Arousal": 5.25}}]}}
    {{"ID": "R2848:S013", "Text": "牛肉口感還可以，給的份量也挺多。", "Quadruplet": [{{"Aspect": "牛肉口感", "Category": "FOOD#QUALITY", "Opinion": "還可以", "Valence": 5.50, "Arousal": 5.33}}, {{"Aspect": "份量", "Category": "FOOD#STYLE_OPTIONS", "Opinion": "挺多", "Valence": 6.00, "Arousal": 5.67}}]}}
    
    **範例(golden valence example)**
    {{"ID": "R2910:S007", "Text": "小卷麵線及小卷米粉都不錯很鮮甜，不過有點鹹。", "Quadruplet": [{{"Aspect": "小卷麵線", "Category": "FOOD#QUALITY", "Opinion": "不錯", "Valence": 5.50, "Arousal": 5.50}}, {{"Aspect": "小卷米粉", "Category": "FOOD#QUALITY", "Opinion": "不錯", "Valence": 5.50, "Arousal": 5.50}}, {{"Aspect": "小卷麵線", "Category": "FOOD#QUALITY", "Opinion": "很鮮甜", "Valence": 6.50, "Arousal": 6.25}}, {{"Aspect": "小卷米粉", "Category": "FOOD#QUALITY", "Opinion": "很鮮甜", "Valence": 6.50, "Arousal": 6.25}}, {{"Aspect": "小卷麵線", "Category": "FOOD#QUALITY", "Opinion": "有點鹹", "Valence": 4.50, "Arousal": 5.00}}, {{"Aspect": "小卷米粉", "Category": "FOOD#QUALITY", "Opinion": "有點鹹", "Valence": 4.50, "Arousal": 5.00}}]}}
    {{"ID": "R1215:S060", "Text": "麵條也相當有嚼勁。", "Quadruplet": [{{"Aspect": "麵條", "Category": "FOOD#QUALITY", "Opinion": "相當有嚼勁", "Valence": 6.50, "Arousal": 6.00}}]}}
    {{"ID": "R1490:S003", "Text": "蔥爆豬肉整體是好吃。", "Quadruplet": [{{"Aspect": "蔥爆豬肉", "Category": "FOOD#QUALITY", "Opinion": "好吃", "Valence": 6.10, "Arousal": 5.80}}]}}
    {{"ID": "R0032:S034", "Text": "蜂蜜紅茶則非常難喝，有股臭味，而且又超貴。", "Quadruplet": [{{"Aspect": "蜂蜜紅茶", "Category": "DRINKS#QUALITY", "Opinion": "非常難喝", "Valence": 2.00, "Arousal": 7.00}}, {{"Aspect": "蜂蜜紅茶", "Category": "DRINKS#QUALITY", "Opinion": "有股臭味", "Valence": 1.50, "Arousal": 6.50}}, {{"Aspect": "蜂蜜紅茶", "Category": "DRINKS#PRICES", "Opinion": "超貴", "Valence": 3.25, "Arousal": 6.75}}]}}
    {{"ID": "R0093:S028", "Text": "炸鱈魚白子與白子蒸蛋獲在場評價最高。", "Quadruplet": [{{"Aspect": "炸鱈魚白子", "Category": "FOOD#QUALITY", "Opinion": "評價最高", "Valence": 8.25, "Arousal": 7.50}}, {{"Aspect": "白子蒸蛋", "Category": "FOOD#QUALITY", "Opinion": "評價最高", "Valence": 8.25, "Arousal": 7.50}}]}}

    **請以純粹的 JSON 格式回應，且僅包含 JSON 列表，勿加入任何解釋或額外的文字。**
    **回應格式 (JSON Array):**
    [
      {{"Aspect": "...", "Opinion": "...", "Category": "...", "Valence": 4.5, "Arousal": 3.0}},
      ...
    ]
    """
    return prompt

# --- 4. Gemini 呼叫和推论函数 ---

def run_inference(args, inference_data, output_filename, limit=None): # 添加 limit 參數
    """
    对每个数据样本运行推论并保存结果。
    """
    output_data_quadra = []
    
    # 初始化 Gemini Client
    try:
        client = genai.Client()
    except Exception as e:
        print(f"Error initializing Gemini Client: {e}", file=sys.stderr)
        print("Please ensure your API key is set correctly in your environment (GEMINI_API_KEY).", file=sys.stderr)
        return # 提前退出

    model_name = 'gemini-2.5-flash'
    print(f"Starting inference with model: {model_name}...")

    for i, data in enumerate(inference_data):
        # 檢查是否達到限制
        if limit is not None and i >= limit:
            print(f"Limit of {limit} samples reached. Stopping inference.")
            break 
            
        data_id = data['ID']
        text = data['Text']
        
        print(f"\n--- Processing {i + 1}/{limit} (ID: {data_id}) ---")
        
        # 1. 构建 Prompt
        prompt = construct_gemini_prompt(text, CATEGORY_LIST)
        
        # 2. 呼叫 Gemini API
        gemini_response_text = ""
        
        try:
            # TODO: 您需要確保您的 API 金鑰已設定，並能夠成功呼叫 API
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            gemini_response_text = response.text

            # 3. 解析模型回應
            quadruplets_raw = []
            try:
                json_text = gemini_response_text.strip()
                # 打印模型原始回應 (用於調試)
                print(f"Gemini Raw Response: \n{json_text}") 
                
                # 移除 markdown code block markers
                if json_text.startswith("```json"):
                    json_text = json_text[7:].strip()
                if json_text.endswith("```"):
                    json_text = json_text[:-3].strip()
                
                if json_text:
                    quadruplets_raw = json.loads(json_text)

            except Exception as e:
                print(f"Error: Failed to decode JSON for ID: {data_id}. Response: {gemini_response_text[:50]}... Error: {e}", file=sys.stderr)
                quadruplets_raw = []

            # 4. 格式化輸出
            formatted_quadra = format_quadruplet_output(data_id, quadruplets_raw)
            output_data_quadra.append(formatted_quadra)
            # 打印格式化後的結果 (用於檢查格式)
            print(f"Formatted Output: {json.dumps(formatted_quadra, ensure_ascii=False)}")


        # except APIError as e: # 如果您使用了 from google.generativeai.errors import APIError
        #     print(f"API Error during API call for ID {data_id}: {e}", file=sys.stderr)
        #     output_data_quadra.append(format_quadruplet_output(data_id, []))
        except Exception as e:
            print(f"Critical error during API call for ID {data_id}: {e}", file=sys.stderr)
            output_data_quadra.append(format_quadruplet_output(data_id, []))

        # 稍微等待一下，避免速率限制問題
        time.sleep(3) 
            
    # 5. 保存結果
    output_dir = os.path.join(args.output_path, "subtask_3")
    os.makedirs(output_dir, exist_ok=True)
    out_put_file_task3_name = os.path.join(output_dir, output_filename)
    
    # 針對測試模式，更改輸出文件名以區分
    if limit is not None:
        out_put_file_task3_name = os.path.join(output_dir, f"test_limit_{limit}_{output_filename}")


    print(f"\nWriting results to {out_put_file_task3_name}...")
    with open(out_put_file_task3_name, 'w', encoding='utf-8') as f:
        for item in output_data_quadra:
            # 確保輸出是單行 JSON
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + '\n')
            
    print(f"Inference completed. Results saved to {out_put_file_task3_name}")

# --- 5. 主程序入口 ---

if __name__ == '__main__':
    args = parser_getting()

    # 显式地设置输入和输出文件名
    args.infer_data = "zho_restaurant_dev_task3.jsonl"
    output_filename = "pred_zho_restaurant.jsonl"
    args.output_path = "./tasks/" # 确保默认路径有 subtask_3 子文件夹

    # --- 设置测试限制为 3 筆 ---
    test_limit = 500 
    
    # 嘗試加载数据
    inference_data = load_inference_data(args)
        
    if not inference_data:
        print("No data loaded. Exiting.", file=sys.stderr)
        sys.exit(0)
    
    # 運行推論 (限制為 3 筆)
    run_inference(args, inference_data, output_filename, limit=test_limit)