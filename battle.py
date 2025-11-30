import json
import os
import sys
import argparse
import time
from google import genai 

# --- 1. 筆電領域類別定義 (已修改變數名稱) ---
laptop_entity_labels = ['LAPTOP', 'DISPLAY', 'KEYBOARD', 'MOUSE', 'MOTHERBOARD', 'CPU', 'FANS_COOLING', 'PORTS', 'MEMORY', 'POWER_SUPPLY', 'OPTICAL_DRIVES', 'BATTERY', 'GRAPHICS', 'HARD_DISK', 'MULTIMEDIA_DEVICES', 'HARDWARE', 'SOFTWARE', 'OS', 'WARRANTY', 'SHIPPING', 'SUPPORT', 'COMPANY']
laptop_attribute_labels = ['GENERAL', 'PRICE', 'QUALITY', 'DESIGN_FEATURES', 'OPERATION_PERFORMANCE', 'USABILITY', 'PORTABILITY', 'CONNECTIVITY', 'MISCELLANEOUS']

def combine_lists(list1, list2):
    combined_list = list1 + list2
    category_dict = {category: i for i, category in enumerate(combined_list)}
    return category_dict, combined_list

# 修改：將變數名稱從 restaurant 改為 laptop
laptop_category_dict, laptop_category_list = combine_lists(laptop_entity_labels, laptop_attribute_labels)
CATEGORY_LIST = laptop_category_list 

# --- 2. 配置和 Argparse ---

def parser_getting():
    parser = argparse.ArgumentParser(description='DimABSA inference for Task 3 in Chinese.')
    parser.add_argument('--data_path', type=str, default="./dataset/", help="Base path for data files.")
    parser.add_argument('--infer_data', type=str, default="zho_laptop_dev_task3.jsonl", help="Inference data file name.") # 預設改為 laptop
    parser.add_argument('--output_path', type=str, default="./tasks/")
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of samples to process for testing.")
    
    args = parser.parse_args()
    return args

# 映射輸出文件名
OUT_PUT_FILE_NAME_MAP = {
    'res_zho': "pred_zho_laptop.jsonl", # 針對筆電領域的輸出文件名
}
OUTPUT_FILENAME = OUT_PUT_FILE_NAME_MAP['res_zho']


# --- 3. 辅助函数 ---

def load_inference_data(args):
    """
    讀取指定路徑的 JSONL 文件。
    """
    file_path = os.path.join(args.data_path, args.infer_data)
    inference_datasets = []
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
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
    格式化 Task 3 的輸出為指定的 JSONL 格式 (Quadruplet)
    """
    dump_data_quadra = {
        "ID": data_id,
        "Quadruplet": []
    }
    
    # 增加防禦性程式碼，如果回傳不是 list 則跳過
    if not isinstance(quadruplets, list):
        return dump_data_quadra

    for quad in quadruplets:
        if not isinstance(quad, dict): continue # 確保每個元素是 dict

        meta_quadra = {}
        meta_quadra["Aspect"] = quad.get("Aspect", "")
        meta_quadra["Category"] = quad.get("Category", "")
        meta_quadra["Opinion"] = quad.get("Opinion", "")
        
        try:
            valence = float(quad.get("Valence", 0.0))
            arousal = float(quad.get("Arousal", 0.0))
            meta_quadra["VA"] = f"{valence:.2f}#{arousal:.2f}" 
        except (ValueError, TypeError):
            meta_quadra["VA"] = "0.00#0.00"

        dump_data_quadra['Quadruplet'].append(meta_quadra)

    return dump_data_quadra

def construct_gemini_prompt(text, categories):
    """
    構建用於 Gemini 推論的 Prompt。
    修正重點：
    1. 將 '餐廳' 改為 '筆記型電腦'。
    2. f-string 中所有的 JSON 範例大括號 {} 都改為雙大括號 {{}} 以避免 Python 語法錯誤。
    """
    categories_str = ", ".join(categories)
    
    # 注意：下面的 prompt 使用 f-string，因此所有的 JSON 範例括號都必須雙寫 {{...}}
    prompt = f"""
    您是一位情感分析專家，請從提供的中文筆記型電腦評論中提取所有相關的「情感四元組」(Quadruplet)。
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
    {{"ID": "6760124:S484", "Text": "上前代的RTX3050筆電GPU算是比較可惜的地方，若能導入新一代的RTX4050筆電GPU，則會讓這款筆電更具入手的價值", "Quadruplet": [{{"Aspect": "RTX3050筆電GPU", "Category": "GRAPHICS#GENERAL", "Opinion": "可惜", "VA": "4.00#5.25"}}, {{"Aspect": "RTX4050筆電GPU", "Category": "GRAPHICS#GENERAL", "Opinion": "更具入手的價值", "VA": "6.33#6.00"}}]}}
    {{"ID": "7094394:S006", "Text": "16吋i9-14900HXRTX5000Ada行動工作站(ZBookFuryG11/A5SA2PA/64G/2TSSD/W11P/3年保固)價格快20萬...這規格、效能...真的不好說......", "Quadruplet": [{{"Aspect": "規格", "Category": "LAPTOP#DESIGN_FEATURES", "Opinion": "真的不好說", "VA": "4.30#5.50"}}, {{"Aspect": "效能", "Category": "LAPTOP#OPERATION_PERFORMANCE", "Opinion": "真的不好說", "VA": "4.30#5.50"}}, {{"Aspect": "價格快20萬", "Category": "LAPTOP#PRICE", "Opinion": "真的不好說", "VA": "4.25#5.38"}}]}}
    {{"ID": "6764642:S003", "Text": "這台筆電太舊顯卡跟內顯都不支援硬解1080PH264CPU軟解720P應該沒問題，1080P建議還是自己試試......", "Quadruplet": [{{"Aspect": "筆電", "Category": "LAPTOP#GENERAL", "Opinion": "太舊", "VA": "3.88#4.50"}}, {{"Aspect": "內顯", "Category": "GRAPHICS#OPERATION_PERFORMANCE", "Opinion": "不支援硬解1080PH264", "VA": "4.00#5.00"}}, {{"Aspect": "顯卡", "Category": "GRAPHICS#OPERATION_PERFORMANCE", "Opinion": "不支援硬解1080PH264", "VA": "4.00#5.00"}}, {{"Aspect": "CPU軟解720P", "Category": "CPU#OPERATION_PERFORMANCE", "Opinion": "應該沒問題", "VA": "5.75#5.12"}}]}}
    {{"ID": "7022796:S020", "Text": "雷蛇系列讚啦，有些產品真的超讚的", "Quadruplet": [{{"Aspect": "雷蛇系列", "Category": "LAPTOP#GENERAL", "Opinion": "讚", "VA": "6.75#6.50"}}, {{"Aspect": "產品", "Category": "LAPTOP#GENERAL", "Opinion": "真的超讚", "VA": "7.83#8.17"}}]}}
    {{"ID": "7050139:S078", "Text": "這台筆電不錯，貴有一部分是配win11pro的關係，但第二SSD要用2230規格就比較妥協了一點，螢幕規格差了點，尤其和宣傳不一樣真的不太好，雖然大家都知道用這種筆電的在公司都是外接螢幕，筆電本身的螢幕都是外出或開會拿來頂著用，14吋多一個USBtypeA是很好的，雖然大概率會拿來接RJ45，商務機有多I/O就是好算是一台合適的商務機，續航夠重量輕效能可以，要的就是這些，其它就是錦上添花，user覺得普普但企業MIS會很歡迎", "Quadruplet": [{{"Aspect": "筆電", "Category": "LAPTOP#GENERAL", "Opinion": "不錯", "VA": "6.25#4.62"}}, {{"Aspect": "規格", "Category": "LAPTOP#DESIGN_FEATURES", "Opinion": "妥協了一點", "VA": "4.50#5.50"}}, {{"Aspect": "螢幕規格", "Category": "DISPLAY#DESIGN_FEATURES", "Opinion": "差了點", "VA": "4.12#5.12"}}, {{"Aspect": "續航", "Category": "LAPTOP#OPERATION_PERFORMANCE", "Opinion": "夠", "VA": "5.75#4.50"}}, {{"Aspect": "重量", "Category": "LAPTOP#PORTABILITY", "Opinion": "輕", "VA": "6.00#3.75"}}, {{"Aspect": "效能", "Category": "LAPTOP#OPERATION_PERFORMANCE", "Opinion": "可以", "VA": "5.83#4.00"}}, {{"Aspect": "win11pro", "Category": "OS#PRICE", "Opinion": "貴", "VA": "3.50#4.88"}}]}}
        
    **範例(golden valence example)**
    {{"ID": "6696296:S130", "Text": "希望配色能大膽一些，畢竟還是有不少女性使用者", "Quadruplet": [{{"Aspect": "配色", "Category": "LAPTOP#DESIGN_FEATURES", "Opinion": "能大膽一些", "VA": "5.00#4.83"}}]}}
    {{"ID": "6880567:S035", "Text": "入手Intel版本一切都好不知各位是否有推的合適DockingStation?", "Quadruplet": [{{"Aspect": "Intel版本", "Category": "LAPTOP#GENERAL", "Opinion": "一切都好", "VA": "6.50#6.10"}}]}}
    {{"ID": "6860419:S008", "Text": "效能的話，推薦購買Alpha17這一台，我也是買這一台，電腦使用上，順暢度很棒，螢幕2k還不錯，散熱很棒，操一段時間，觸摸的體感溫度不燙（對我來說）：", "Quadruplet": [{{"Aspect": "效能", "Category": "LAPTOP#OPERATION_PERFORMANCE", "Opinion": "推薦購買Alpha17", "VA": "6.38#6.25"}}, {{"Aspect": "順暢度", "Category": "LAPTOP#OPERATION_PERFORMANCE", "Opinion": "很棒", "VA": "6.75#6.50"}}, {{"Aspect": "螢幕", "Category": "DISPLAY#GENERAL", "Opinion": "還不錯", "VA": "6.38#5.75"}}, {{"Aspect": "散熱", "Category": "FANS_COOLING#GENERAL", "Opinion": "很棒", "VA": "6.75#6.50"}}, {{"Aspect": "體感溫度", "Category": "FANS_COOLING#OPERATION_PERFORMANCE", "Opinion": "不燙", "VA": "6.00#5.00"}}]}}
    {{"ID": "6944546:S001", "Text": "我很多年前買過Razer的產品後就再也不買了品質差到不行就只賣個牌子價格有7成是買裡面那張logo貼紙東西便宜也就算了還死貴比中國的三無電子產品還垃圾,起碼人家超便宜Razer就是電子垃圾", "Quadruplet": [{{"Aspect": "品質", "Category": "LAPTOP#QUALITY", "Opinion": "差到不行", "VA": "3.00#7.00"}}, {{"Aspect": "Razer的產品", "Category": "LAPTOP#PRICE", "Opinion": "死貴", "VA": "2.33#7.33"}}, {{"Aspect": "Razer", "Category": "LAPTOP#GENERAL", "Opinion": "電子垃圾", "VA": "2.25#7.38"}}, {{"Aspect": "Razer的產品", "Category": "LAPTOP#GENERAL", "Opinion": "比中國的三無電子產品還垃圾", "VA": "2.00#7.83"}}]}}
    {{"ID": "6967676:S051", "Text": "這台筆電真的是太棒了~阿思的整體性能與處理效能根本100%完美", "Quadruplet": [{{"Aspect": "整體性能與", "Category": "LAPTOP#OPERATION_PERFORMANCE", "Opinion": "根本100%完美", "VA": "8.00#7.88"}}, {{"Aspect": "筆電", "Category": "LAPTOP#GENERAL", "Opinion": "真的是太棒了", "VA": "7.38#7.62"}}, {{"Aspect": "處理效能", "Category": "LAPTOP#OPERATION_PERFORMANCE", "Opinion": "根本100%完美", "VA": "8.00#7.88"}}]}}


    **請以純粹的 JSON 格式回應，且僅包含 JSON 列表，勿加入任何解釋或額外的文字。**
    **回應格式 (JSON Array):**
    [
      {{"Aspect": "...", "Opinion": "...", "Category": "...", "Valence": 4.5, "Arousal": 3.0}},
      ...
    ]
    """
    return prompt

# --- 4. Gemini 呼叫和推论函数 ---

def run_inference(args, inference_data, output_filename, limit=None):
    output_data_quadra = []
    
    try:
        client = genai.Client()
    except Exception as e:
        print(f"Error initializing Gemini Client: {e}", file=sys.stderr)
        print("Please ensure your API key is set correctly in your environment (GEMINI_API_KEY).", file=sys.stderr)
        return

    model_name = 'gemini-2.5-flash'
    print(f"Starting inference with model: {model_name}...")

    for i, data in enumerate(inference_data):
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
            # TODO: 請確認 GEMINI_API_KEY 環境變數已設定
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            gemini_response_text = response.text

            # 3. 解析模型回應
            quadruplets_raw = []
            try:
                json_text = gemini_response_text.strip()
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
            print(f"Formatted Output: {json.dumps(formatted_quadra, ensure_ascii=False)}")

        except Exception as e:
            print(f"Critical error during API call for ID {data_id}: {e}", file=sys.stderr)
            output_data_quadra.append(format_quadruplet_output(data_id, []))

        # 稍微等待一下，避免速率限制問題
        # time.sleep(3) 
            
    # 5. 保存結果
    output_dir = os.path.join(args.output_path, "subtask_3")
    os.makedirs(output_dir, exist_ok=True)
    out_put_file_task3_name = os.path.join(output_dir, output_filename)
    
    if limit is not None:
        out_put_file_task3_name = os.path.join(output_dir, f"test_limit_{limit}_{output_filename}")

    print(f"\nWriting results to {out_put_file_task3_name}...")
    with open(out_put_file_task3_name, 'w', encoding='utf-8') as f:
        for item in output_data_quadra:
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + '\n')
            
    print(f"Inference completed. Results saved to {out_put_file_task3_name}")

# --- 5. 主程序入口 ---

if __name__ == '__main__':
    args = parser_getting()

    # 顯式設置與筆電相關的檔名
    args.infer_data = "zho_laptop_dev_task3.jsonl"
    output_filename = "pred_zho_laptop.jsonl"
    args.output_path = "./tasks/" 

    # --- 设置测试限制 ---
    test_limit = 500 
    
    inference_data = load_inference_data(args)
        
    if not inference_data:
        print("No data loaded. Exiting.", file=sys.stderr)
        sys.exit(0)
    
    run_inference(args, inference_data, output_filename, limit=test_limit)