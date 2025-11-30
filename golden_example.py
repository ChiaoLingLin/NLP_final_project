import pandas as pd
import json
import numpy as np
import os
from typing import Dict, Any, Tuple, List, Set

# --- 設定 ---
INPUT_FILE = './dataset/zho_laptop_train_alltasks.jsonl'
VALENCE_OUTPUT_FILE = 'valence_stats_strict.jsonl' # 輸出檔案名稱變更以區分
AROUSAL_OUTPUT_FILE = 'arousal_stats_strict.jsonl' # 輸出檔案名稱變更以區分
PERCENTILES = [0, 25, 50, 75, 100] # Min, Q1, Median, Q3, Max

# --- 1. 數據提取與緩存函數 ---
def extract_all_data(file_name: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    讀取 JSONL 檔案，返回扁平化數據和原始完整記錄的緩存。
    
    返回: (DataFrame of flattened quadruplets, Dict mapping ID to full original record)
    """
    if not os.path.exists(file_name):
        print(f"錯誤：輸入檔案 '{file_name}' 不存在。")
        return pd.DataFrame(), {}

    records_list = []
    record_cache = {} # 用於儲存完整的原始記錄 (ID, Text, Quadruplet list)
    
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                record_id = data.get('ID')
                
                # 儲存完整的原始記錄到緩存中
                record_cache[record_id] = data
                
                # 扁平化處理每個四元組，用於後續的分數排序和查找
                for quad in data.get('Quadruplet', []):
                    va_str = quad.get('VA')
                    if va_str and '#' in va_str:
                        valence, arousal = map(float, va_str.split('#'))
                        
                        records_list.append({
                            'ID': record_id,
                            'Valence': valence,
                            'Arousal': arousal
                        })
            
            except json.JSONDecodeError:
                continue
            
    print(f"成功提取 {len(records_list)} 筆四元組，並緩存 {len(record_cache)} 筆原始記錄。")
    return pd.DataFrame(records_list), record_cache

# --- 2. 統計提取與輸出函數 ---
def get_percentile_examples_and_output(df: pd.DataFrame, record_cache: Dict[str, Dict[str, Any]], score_type: str, output_file: str):
    """
    計算指定分數類型 Min, Q1, Median, Q3, Max，提取對應記錄的 ID，並將完整的原始記錄寫入 JSONL。
    """
    
    if df.empty:
        print(f"DataFrame ({score_type}) 為空，無法進行統計分析。")
        return

    # 1. 計算百分位數的數值
    scores = df[score_type].values
    percentile_values = np.percentile(scores, PERCENTILES)
    
    # 2. 依分數類型排序 DataFrame
    df_sorted = df.sort_values(by=score_type, ascending=True).reset_index(drop=True)
    
    selected_ids: Set[str] = set() # 儲存最終選定的完整記錄 ID (避免重複)
    selected_score_values: Set[float] = set() # 儲存已選分數，防止同分數重複

    # 3. 遍歷 Min, Q1, Median, Q3, Max 進行數據點提取
    for idx, p in enumerate(PERCENTILES):
        target_score = percentile_values[idx]
        
        # 尋找第一個分數大於或等於目標分數的索引
        if p == 0: # Minimum
            selected_record = df_sorted.iloc[0]
        elif p == 100: # Maximum
            selected_record = df_sorted.iloc[-1]
        else: # Q1, Median, Q3
            try:
                # 找到第一個 score >= target_score 的數據點索引
                first_match_index = np.argmax(df_sorted[score_type].values >= target_score)
                selected_record = df_sorted.iloc[first_match_index]
            except:
                continue

        current_score_value = selected_record[score_type]
        current_id = selected_record['ID']

        # 檢查是否已提取過相同分數 (確保每個統計點的分數值是唯一的)
        if current_score_value in selected_score_values:
            continue
        
        selected_score_values.add(current_score_value)
        selected_ids.add(current_id) # 儲存該分數對應的原始記錄 ID

    # 4. 根據選定的 ID 獲取完整的原始記錄
    final_records: List[Dict[str, Any]] = [
        record_cache[rid] 
        for rid in selected_ids 
        if rid in record_cache
    ]

    # 5. 寫入 JSONL 檔案 (保持原始格式)
    print(f"\n開始寫入 {score_type} 統計範例 (共 {len(final_records)} 筆，已去重) 至 {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for record in final_records:
                # 嚴格保持原始資料型式：{"ID": "...", "Text": "...", "Quadruplet": [...]}
                json_line = json.dumps(record, ensure_ascii=False)
                outfile.write(json_line + '\n')
        print(f"成功儲存至 {output_file}。")
    except Exception as e:
        print(f"寫入檔案時發生錯誤: {e}")


# --- Main Execution ---
def main():
    """主程式執行流程。"""
    
    # 1. 提取並緩存數據
    df_flattened, record_cache = extract_all_data(INPUT_FILE)
    
    if df_flattened.empty or not record_cache:
        print("無法進行分析。")
        return
        
    # 2. 處理 Valence 分數
    get_percentile_examples_and_output(df_flattened, record_cache, 'Valence', VALENCE_OUTPUT_FILE)

    # 3. 處理 Arousal 分數
    get_percentile_examples_and_output(df_flattened, record_cache, 'Arousal', AROUSAL_OUTPUT_FILE)

if __name__ == '__main__':
    main()