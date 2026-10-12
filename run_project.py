import requests
import torch
import os
import time
from transformers import AutoTokenizer, BertForSequenceClassification

# ==========================================
# ⚙️ 設定區
# ==========================================
OLLAMA_MODEL_NAME = "hf.co/jimmm1212/my_expert_llama_v4"
BERT_MODEL_PATH = r"C:\Users\奶茶\Desktop\My_Project\perfect_bert_model_v4"
OUTPUT_TXT_FILE = "result.txt"

# 你的 10 大專屬分類 (對應 0~9)
CLASS_NAMES = [
    "Aggressive or Violent Behavior",  # 0
    "Communication Gesture",  # 1
    "Daily Personal Care",  # 2
    "Emotional Expression",  # 3
    "Friendly Social Interaction",  # 4
    "General Object Interaction",  # 5
    "Medical Emergency or Pain",  # 6
    "Physical Exercise or Sport",  # 7
    "Physiological Symptom",  # 8
    "Work or Study Activity"  # 9
]

# 🌟 新增：機器人對應這 10 種狀態的關懷語句輸出 (順序必須與 CLASS_NAMES 一致)
RESPONSE_MESSAGES = [
    "請保持冷靜，深呼吸，暴力無法解決問題，我們慢慢說。",  # 0: Aggressive or Violent Behavior
    "我收到你的信號了，隨時準備好聆聽你的想法。",  # 1: Communication Gesture
    "把自己照顧好是最重要的小事，你看起來很有精神喔！",  # 2: Daily Personal Care
    "無論開心或難過，我都會在這裡靜靜陪著你。",  # 3: Emotional Expression
    "看到你與人相處這麼融洽，我也覺得心裡很溫暖呢！",  # 4: Friendly Social Interaction
    "需要幫忙拿嗎？操作物品時請小心手部安全喔。",  # 5: General Object Interaction
    "偵測到異常狀況！請保持冷靜，我正在幫你聯繫緊急支援！",  # 6: Medical Emergency or Pain
    "充滿活力的感覺真棒！動一動身體，心情也會變好呢！",  # 7: Physical Exercise or Sport
    "身體是不是有點累了？別硬撐，喝口水休息一下吧。",  # 8: Physiological Symptom
    "你專注的樣子真帥氣，但也要記得適時讓眼睛休息一下喔。"  # 9: Work or Study Activity
]


# ==========================================
# 🧠 階段一：Llama 3 情境推論
# ==========================================
def get_llama_feedback(action_name):
    print(f"\n[階段一] 正在詢問 Llama 3 關於動作 ({action_name}) 的對話回饋...")

    prompt_text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
what the feedback action if a person is making an action of ({action_name}) on dialog

### Response:
"""

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "stop": ["<|end_of_text|>", "### Instruction:"]
        }
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        llama_output = response.json().get("response", "").strip()
        print(f"✅ Llama 3 推論結果:\n{llama_output}\n")
        return llama_output
    except Exception as e:
        print(f"❌ 呼叫 Ollama 失敗！錯誤: {e}")
        return ""


# ==========================================
# 🧠 階段二：載入 BERT 模型與分類
# ==========================================
print("⏳ 正在載入 BERT 分類模型，請稍候...")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH, num_labels=len(CLASS_NAMES))
bert_model.eval()
print("✅ BERT 模型載入完成！")


def classify_with_bert(analyzed_text):
    print("[階段二] 正在將 Llama 的推論結果交給 BERT 進行預測...")
    inputs = tokenizer(analyzed_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()

    return predicted_class_id


# ==========================================
# 🚀 專題主程式測試 (雙模式)
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🔥 動作分析雙模型系統已啟動 (V4 滿分常駐版) 🔥")
    print(f"📝 每次的機器人關懷語句將會自動記錄到: {OUTPUT_TXT_FILE}")
    print("=" * 50)

    # 系統啟動時，先清空建立一個全新的 result.txt 檔案
    with open(OUTPUT_TXT_FILE, "w", encoding="utf-8") as f:
        pass

    input_counter = 1

    mode = input("\n👉 請選擇模式 [1] 手動輸入 [2] 從檔案即時監聽: ").strip()

    if mode == '2':
        file_name = input(f"👉 請輸入檔案名稱 (直接按 Enter 預設為 '輸入模板.txt'): ").strip()
        if not file_name:
            file_name = "輸入模板.txt"

        if not os.path.exists(file_name):
            print(f"\n❌ 找不到檔案 '{file_name}'，請確認檔案是否有放在同一個資料夾中！")
        else:
            print(f"\n📂 成功開啟檔案 '{file_name}'！現在進入「後端即時監聽模式」👀")
            print("💡 (提示：你可以隨時打開 txt 檔加入新動作並存檔，程式會自動抓取！若要離開請按 Ctrl+C)\n")

            with open(file_name, "r", encoding="utf-8") as f:
                while True:
                    line = f.readline()

                    if not line:
                        time.sleep(1)
                        continue

                    user_action = line.strip()
                    if not user_action:
                        continue

                    print(f"\n▶️ 正在處理第 {input_counter} 個動作: {user_action}")

                    llama_feedback = get_llama_feedback(user_action)

                    if llama_feedback:
                        predicted_id = classify_with_bert(llama_feedback)
                        final_result_name = CLASS_NAMES[predicted_id]

                        # 🌟 取得對應的中文關懷句子
                        final_message = RESPONSE_MESSAGES[predicted_id]

                        print("=" * 50)
                        print(f"🏆 【BERT 最終判定類別】: {predicted_id} ({final_result_name})")
                        print(f"💬 【機器人對話輸出】: {final_message}")
                        print("=" * 50)

                        # 🌟 將最終關懷句子寫入 TXT
                        with open(OUTPUT_TXT_FILE, "a", encoding="utf-8") as f_out:
                            f_out.write(f"{input_counter}\n{final_message}\n")

                        print(f"💾 (已將對話結果儲存至 {OUTPUT_TXT_FILE})")
                        input_counter += 1

    else:
        print("\n⌨️ 進入手動輸入模式！")
        while True:
            user_action = input(f"\n👉 請輸入第 {input_counter} 個動作 (輸入 'q' 離開): ")
            if user_action.lower() == 'q':
                break

            llama_feedback = get_llama_feedback(user_action)

            if llama_feedback:
                predicted_id = classify_with_bert(llama_feedback)
                final_result_name = CLASS_NAMES[predicted_id]

                # 🌟 取得對應的中文關懷句子
                final_message = RESPONSE_MESSAGES[predicted_id]

                print("=" * 50)
                print(f"🏆 【BERT 最終判定類別】: {predicted_id} ({final_result_name})")
                print(f"💬 【機器人對話輸出】: {final_message}")
                print("=" * 50)

                # 🌟 將最終關懷句子寫入 TXT
                with open(OUTPUT_TXT_FILE, "a", encoding="utf-8") as f_out:
                    f_out.write(f"{input_counter}\n{final_message}\n")

                print(f"💾 (已將對話結果儲存至 {OUTPUT_TXT_FILE})")
                input_counter += 1