import os

def main():
    ui_path = "app/ui.py"
    with open(ui_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # 1. Sửa lỗi nháy kép HTML class
    content = content.replace('class="badge-completed"', "class='badge-completed'")
    content = content.replace('class="badge-pending"', "class='badge-pending'")
    
    # 2. Sửa lỗi dính dòng comment ở khối except nếu chưa được sửa
    target_except = "except Exception as e:# Initialize Streamlit Session States\nif \"messages\" not in st.session_state:"
    replacement_except = "except Exception as e:\n        return None, str(e)\n\n# Initialize Streamlit Session States\nif \"messages\" not in st.session_state:"
    if target_except in content:
        content = content.replace(target_except, replacement_except)
        print("Sửa lỗi thụt lề thành công!")

    # 3. Ghi lại file
    with open(ui_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Sửa đổi quotes trong ui.py thành công!")

if __name__ == "__main__":
    main()
