import re

def main():
    path = "app/ui.py"
    with open(path, "rb") as f:
        data = f.read()
        
    text = data.decode("utf-8", errors="ignore")
    
    # Let's fix the specific corrupted block:
    # "st.rerun()i tệp lên API server thành công!"
    # We will search for any trailing junk after st.rerun() in that block
    cleaned = re.sub(r"st\.rerun\(\)[^\n]*i tệp lên API server thành công!", "st.rerun()", text)
    cleaned = re.sub(r"st\.rerun\(\)[^\n\w]*", "st.rerun()\n", cleaned)
    
    # Also clean any other weird character
    cleaned = cleaned.replace("", "")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(cleaned)
        
    print("Cleaned app/ui.py successfully!")

if __name__ == "__main__":
    main()
