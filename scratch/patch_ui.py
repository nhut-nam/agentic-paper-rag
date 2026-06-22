import os

def main():
    ui_path = "app/ui.py"
    
    # Read the file with errors ignore/replace if needed, but since it has emoji we read utf-8
    with open(ui_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Define targeted replacements
    replacements = [
        (
            """    except Exception as e:# Initialize Streamlit Session States
if "messages" not in st.session_state:""",
            """    except Exception as e:
        return None, str(e)

# Initialize Streamlit Session States
if "messages" not in st.session_state:"""
        ),
        (
            """if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None
if "active_doc_name" not in st.session_state:
    st.session_state.active_doc_name = None""",
            """if "active_doc_ids" not in st.session_state:
    st.session_state.active_doc_ids = []
if "active_doc_names" not in st.session_state:
    st.session_state.active_doc_names = []"""
        ),
        (
            """        selected_pdf_path = st.selectbox(
            "Chọn tài liệu hoạt động",
            options=list(pdf_options.keys()),
            format_func=lambda x: pdf_options[x]
        )
        
        if selected_pdf_path:
            db_doc = path_to_doc.get(selected_pdf_path)
            if not db_doc:
                # File not registered on server yet. Click to register
                if st.button("Đăng ký tệp này lên API Server", use_container_width=True):
                    with st.spinner("Đang đăng ký tệp..."):
                        res_upload = api_upload_pdf(selected_pdf_path)
                        if res_upload and res_upload.get("status") == "uploaded":
                            st.success("Đăng ký thành công!")
                            st.rerun()
            else:
                st.session_state.active_doc_id = db_doc["doc_id"]
                st.session_state.active_doc_name = os.path.basename(db_doc["path"])
                
                # Render status & ingestion control
                doc_id = db_doc["doc_id"]
                status = db_doc["status"]
                
                st.markdown(f"**Tài liệu hiện tại:** `{st.session_state.active_doc_name}`")
                
                # Dynamic badge status
                if status in ["chunked", "completed"]:
                    st.markdown('Trạng thái: <span class="badge-completed">Đã Ingest & Phân Mảnh</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'Trạng thái: <span class="badge-pending">{status}</span>', unsafe_allow_html=True)
                
                # Trigger Ingestion Pipeline
                ingest_label = "🚀 Bắt đầu Ingest Tài Liệu" if status not in ["chunked", "completed"] else "🔄 Ingest Lại Tài Liệu (Cập nhật vector)"
                if st.button(ingest_label, use_container_width=True):
                    res_ingest = api_trigger_ingest(doc_id)
                    if res_ingest and "started" in res_ingest.get("status", ""):
                        # Start polling status
                        status_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        
                        for percent in range(1, 101):
                            # Fetch latest status from api
                            latest_doc = api_get_status(doc_id)
                            current_status = latest_doc["status"] if latest_doc else "unknown"
                            
                            status_placeholder.info(f"Đang xử lý Ingest... (Trạng thái hiện tại: {current_status})")
                            progress_bar.progress(percent)
                            
                            if current_status in ["chunked", "completed"]:
                                st.success("Ingest và đánh chỉ mục tài liệu hoàn tất!")
                                time.sleep(1)
                                st.rerun()
                                break
                            time.sleep(1.5)
                        else:
                            st.warning("Quá trình Ingest đang chạy ngầm trên server. Vui lòng F5 trang sau ít phút.")
                            st.rerun()""",
            """        if "active_pdf_paths" not in st.session_state:
            st.session_state.active_pdf_paths = []

        selected_pdf_paths = st.multiselect(
            "Chọn tài liệu hoạt động",
            options=list(pdf_options.keys()),
            format_func=lambda x: pdf_options[x],
            default=st.session_state.active_pdf_paths
        )
        st.session_state.active_pdf_paths = selected_pdf_paths
        
        if selected_pdf_paths:
            active_ids = []
            active_names = []
            unregistered_paths = []
            pending_ingest_docs = []
            
            for path in selected_pdf_paths:
                db_doc = path_to_doc.get(path)
                if not db_doc:
                    unregistered_paths.append(path)
                else:
                    active_ids.append(db_doc["doc_id"])
                    active_names.append(os.path.basename(db_doc["path"]))
                    if db_doc["status"] not in ["chunked", "completed"]:
                        pending_ingest_docs.append(db_doc)
                        
            st.session_state.active_doc_ids = active_ids
            st.session_state.active_doc_names = active_names
            
            if unregistered_paths:
                st.warning(f"Có {len(unregistered_paths)} tệp chưa đăng ký trên API server.")
                if st.button("Đăng ký tất cả các tệp này", use_container_width=True):
                    with st.spinner("Đang đăng ký các tệp tin..."):
                        for u_path in unregistered_paths:
                            api_upload_pdf(u_path)
                        st.success("Đăng ký thành công!")
                        time.sleep(0.5)
                        st.rerun()
            
            st.markdown("**Trạng thái các tài liệu đã chọn:**")
            for path in selected_pdf_paths:
                db_doc = path_to_doc.get(path)
                if db_doc:
                    status = db_doc["status"]
                    filename = os.path.basename(path)
                    if status in ["chunked", "completed"]:
                        st.markdown(f"✅ `{filename}`: <span class='badge-completed'>Đã Ingest</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"⏳ `{filename}`: <span class='badge-pending'>{status}</span>", unsafe_allow_html=True)
            
            if pending_ingest_docs:
                if st.button("🚀 Ingest các tài liệu chưa xử lý", use_container_width=True):
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    
                    for doc in pending_ingest_docs:
                        api_trigger_ingest(doc["doc_id"])
                        
                    for percent in range(1, 101):
                        all_done = True
                        statuses = []
                        for doc in pending_ingest_docs:
                            latest_doc = api_get_status(doc["doc_id"])
                            current_status = latest_doc["status"] if latest_doc else "unknown"
                            statuses.append(f"{os.path.basename(doc['path'])}: {current_status}")
                            if current_status not in ["chunked", "completed"]:
                                all_done = False
                                
                        status_placeholder.info("Đang xử lý Ingest...\\n" + "\\n".join(statuses))
                        progress_bar.progress(percent)
                        
                        if all_done:
                            st.success("Tất cả tài liệu đã Ingest hoàn tất!")
                            time.sleep(1)
                            st.rerun()
                            break
                        time.sleep(1.5)
                    else:
                        st.warning("Quá trình Ingest đang chạy ngầm. Vui lòng F5 trang sau ít phút.")
                        st.rerun()
            else:
                if st.button("🔄 Ingest lại tất cả tài liệu", use_container_width=True):
                    for path in selected_pdf_paths:
                        db_doc = path_to_doc.get(path)
                        if db_doc:
                            api_trigger_ingest(db_doc["doc_id"])
                    st.success("Đã yêu cầu Ingest lại!")
                    time.sleep(0.5)
                    st.rerun()
        else:
            st.session_state.active_doc_ids = []
            st.session_state.active_doc_names = []"""
        ),
        (
            """if st.session_state.active_doc_id:
    st.info(f"📍 **Đang kết nối tài liệu:** `{st.session_state.active_doc_name}` (ID: `{st.session_state.active_doc_id[:8]}...`)")
else:
    st.warning("⚠️ Chưa chọn tài liệu. Agent sẽ trả lời dựa trên kiến thức toàn cục hoặc web search.")""",
            """if st.session_state.get("active_doc_ids"):
    names_str = ", ".join([f"`{name}`" for name in st.session_state.active_doc_names])
    st.info(f"📍 **Đang kết nối ({len(st.session_state.active_doc_ids)} tài liệu):** {names_str}")
else:
    st.warning("⚠️ Chưa chọn tài liệu. Agent sẽ trả lời dựa trên kiến thức toàn cục hoặc web search.")"""
        ),
        (
            """                payload = {"query": query_to_run}
                if st.session_state.active_doc_id:
                    payload["doc_id"] = st.session_state.active_doc_id
                    
                res = requests.post(f"{BASE_URL}/agent/query", params=payload)""",
            """                payload = {"query": query_to_run}
                if st.session_state.get("active_doc_ids"):
                    payload["doc_ids"] = ",".join(st.session_state.active_doc_ids)
                    
                res = requests.post(f"{BASE_URL}/agent/query", params=payload)"""
        )
    ]

    for target, replacement in replacements:
        # Standardize line endings just in case
        target_norm = target.replace("\r\n", "\n")
        replacement_norm = replacement.replace("\r\n", "\n")
        content_norm = content.replace("\r\n", "\n")
        
        if target_norm in content_norm:
            content_norm = content_norm.replace(target_norm, replacement_norm)
            print("Successfully patched a block!")
        else:
            print("Target block not found. Checking alternate line endings...")
            # Try exact match as-is
            if target in content:
                content = content.replace(target, replacement)
                print("Successfully patched exact block!")
            else:
                print("Failed to match block:")
                print(target[:100] + "...")
        content = content_norm

    with open(ui_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Patching of ui.py completed!")

if __name__ == "__main__":
    main()
