import streamlit as st
import pandas as pd
import requests
import base64
import re
import os
from io import BytesIO
from PIL import Image
from openai import OpenAI
import json
import time
import zipfile
import traceback
import sys
# Ensure streamlit-image-comparison is installed (added to requirements.txt)
try:
    from streamlit_image_comparison import image_comparison
    image_comparison_available = True
except ModuleNotFoundError:
    image_comparison_available = False # Handle if not installed

# --- Configuration & Page Setup ---
st.set_page_config(layout="wide", page_title="Image SEO Optimizer")

# --- Helper Functions (remain the same) ---
def sanitize_filename(filename):
    sanitized = filename.lower().replace(' ', '-')
    sanitized = re.sub(r'[^a-z0-9\-]', '', sanitized)
    sanitized = re.sub(r'-+', '-', sanitized)
    sanitized = sanitized.strip('-')
    return sanitized

def truncate_filename(filename, max_length=80):
    if len(filename) <= max_length: return filename
    base_name, extension = os.path.splitext(filename)
    available_length = max_length - len(extension); truncated_base = base_name[:available_length]
    truncated_base = truncated_base.rstrip('-'); return f"{truncated_base}{extension}"

def get_image_from_source(source):
    image = None; error = None; original_filename = None; original_data = None
    is_url = isinstance(source, str)
    try:
        if is_url:
            response = requests.get(source, timeout=15); response.raise_for_status()
            if 'image' not in response.headers.get('Content-Type', '').lower(): raise ValueError("URL not valid image type.")
            original_data = response.content
            image = Image.open(BytesIO(original_data)); original_filename = source.split('/')[-1].split('?')[0]
        else:
            original_data = source.getvalue()
            source.seek(0)
            if source.type not in ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif', 'image/bmp', 'image/avif']: raise ValueError(f"Unsupported format: {source.type}")
            image = Image.open(source); original_filename = source.name
    except requests.exceptions.RequestException as e: error = f"Error fetching URL {source}: {e}"
    except ValueError as e: error = f"Error processing {'URL' if is_url else 'file'} {source if is_url else source.name}: {e}"
    except Exception as e: error = f"Unexpected error processing {'URL' if is_url else 'file'} {source if is_url else source.name}: {e}"
    return image, original_filename, error, original_data

# --- Format Size Helper Function ---
def format_size(size_bytes):
    if size_bytes is None or size_bytes < 0: return "N/A"
    if size_bytes == 0: return "0 B"
    size_kb = size_bytes / 1024
    if size_kb < 1024: return f"{size_kb:.1f} KB"
    size_mb = size_kb / 1024; return f"{size_mb:.1f} MB"

# --- Main App UI ---
st.title("Image SEO Optimizer")
st.write(f"Streamlit Version (for debugging): {st.__version__}") # Keep this!
st.write("**Author:** Brandon Lazovic")
st.markdown("""Optimize images for SEO: AI filenames/alt text, project numbers, WebP compression, zip download.""")

with st.expander("SEO Best Practices for Images"):
    st.markdown("""- **Alt Text:** Concise, descriptive, keyword-rich, context-relevant, < 125 chars.
- **File Names:** Short, descriptive, hyphen-separated, keyword-rich. Use `.webp`, `.jpg`, `.png`.
- **Compression:** Balance quality/size. WebP often best for web. Savings vary based on original.
- **Context & Responsiveness:** Place near relevant text, use `srcset`.""")

st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("1. OpenAI API Key", type="password", help="Required for AI generation.")
target_keyword = st.sidebar.text_input("2. Target Keyword", help="Primary keyword for optimization.")
project_number = st.sidebar.text_input("3. Project Number (Optional)", help="Appended to filenames (e.g., image-slug-PN123.webp).")
compression_quality = st.sidebar.slider("4. WebP Compression Quality", 1, 100, 80, help="Adjust WebP quality (0-100). Higher = better quality, larger file.")

sanitized_project_number = ""
if project_number:
    sanitized_project_number = re.sub(r'[^a-zA-Z0-9\-]', '', project_number).strip('-')
    if project_number != sanitized_project_number: st.sidebar.warning(f"PN sanitized: `{sanitized_project_number}`")
    if not sanitized_project_number: st.sidebar.warning("Invalid PN after sanitization, not used.")

st.header("🖼️ Image Input (Max 20)")
col1, col2 = st.columns(2)
with col1: uploaded_files = st.file_uploader("Upload Images", type=['png','jpg','jpeg','webp','gif','bmp','avif'], accept_multiple_files=True, help="Max 20 images.")
with col2: image_urls_input = st.text_area("Or Enter Image URLs (one per line)", height=150, help="Direct image URLs.")

image_sources_input = []
if uploaded_files: image_sources_input.extend(uploaded_files)
if image_urls_input: urls = [url.strip() for url in image_urls_input.strip().split('\n') if url.strip()]; image_sources_input.extend(urls)

total_images = len(image_sources_input)

# Initialize Session State
if 'processed_data' not in st.session_state: st.session_state.processed_data = []
# Removed skipped/errors from state, handle directly
if 'compare_index' not in st.session_state: st.session_state.compare_index = None


if not api_key: st.warning("🚨 Enter OpenAI API key in sidebar.")
elif not target_keyword: st.warning("🎯 Enter target keyword in sidebar.")
elif not image_sources_input: st.info("➕ Upload images or provide URLs.")
elif total_images > 20: st.error(f"❌ Max 20 images ({total_images} provided).")
else:
    if st.button("✨ Optimize Images", type="primary"):
        st.session_state.processed_data = [] # Clear previous results
        st.session_state.compare_index = None # Reset comparison view

        if not target_keyword: st.error("❌ Target keyword cannot be empty."); st.stop()

        client = OpenAI(api_key=api_key)
        st.header("⏳ Processing Images...")
        progress_bar = st.progress(0, text="Initializing...")
        start_time = time.time(); optimized_filenames_set = set()

        # Use lists to collect results and errors during processing
        temp_processed_data = []
        temp_skipped_files = []
        temp_processing_errors = []


        for idx, source in enumerate(image_sources_input):
            progress_text = f"Processing image {idx + 1}/{total_images}..."; progress_bar.progress((idx + 1)/total_images, text=progress_text)
            image, original_filename, error, original_image_data = get_image_from_source(source)
            source_identifier = source if isinstance(source, str) else source.name

            if error: temp_skipped_files.append(f"{source_identifier} (Load Error)"); temp_processing_errors.append(error); continue
            if not image or not original_filename or original_image_data is None:
                 temp_skipped_files.append(f"{source_identifier} (Load Fail - Missing Data)")
                 temp_processing_errors.append(f"Failed load or missing data: {source_identifier}"); continue

            try:
                original_size_bytes = len(original_image_data) if original_image_data else 0
                compressed_buffer = BytesIO()
                image.save(compressed_buffer, format="WEBP", quality=compression_quality, lossless=False)
                compressed_image_data = compressed_buffer.getvalue()
                compressed_size_bytes = len(compressed_image_data)
                savings_percentage = ((original_size_bytes - compressed_size_bytes) / original_size_bytes) * 100 if original_size_bytes > 0 else 0.0

                openai_image_buffer = BytesIO(); image.save(openai_image_buffer, format="PNG")
                base64_image = base64.b64encode(openai_image_buffer.getvalue()).decode('utf-8')

                sanitized_keyword_for_api = target_keyword
                # Add keyword sanitization back if needed for API encoding issues
                # try: ... except ...

                # --- Stricter Prompt for JSON ---
                prompt_text_for_api = f"""
Analyze the image for its most dominant and specific visual elements.
Target Keyword for context: '{sanitized_keyword_for_api}'

Your task is to generate ONLY a single, valid JSON object containing two keys: "base_filename" and "alt_text".
- **base_filename:** A concise, descriptive name using 3-5 core keywords from the image. Separate EVERY word with a hyphen (-). Prioritize unique visual details. Example: `black-casement-window-living-room`. Do NOT include a file extension.
- **alt_text:** A natural, descriptive sentence (< 125 chars). Include the target keyword naturally if relevant to the image. Describe key visual elements and context.

**IMPORTANT: Output ONLY the JSON object below, with no other text, explanations, or markdown.**
```json
{{
  "base_filename": "your-concise-hyphenated-base-name",
  "alt_text": "Your descriptive alt text sentence."
}}```
"""
                final_sanitized_prompt_text = prompt_text_for_api # Assume prompt encoding is ok now

                messages = [{"role": "user", "content": [{"type": "text", "text": final_sanitized_prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}]
                model_to_use = "gpt-4.1" # Or "gpt-4-turbo"
                print(f"Console Log: Attempting API call img {idx+1} (Model: {model_to_use})")
                base_filename_from_api = f"api-error-{idx+1}" # Default in case of complete failure
                alt_text = f"Image related to {target_keyword} (API error)" # Default

                try:
                    response = client.chat.completions.create(model=model_to_use, messages=messages, max_tokens=200, temperature=0.2)
                    output = response.choices[0].message.content.strip()

                    if output.startswith("```json"): output = output[len("```json"):].strip()
                    if output.endswith("```"): output = output[:-len("```")].strip()
                    try:
                        result = json.loads(output)
                        base_filename_from_api = result.get("base_filename", f"optimized-image-{idx+1}")
                        alt_text = result.get("alt_text", f"Image related to {target_keyword}")
                    except json.JSONDecodeError as json_e:
                        st.warning(f"⚠️ OpenAI response format unexpected for image {idx+1}. Using defaults. (Check Console Log)")
                        print(f"Console: JSON Parse Err img {idx+1}. Error: {json_e}. Raw output: '{output}'")
                        base_filename_from_api = f"api-parse-error-{idx+1}"
                        alt_text = f"Image related to {target_keyword} (API parse error)"
                        temp_processing_errors.append(f"API JSON Parse Err {original_filename}: {json_e}") # Log specific parse error

                except Exception as api_e:
                    error_detail = str(api_e)
                    st.error(f"⚠️ OpenAI API error img {idx+1} ({original_filename}): {error_detail}. Defaults used.")
                    # Use default base_filename/alt_text defined above
                    temp_processing_errors.append(f"OpenAI API Err {original_filename}: {error_detail}") # Log API error


                sanitized_base_name = sanitize_filename(base_filename_from_api)
                filename_core = sanitized_base_name
                if sanitized_project_number: filename_core += f"-{sanitized_project_number}"
                final_filename_with_ext = f"{filename_core}.webp"
                final_filename_with_ext = truncate_filename(final_filename_with_ext)
                unique_filename = final_filename_with_ext; counter = 1
                while unique_filename in optimized_filenames_set:
                    core_name_no_ext, ext = os.path.splitext(final_filename_with_ext)
                    if counter > 1 and core_name_no_ext.endswith(f'-{counter-1}'): core_name_no_ext = core_name_no_ext[:-len(f'-{counter-1}')]
                    unique_filename = f"{core_name_no_ext}-{counter}{ext}";
                    counter += 1
                optimized_filenames_set.add(unique_filename)

                temp_processed_data.append({
                    "status": "success", "original_filename": original_filename, "optimized_filename": unique_filename,
                    "alt_text": alt_text, "original_size_bytes": original_size_bytes,
                    "compressed_size_bytes": compressed_size_bytes, "savings_percentage": savings_percentage,
                    "image_url": source if isinstance(source, str) else None,
                    "compressed_data": compressed_image_data, "original_data": original_image_data,
                })
            except Exception as process_e:
                 temp_skipped_files.append(f"{source_identifier} (Processing Error)")
                 temp_processing_errors.append(f"Error processing {original_filename if 'original_filename' in locals() else source_identifier}: {process_e}")
                 print(f"--- Error processing {source_identifier} ---"); traceback.print_exc(); print("--- End Error ---"); continue
            time.sleep(0.7)

        progress_bar.empty(); end_time = time.time()
        # Store results in session state AFTER loop
        st.session_state.processed_data = temp_processed_data
        # We don't need skipped/errors in session state if we display them directly
        # st.session_state.skipped_files = temp_skipped_files
        # st.session_state.processing_errors = temp_processing_errors

        st.success(f"✅ Optimization complete: {len(temp_processed_data)} items processed in {end_time - start_time:.2f}s!")

        # Display skipped/errors immediately after processing
        if temp_skipped_files:
            st.header("⚠️ Skipped / Processing Errors")
            for i, file_info in enumerate(temp_skipped_files):
                st.warning(f"- {file_info}")
                if i < len(temp_processing_errors): st.error(f"  Details: {temp_processing_errors[i]}", icon="🐛")

        st.rerun() # Rerun to display results from session state


# --- Results Display Section (Reads from Session State) ---
if st.session_state.processed_data:
    processed_data_display = [item for item in st.session_state.processed_data if item.get('status') == 'success']
    # Error display moved to after processing run

    if processed_data_display:
        st.header("📊 Results & Downloads")
        display_df_data = [{"Original Filename": item["original_filename"], "Optimized Filename": item["optimized_filename"],
                            "Alt Text": item["alt_text"], "Original Size": format_size(item["original_size_bytes"]),
                            "Compressed Size": format_size(item["compressed_size_bytes"]),
                            "Savings (%)": f"{item['savings_percentage']:.1f}%" if item.get('original_size_bytes', 0) > 0 else "N/A",
                            "Original Source": item["image_url"] if item["image_url"] else "Uploaded"} for item in processed_data_display]
        if display_df_data: st.dataframe(pd.DataFrame(display_df_data))
        else: st.info("No successful results to display in table.")

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1: # CSV Download
            if display_df_data: csv = pd.DataFrame(display_df_data).to_csv(index=False).encode('utf-8'); st.download_button("📥 CSV Summary", csv, 'image_seo_summary.csv', 'text/csv', key='csv_download')
        with col_dl2: # ZIP Download
            if processed_data_display:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer,'w',zipfile.ZIP_DEFLATED) as zf:
                    for item in processed_data_display: zf.writestr(item["optimized_filename"], item["compressed_data"])
                st.download_button("📦 Optimized Images (.zip)", zip_buffer.getvalue(), 'optimized_images.zip', 'application/zip', key='zip_download')

        st.header("🖼️ Optimized Image Details")
        for i_disp, item in enumerate(processed_data_display):
            with st.expander(f"Image {i_disp+1}: {item.get('optimized_filename', 'N/A')}", expanded=False):
                st.markdown(f"**Optimized Filename:** `{item.get('optimized_filename', 'N/A')}`")
                original_fsize = format_size(item.get("original_size_bytes"))
                compressed_fsize = format_size(item.get("compressed_size_bytes"))
                savings_fpercent = f"{item.get('savings_percentage', 0.0):.1f}%" if item.get('original_size_bytes', 0) > 0 else "N/A"
                st.markdown(f"**Size:** {original_fsize} -> {compressed_fsize} (**Savings: {savings_fpercent}**)")
                st.markdown(f"**Alt Text:**"); st.text_area("Generated Alt Text", value=item.get('alt_text', 'N/A'), height=75, key=f"alt_{i_disp}_{item['original_filename']}", disabled=True)

                if 'compressed_data' in item and item['compressed_data']:
                    st.image(BytesIO(item["compressed_data"]), caption="Compressed Preview", width=250)
                st.markdown("---")
                compare_key = f"compare_btn_{i_disp}_{item['original_filename']}"
                # Only show compare button if component is available and data exists
                if image_comparison_available and 'original_data' in item and item['original_data'] and 'compressed_data' in item and item['compressed_data']:
                    if st.button("Compare Original vs. Compressed", key=compare_key):
                        st.session_state.compare_index = i_disp
                        st.rerun() # Rerun to show the comparison container
                elif not image_comparison_available:
                     st.caption("Comparison slider requires 'streamlit-image-comparison' library.")


    # --- Comparison Container Logic (Alternative to st.dialog) ---
    if st.session_state.get('compare_index') is not None:
        idx_to_compare = st.session_state.compare_index
        successful_processed_data = [item for item in st.session_state.processed_data if item.get('status') == 'success']

        if 0 <= idx_to_compare < len(successful_processed_data):
            item_to_compare = successful_processed_data[idx_to_compare]

            # Create a container to hold the comparison view
            with st.container(border=True): # Added border for visibility
                 st.subheader(f"Comparing: {item_to_compare.get('original_filename', 'Original Image')}")
                 try:
                    original_img_compare = Image.open(BytesIO(item_to_compare["original_data"]))
                    compressed_img_compare = Image.open(BytesIO(item_to_compare["compressed_data"]))

                    if image_comparison_available:
                        image_comparison(
                            img1=original_img_compare,
                            img2=compressed_img_compare,
                            label1="Original",
                            label2=f"WebP Q{compression_quality}",
                            width=700, # Explicit width might work better here than in expander
                            starting_position=50,
                            show_labels=True,
                            make_responsive=True,
                            in_memory=True
                        )
                    else:
                        # Fallback if component not installed
                        col1_comp, col2_comp = st.columns(2)
                        with col1_comp: st.image(original_img_compare, caption="Original")
                        with col2_comp: st.image(compressed_img_compare, caption=f"WebP Q{compression_quality}")

                    if st.button("Close Comparison", key=f"close_compare_{idx_to_compare}"):
                        st.session_state.compare_index = None # Reset the index
                        st.rerun() # Rerun to hide the container

                 except Exception as compare_err:
                     st.error(f"Could not load images for comparison: {compare_err}")
                     if st.button("Close Error", key=f"close_err_compare_{idx_to_compare}"):
                          st.session_state.compare_index = None
                          st.rerun()
        else:
             print(f"Error: compare_index {idx_to_compare} out of bounds for successful data.")
             st.session_state.compare_index = None # Reset invalid index


st.markdown("---"); st.markdown("Built with ❤️ using Streamlit & OpenAI")
