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
from streamlit_image_comparison import image_comparison # Import for slider

# --- Configuration & Page Setup ---
st.set_page_config(layout="wide", page_title="Image SEO Optimizer")

# --- Helper Functions ---
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

# --- Main App UI ---
st.title("Image SEO Optimizer")
# st.write(f"**Debug Info:** Python Default Encoding: `{sys.getdefaultencoding()}`, Filesystem Encoding: `{sys.getfilesystemencoding()}`")
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
processed_data = []; skipped_files = []; processing_errors = []

def format_size(size_bytes):
    if size_bytes is None or size_bytes < 0: return "N/A"
    if size_bytes == 0: return "0 B"
    size_kb = size_bytes / 1024
    if size_kb < 1024: return f"{size_kb:.1f} KB"
    size_mb = size_kb / 1024; return f"{size_mb:.1f} MB"

if not api_key: st.warning("🚨 Enter OpenAI API key in sidebar.")
elif not target_keyword: st.warning("🎯 Enter target keyword in sidebar.")
elif not image_sources_input: st.info("➕ Upload images or provide URLs.")
elif total_images > 20: st.error(f"❌ Max 20 images ({total_images} provided).")
else:
    if st.button("✨ Optimize Images", type="primary"):
        if not target_keyword: st.error("❌ Target keyword cannot be empty."); st.stop()

        client = OpenAI(api_key=api_key)
        st.header("⏳ Processing Images...")
        progress_bar = st.progress(0, text="Initializing...")
        start_time = time.time(); optimized_filenames_set = set()

        for idx, source in enumerate(image_sources_input):
            progress_text = f"Processing image {idx + 1}/{total_images}..."; progress_bar.progress((idx + 1)/total_images, text=progress_text)
            image, original_filename, error, original_image_data = get_image_from_source(source)
            source_identifier = source if isinstance(source, str) else source.name

            if error: skipped_files.append(f"{source_identifier} (Load Error)"); processing_errors.append(error); continue
            if not image or not original_filename or original_image_data is None:
                 skipped_files.append(f"{source_identifier} (Load Fail - Missing Data)")
                 processing_errors.append(f"Failed load or missing data: {source_identifier}"); continue

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
                try:
                    if isinstance(target_keyword, str):
                        ascii_encoded_keyword = target_keyword.encode('ascii', 'ignore'); sanitized_keyword_for_api = ascii_encoded_keyword.decode('ascii')
                        if sanitized_keyword_for_api != target_keyword: print(f"Console: Keyword sanitized for API img {idx+1}.")
                    else: sanitized_keyword_for_api = str(target_keyword)
                except Exception as e: print(f"Console: Keyword sanitize err: {e}.")

                prompt_text_for_api = f"""
Analyze the image for its most dominant and specific visual elements. Generate an SEO-optimized file name (NO extension) and alt text.
Target Keyword: '{sanitized_keyword_for_api}'

Guidelines:
- **Filename:** Create a **concise and descriptive name** using 3-5 core keywords/concepts from the image. **MUST separate EVERY word with a hyphen (-).** Prioritize the most unique visual details. Example: `black-casement-window-living-room`, `hand-opening-fiberglass-window`. Do NOT include a file extension.
- **Alt Text:** Natural, descriptive sentence (< 125 chars). Include target keyword naturally. Describe key elements and context.

Output ONLY in this exact JSON format (no extra text or markdown):
{{
  "base_filename": "your-concise-hyphenated-base-name",
  "alt_text": "Your descriptive alt text sentence."
}}
"""
                final_sanitized_prompt_text = prompt_text_for_api
                try:
                    encoded_prompt = prompt_text_for_api.encode('ascii', 'ignore'); final_sanitized_prompt_text = encoded_prompt.decode('ascii')
                    if final_sanitized_prompt_text != prompt_text_for_api: print(f"Console: Full prompt sanitized for API img {idx+1}.")
                except Exception as e_prompt_sanitize: print(f"Console: Full prompt sanitize err: {e_prompt_sanitize}.")

                messages = [{"role": "user", "content": [{"type": "text", "text": final_sanitized_prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}]

                model_to_use = "gpt-4.1" # Or "gpt-4-turbo"
                print(f"Console Log: Attempting API call img {idx+1} (Model: {model_to_use})")
                try:
                    response = client.chat.completions.create(model=model_to_use, messages=messages, max_tokens=150, temperature=0.2)
                    output = response.choices[0].message.content.strip()

                    if output.startswith("```json"): output = output.strip("```json").strip("```").strip()
                    elif output.startswith("```"): output = output.strip("```").strip()
                    try:
                        result = json.loads(output)
                        base_filename_from_api = result.get("base_filename", f"optimized-image-{idx+1}")
                        alt_text = result.get("alt_text", f"Image related to {target_keyword}")
                    except json.JSONDecodeError as json_e:
                        st.warning(f"⚠️ OpenAI response format unexpected for image {idx+1}. Using defaults. (Error: {json_e})")
                        print(f"Console: JSON Parse Err img {idx+1}. Raw output snippet: {output[:100]}...")
                        processing_errors.append(f"API JSON Parse Err {original_filename}: {json_e}")
                        base_filename_from_api = f"api-parse-error-{idx+1}"
                        alt_text = f"Image related to {target_keyword} (API parse error)"
                except Exception as api_e:
                    error_detail = str(api_e)
                    if "invalid_request_error" in error_detail.lower() and "model does not exist" in error_detail.lower():
                         st.error(f"❌ OpenAI API error: Model '{model_to_use}' not found. Try 'gpt-4-turbo'.")
                         processing_errors.append(f"OpenAI API Err {original_filename}: Invalid Model '{model_to_use}'")
                    elif isinstance(api_e, UnicodeEncodeError): error_detail += (f" Still getting UnicodeEncodeError!")
                    else: st.error(f"⚠️ OpenAI API error img {idx+1} ({original_filename}): {error_detail}. Defaults used.")
                    if not ("invalid_request_error" in error_detail.lower() and "model does not exist" in error_detail.lower()):
                        processing_errors.append(f"OpenAI API Err {original_filename}: {error_detail}")
                    base_filename_from_api = f"api-error-{idx+1}"
                    alt_text = f"Image related to {target_keyword} (API error)"

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

                processed_data.append({
                    "original_filename": original_filename, "optimized_filename": unique_filename,
                    "alt_text": alt_text, "original_size_bytes": original_size_bytes,
                    "compressed_size_bytes": compressed_size_bytes, "savings_percentage": savings_percentage,
                    "image_url": source if isinstance(source, str) else None,
                    "compressed_data": compressed_image_data, "pil_image": image
                })
            except Exception as process_e:
                 skipped_files.append(f"{source_identifier} (Processing Error)"); processing_errors.append(f"Error processing {original_filename}: {process_e}")
                 print(f"--- Error processing {original_filename} ---"); traceback.print_exc(); print("--- End Error ---"); continue
            time.sleep(0.7)

        progress_bar.empty(); end_time = time.time()
        st.success(f"✅ Optimization done: {len(processed_data)}/{total_images} images in {end_time - start_time:.2f}s!")

        # --- Results Display Section ---
        if processed_data:
            st.header("📊 Results & Downloads")
            display_df_data = [{"Original Filename": item["original_filename"], "Optimized Filename": item["optimized_filename"],
                                "Alt Text": item["alt_text"], "Original Size": format_size(item["original_size_bytes"]),
                                "Compressed Size": format_size(item["compressed_size_bytes"]),
                                "Savings (%)": f"{item['savings_percentage']:.1f}%" if item['original_size_bytes'] > 0 else "N/A",
                                "Original Source": item["image_url"] if item["image_url"] else "Uploaded"} for item in processed_data]
            if display_df_data: st.dataframe(pd.DataFrame(display_df_data))
            else: st.info("No data for table.")

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if display_df_data: csv = pd.DataFrame(display_df_data).to_csv(index=False).encode('utf-8'); st.download_button("📥 CSV Summary", csv, 'image_seo_summary.csv', 'text/csv', key='csv_download')
            with col_dl2:
                if processed_data:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer,'w',zipfile.ZIP_DEFLATED) as zf:
                        for item in processed_data: zf.writestr(item["optimized_filename"], item["compressed_data"])
                    st.download_button("📦 Optimized Images (.zip)", zip_buffer.getvalue(), 'optimized_images.zip', 'application/zip', key='zip_download')

            # --- Modified Image Preview Section ---
            st.header("🖼️ Optimized Image Preview & Comparison")
            if processed_data:
                for i_disp, item in enumerate(processed_data):
                    with st.expander(f"Image {i_disp+1}: {item.get('optimized_filename', 'N/A')}", expanded=False):

                        # --- Display Text Info FIRST ---
                        st.markdown(f"**Optimized Filename:** `{item.get('optimized_filename', 'N/A')}`")
                        original_fsize = format_size(item.get("original_size_bytes"))
                        compressed_fsize = format_size(item.get("compressed_size_bytes"))
                        savings_fpercent = f"{item.get('savings_percentage', 0.0):.1f}%" if item.get('original_size_bytes', 0) > 0 else "N/A"
                        st.markdown(f"**Size:** {original_fsize} -> {compressed_fsize} (**Savings: {savings_fpercent}**)")
                        st.markdown(f"**Alt Text:**"); st.text_area("Generated Alt Text", value=item.get('alt_text', 'N/A'), height=75, key=f"alt_{i_disp}", disabled=True)
                        st.markdown("---") # Add a separator
                        # --- End Text Info ---

                        # --- Display Comparison Slider AFTER Text ---
                        if 'pil_image' in item and 'compressed_data' in item and item['compressed_data']:
                            try:
                                compressed_pil_image = Image.open(BytesIO(item["compressed_data"]))
                                st.markdown("**Compression Comparison (Drag Slider):**")
                                # --- Use image_comparison component - REMOVED width ---
                                image_comparison(
                                    img1=item["pil_image"],
                                    img2=compressed_pil_image,
                                    label1="Original",
                                    label2=f"WebP Q{compression_quality}",
                                    # width=600, # Relying on responsive width
                                    starting_position=50,
                                    show_labels=True,
                                    make_responsive=True,
                                    in_memory=True
                                )
                            except Exception as img_load_err:
                                st.warning(f"Could not display comparison slider for image {i_disp+1}: {img_load_err}")
                                # Fallback: show static original image if slider fails
                                st.markdown("**Original Image (Preview):**")
                                st.image(item["pil_image"], width=300)
                        else:
                            st.warning("Comparison slider data missing.")
                            # Optionally show static original if slider fails/data missing
                            if 'pil_image' in item:
                                st.markdown("**Original Image (Preview):**")
                                st.image(item["pil_image"], width=300)

            else:
                st.info("No images processed to preview.")
            # --- End Modified Image Preview Section ---

        if skipped_files:
            st.header("⚠️ Skipped / Errored Items")
            for i, file_info in enumerate(skipped_files):
                st.warning(f"- {file_info}")
                if i < len(processing_errors): st.error(f"  Details: {processing_errors[i]}", icon="🐛")

st.markdown("---"); st.markdown("Built with ❤️ using Streamlit & OpenAI")
