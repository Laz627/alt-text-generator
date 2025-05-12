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
import traceback # For detailed error logging in console
import sys # Added to check encoding

# --- Configuration & Page Setup ---
st.set_page_config(layout="wide", page_title="Image SEO Optimizer")

# --- Helper Functions ---
def sanitize_filename(filename):
    """Removes invalid characters and replaces spaces with hyphens."""
    sanitized = re.sub(r'[^a-zA-Z0-9\-\.]', '', filename.replace(' ', '-').lower())
    sanitized = sanitized.strip('-.')
    sanitized = re.sub(r'-+', '-', sanitized)
    return sanitized

def truncate_filename(filename, max_length=100):
    """Truncates filename if it exceeds max_length, preserving the extension."""
    if len(filename) <= max_length:
        return filename
    base_name, extension = os.path.splitext(filename)
    available_length = max_length - len(extension)
    truncated_base = base_name[:available_length]
    truncated_base = truncated_base.rstrip('-')
    return f"{truncated_base}{extension}"

def get_image_from_source(source):
    """Loads a PIL Image from either an uploaded file or a URL."""
    image = None; error = None; original_filename = None; is_url = isinstance(source, str)
    try:
        if is_url:
            response = requests.get(source, timeout=15); response.raise_for_status()
            if 'image' not in response.headers.get('Content-Type', '').lower(): raise ValueError("URL not valid image type.")
            image = Image.open(BytesIO(response.content)); original_filename = source.split('/')[-1].split('?')
        else:
            if source.type not in ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif', 'image/bmp', 'image/avif']: raise ValueError(f"Unsupported format: {source.type}")
            image = Image.open(source); original_filename = source.name
    except requests.exceptions.RequestException as e: error = f"Error fetching URL {source}: {e}"
    except ValueError as e: error = f"Error processing {'URL' if is_url else 'file'} {source if is_url else source.name}: {e}"
    except Exception as e: error = f"Unexpected error processing {'URL' if is_url else 'file'} {source if is_url else source.name}: {e}"
    return image, original_filename, error

# --- Main App UI ---
st.title("Image SEO Optimizer") # Removed emoji
# Print system encoding info
st.write(f"**Debug Info:** Python Default Encoding: `{sys.getdefaultencoding()}`, Filesystem Encoding: `{sys.getfilesystemencoding()}`")
st.write("**Author:** Brandon Lazovic")
st.markdown("""Welcome! Optimize images for SEO: AI filenames/alt text, project numbers, compression, zip download.""") # Shorter intro

with st.expander("💡 ricorda: SEO Best Practices for Images"):
    st.markdown("""- **Alt Text:** Concise, descriptive, keyword-rich (naturally), context-relevant, < 125 chars. Accessibility/Search.
- **File Names:** Short, descriptive, hyphen-separated, keyword-rich (naturally). Avoid generic names. Use standard types.
- **Compression:** Balance quality/size (JPEG photos, PNG graphics). Faster loads.
- **Context:** Place near relevant text. Use captions.
- **Responsiveness:** `srcset` / `<picture>` for sizes.
- **Structured Data:** Schema markup (`ImageObject`) for rich results.""") # More concise

st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("1. OpenAI API Key", type="password", help="Required for AI generation.")
target_keyword = st.sidebar.text_input("2. Target Keyword", help="Primary keyword for optimization.")
project_number = st.sidebar.text_input("3. Project Number (Optional)", help="Appended to filenames (e.g., image-slug-PN123.jpg).")
compression_quality = st.sidebar.slider("4. Compression Quality (JPEG Output)", 1, 100, 85, help="Adjust JPEG quality (higher = larger file).")

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
            image, original_filename, error = get_image_from_source(source)
            source_identifier = source if isinstance(source, str) else source.name

            if error: skipped_files.append(f"{source_identifier} (Load Error)"); processing_errors.append(error); continue
            if not image or not original_filename: skipped_files.append(f"{source_identifier} (Load Fail)"); processing_errors.append(f"Failed load: {source_identifier}"); continue

            try:
                compressed_buffer = BytesIO(); img_for_compression = image.copy()
                if img_for_compression.mode in ('RGBA', 'LA', 'P'): img_for_compression = img_for_compression.convert('RGB')
                img_for_compression.save(compressed_buffer, format="JPEG", quality=compression_quality, optimize=True); compressed_image_data = compressed_buffer.getvalue()

                openai_image_buffer = BytesIO(); image.save(openai_image_buffer, format="PNG")
                base64_image = base64.b64encode(openai_image_buffer.getvalue()).decode('utf-8')

                sanitized_keyword_for_api = target_keyword
                try:
                    if isinstance(target_keyword, str):
                        ascii_encoded_keyword = target_keyword.encode('ascii', 'ignore'); sanitized_keyword_for_api = ascii_encoded_keyword.decode('ascii')
                        if sanitized_keyword_for_api != target_keyword: print(f"Console: Keyword '{target_keyword}' sanitized to '{sanitized_keyword_for_api}' for API img {idx+1}.")
                    else: print(f"Console: Warn - target_keyword not string img {idx+1}."); sanitized_keyword_for_api = str(target_keyword)
                except Exception as e: print(f"Console: Keyword sanitize err: {e}. Using original: '{target_keyword}'")

                prompt_text_for_api = f"""Analyze...\nTarget Keyword: '{sanitized_keyword_for_api}'\nGuidelines...\nOutput JSON...\n{{ "base_filename": "...", "alt_text": "..." }}""" # Simplified for brevity here, use full prompt

                final_sanitized_prompt_text = prompt_text_for_api
                try: # Aggressive sanitization (keep it just in case, though likely not the issue now)
                    encoded_prompt = prompt_text_for_api.encode('ascii', 'ignore'); final_sanitized_prompt_text = encoded_prompt.decode('ascii')
                    if final_sanitized_prompt_text != prompt_text_for_api: print(f"Console: Full prompt sanitized for API img {idx+1}.")
                except Exception as e_prompt_sanitize: print(f"Console: Full prompt sanitize err: {e_prompt_sanitize}.")

                # Debugging removed for brevity, keep if needed

                messages = [{"role": "user", "content": [{"type": "text", "text": final_sanitized_prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}]

                # --- API Call Section Modified ---
                print(f"Console Log: Attempting API call for image {idx+1} WITHOUT response_format parameter.")
                try:
                    # REMOVE response_format parameter for this test
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_tokens=150,
                        temperature=0.1
                        # response_format={"type": "json_object"} # <--- LINE REMOVED
                    )
                    output = response.choices.message.content.strip()

                    # --- BEGIN Manual JSON extraction ---
                    if output.startswith("```json"): output = output.strip("```json").strip("```").strip()
                    elif output.startswith("```"): output = output.strip("```").strip()
                    try:
                        result = json.loads(output)
                        base_filename = result.get("base_filename", f"optimized-image-{idx+1}")
                        alt_text = result.get("alt_text", f"Image related to {target_keyword}")
                    except json.JSONDecodeError as json_e:
                        st.warning(f"⚠️ Could not parse API response for image {idx+1} (manual parse failed): {json_e}. Response: '{output[:100]}...'. Using defaults.")
                        processing_errors.append(f"API JSON Parse Err {original_filename}: {json_e}")
                        base_filename = f"api-parse-error-{idx+1}-{sanitized_project_number if sanitized_project_number else ''}".rstrip('-')
                        alt_text = f"Image related to {target_keyword} (API parse error)"
                    # --- END Manual JSON extraction ---

                except Exception as api_e:
                    error_detail = str(api_e)
                    if isinstance(api_e, UnicodeEncodeError):
                        error_detail += (f" Still getting UnicodeEncodeError! "
                                         f"Orig Keyword: '{target_keyword}', Sanitized: '{sanitized_keyword_for_api}'. "
                                         f"This usually points to an environment issue (check default encoding printed above) or a bug deep in the request library.")
                    elif "response_format" in error_detail.lower(): error_detail += " API trouble generating JSON." # Less likely now
                    st.error(f"⚠️ OpenAI API error img {idx+1} ({original_filename}): {error_detail}. Defaults used.")
                    processing_errors.append(f"OpenAI API Err {original_filename}: {error_detail}")
                    base_filename = f"api-error-{idx+1}-{sanitized_project_number if sanitized_project_number else ''}".rstrip('-')
                    alt_text = f"Image related to {target_keyword} (API error)"
                # --- End API Call Section Modified ---


                final_base_filename = sanitize_filename(base_filename)
                if sanitized_project_number: final_base_filename += f"-{sanitized_project_number}"
                final_filename_with_ext = f"{final_base_filename}.jpg"; final_filename_with_ext = truncate_filename(final_filename_with_ext)

                unique_filename = final_filename_with_ext; counter = 1
                while unique_filename in optimized_filenames_set:
                    name, ext = os.path.splitext(final_filename_with_ext); name_part_to_strip = f"-{counter-1}"
                    if name.endswith(name_part_to_strip): name = name[:-len(name_part_to_strip)]
                    if not name: name = f"duplicate-base-{idx+1}"
                    unique_filename = f"{name}-{counter}{ext}"; counter += 1
                optimized_filenames_set.add(unique_filename)

                processed_data.append({"original_filename": original_filename, "optimized_filename": unique_filename, "alt_text": alt_text, "image_url": source if isinstance(source, str) else None, "compressed_data": compressed_image_data, "pil_image": image })
            except Exception as process_e:
                 skipped_files.append(f"{source_identifier} (Processing Error)"); processing_errors.append(f"Error processing {original_filename}: {process_e}")
                 print(f"--- Error processing {original_filename} ---"); traceback.print_exc(); print("--- End Error ---"); continue
            time.sleep(0.5) # Keep small delay

        progress_bar.empty(); end_time = time.time()
        st.success(f"✅ Optimization done: {len(processed_data)}/{total_images} images in {end_time - start_time:.2f}s!")

        if processed_data:
            st.header("📊 Results & Downloads")
            df_data = [{"Original Filename":i["original_filename"],"Optimized Filename":i["optimized_filename"],"Alt Text":i["alt_text"],"Original Source":i["image_url"] if i["image_url"] else "Uploaded"} for i in processed_data]
            if df_data: st.dataframe(pd.DataFrame(df_data))
            else: st.info("No data for table.")

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if df_data: csv = pd.DataFrame(df_data).to_csv(index=False).encode('utf-8'); st.download_button("📥 CSV Summary", csv, 'image_seo_summary.csv', 'text/csv', key='csv_dl')
            with col_dl2:
                if processed_data:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer,'w',zipfile.ZIP_DEFLATED) as zf: [zf.writestr(item["optimized_filename"], item["compressed_data"]) for item in processed_data]
                    st.download_button("📦 Optimized Images (.zip)", zip_buffer.getvalue(), 'optimized_images.zip', 'application/zip', key='zip_dl')

            st.header("🖼️ Optimized Image Preview")
            if processed_data:
                for i_disp, item in enumerate(processed_data):
                    with st.expander(f"Image {i_disp+1}: {item.get('optimized_filename', 'N/A')}", expanded=False):
                        if 'pil_image' in item: st.image(item["pil_image"], caption=f"Optimized: {item.get('optimized_filename', 'N/A')}", width=300)
                        else: st.warning("No preview.")
                        st.markdown(f"**Alt Text:**"); st.text_area("Generated Alt Text", value=item.get('alt_text', 'N/A'), height=75, key=f"alt_{i_disp}", disabled=True)
            else: st.info("No images processed to preview.")

        if skipped_files:
            st.header("⚠️ Skipped / Errored Items")
            for i, file_info in enumerate(skipped_files):
                st.warning(f"- {file_info}")
                if i < len(processing_errors): st.error(f"  Details: {processing_errors[i]}", icon="🐛")

st.markdown("---"); st.markdown("Built with ❤️ using Streamlit & OpenAI") # Keep heart emoji here? Seems less likely to cause issues than rocket.
