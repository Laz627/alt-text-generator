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

# Ensure streamlit-image-comparison is installed (added to requirements.txt)
try:
    from streamlit_image_comparison import image_comparison
    image_comparison_available = True
except ModuleNotFoundError:
    image_comparison_available = False # Handle if not installed

# --- Configuration & Page Setup ---
st.set_page_config(layout="wide", page_title="Image SEO Optimizer")

# --- Callback Functions for Editable Fields ---
def update_bulk_item(index, field_key_prefix):
    """Callback to update an item in the bulk processed data list."""
    widget_key = f"{field_key_prefix}_{index}"
    if widget_key in st.session_state:
        st.session_state.bulk_processed_data[index][field_key_prefix] = st.session_state[widget_key]

def update_single_item(field_to_update, widget_key):
    """Callback to update a field in the single processed item dictionary."""
    if widget_key in st.session_state and st.session_state.single_processed_item:
        st.session_state.single_processed_item[field_to_update] = st.session_state[widget_key]

# --- Helper Functions (Unchanged) ---
def sanitize_filename(filename):
    """Sanitizes a string to be a valid filename."""
    sanitized = filename.lower().replace(' ', '-')
    sanitized = re.sub(r'[^a-z0-9\-.]', '', sanitized) # Allow dots for extension
    sanitized = re.sub(r'-+', '-', sanitized)
    sanitized = sanitized.strip('-')
    return sanitized

def truncate_filename(filename, max_length=80):
    """Truncates a filename to a max length, preserving the extension."""
    if len(filename) <= max_length: return filename
    base_name, extension = os.path.splitext(filename)
    available_length = max_length - len(extension)
    truncated_base = base_name[:available_length]
    truncated_base = truncated_base.rstrip('-')
    return f"{truncated_base}{extension}"

def get_image_from_source(source):
    """Fetches and opens an image from a URL or an uploaded file."""
    image, error, original_filename, original_data = None, None, None, None
    is_url = isinstance(source, str)
    try:
        if is_url:
            response = requests.get(source, timeout=15)
            response.raise_for_status()
            if 'image' not in response.headers.get('Content-Type', '').lower():
                raise ValueError("URL does not point to a valid image type.")
            original_data = response.content
            image = Image.open(BytesIO(original_data))
            original_filename = source.split('/')[-1].split('?')[0]
        else: # UploadedFile object
            original_data = source.getvalue()
            source.seek(0)
            if source.type not in ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif', 'image/bmp', 'image/avif']:
                raise ValueError(f"Unsupported file format: {source.type}")
            image = Image.open(source)
            original_filename = source.name
    except requests.exceptions.RequestException as e: error = f"Error fetching URL {source}: {e}"
    except ValueError as e: error = f"Error processing {'URL' if is_url else 'file'} {source if is_url else source.name}: {e}"
    except Exception as e: error = f"An unexpected error occurred with {'URL' if is_url else 'file'} {source if is_url else source.name}: {e}"
    return image, original_filename, error, original_data

def process_image(source, client, config, existing_filenames=None):
    """Processes a single image: fetches, compresses, and gets AI metadata."""
    if existing_filenames is None: existing_filenames = set()

    image, original_filename, error, original_image_data = get_image_from_source(source)
    source_identifier = source if isinstance(source, str) else source.name

    if error: return {"status": "error", "message": error, "original_filename": source_identifier}
    if not image or not original_filename or original_image_data is None: return {"status": "error", "message": "Failed to load image data.", "original_filename": source_identifier}

    try:
        original_size_bytes = len(original_image_data)
        compressed_buffer = BytesIO()
        image.save(compressed_buffer, format="WEBP", quality=config['compression_quality'])
        compressed_image_data = compressed_buffer.getvalue()
        compressed_size_bytes = len(compressed_image_data)
        savings_percentage = ((original_size_bytes - compressed_size_bytes) / original_size_bytes) * 100 if original_size_bytes > 0 else 0.0

        openai_image_buffer = BytesIO()
        image.save(openai_image_buffer, format="PNG")
        base64_image = base64.b64encode(openai_image_buffer.getvalue()).decode('utf-8')

        has_filename_context = config['product_type'] or config['city_geo_target']
        has_alt_text_context = config['service_type'] or config['product_type'] or config['city_geo_target'] or config['additional_context']
        
        filename_instructions = """- Use the **Product Type** as the primary basis for the filename...\n- **GOOD Example:** `lifestyle-series-picture-windows-patio-door-salina-ks`""" if has_filename_context else """- **Analyze the image for its main visual subject**...\n- **GOOD Example:** `tan-siding-bay-window-exterior`"""
        alt_text_instructions = """- Tell a short, descriptive story about the image...\n- **GOOD Example:** "A two-story wall of Pella Lifestyle Series..." """ if has_alt_text_context else """- Identify 2-3 of the most important visual details...\n- **GOOD Example:** "Large picture windows above a three-panel sliding glass door..." """

        prompt_text_for_api = f"""Your task is to analyze the image and generate a single, valid JSON object with `base_filename` and `alt_text`.
**BACKGROUND INFORMATION:**
- Primary Keyword: '{config['keyword']}'
- Service Type: '{config['service_type'] or "Not provided"}'
- Product Type: '{config['product_type'] or "Not provided"}'
- Location: '{config['city_geo_target'] or "Not provided"}'
- Additional Project Context: '{config['additional_context'] or "Not provided"}'
**YOUR INSTRUCTIONS:**
1.  **For `base_filename`**: {filename_instructions}
2.  **For `alt_text`**: {alt_text_instructions}
**IMPORTANT: Output ONLY the final, polished JSON object.**
```json
{{
  "base_filename": "your-descriptive-hyphenated-name",
  "alt_text": "Your natural, human-sounding alt text."
}}```"""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text_for_api}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}]
        
        base_filename_from_api, alt_text, api_error = f"api-error-processing", f"Image related to {config['keyword']}", None
        
        try:
            response = client.chat.completions.create(model="gpt-4.1", messages=messages, max_tokens=200, temperature=0.4, response_format={"type": "json_object"})
            result = json.loads(response.choices[0].message.content.strip())
            base_filename_from_api = result.get("base_filename", f"optimized-image-{original_filename}")
            alt_text = result.get("alt_text", f"Image of {config['keyword']}")
            if base_filename_from_api.endswith(('.webp', '.png', '.jpg')): base_filename_from_api = os.path.splitext(base_filename_from_api)[0]
        except Exception as e: api_error = f"OpenAI API/JSON Err: {e}"

        sanitized_base_name = sanitize_filename(base_filename_from_api)
        filename_core = sanitized_base_name
        if config['sanitized_project_number']: filename_core += f"-{config['sanitized_project_number']}"
        final_filename_with_ext = truncate_filename(f"{filename_core}.webp")

        unique_filename, counter = final_filename_with_ext, 1
        while unique_filename in existing_filenames:
            core_name, ext = os.path.splitext(final_filename_with_ext)
            unique_filename = f"{core_name}-{counter}{ext}"; counter += 1
        
        return {
            "status": "success", "original_filename": original_filename, "optimized_filename": unique_filename,
            "alt_text": alt_text, "original_size_bytes": original_size_bytes, "compressed_size_bytes": compressed_size_bytes, 
            "savings_percentage": savings_percentage, "image_url": source if isinstance(source, str) else None,
            "compressed_data": compressed_image_data, "original_data": original_image_data, "api_error": api_error
        }
    except Exception as e: return {"status": "error", "message": f"Error during processing: {e}", "original_filename": original_filename}

# --- Main App UI ---
st.title("Image SEO Optimizer")
st.markdown("Optimize images for SEO: AI filenames/alt text, WebP compression, and flexible download capabilities. **You can edit filenames and alt text before downloading.**")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("1. OpenAI API Key", type="password", help="Required for AI generation.")
keyword = st.sidebar.text_input("2. Keyword", help="Primary keyword for optimization (e.g., 'home exterior').")
service_type = st.sidebar.text_input("3. Service Type (Optional)", help="e.g., 'window replacement'.")
product_type = st.sidebar.text_input("4. Product Type (Optional)", help="e.g., 'Pella casement windows'.")
city_geo_target = st.sidebar.text_input("5. City / GEO Target (Optional)", help="e.g., 'Topeka, KS'.")
project_number = st.sidebar.text_input("6. Project Number (Optional)", help="Appended to filenames (e.g., image-slug-PN123.webp).")
additional_context = st.sidebar.text_area("7. Additional Context (Optional)", help="Provide unique details.")
compression_quality = st.sidebar.slider("8. WebP Compression Quality", 1, 100, 80, help="Adjust WebP quality.")

sanitized_project_number = ""
if project_number:
    sanitized_project_number = re.sub(r'[^a-zA-Z0-9\-]', '', project_number).strip('-')
    if project_number != sanitized_project_number: st.sidebar.warning(f"PN sanitized: `{sanitized_project_number}`")
    if not sanitized_project_number: st.sidebar.warning("Invalid PN after sanitization.")

config = {"keyword": keyword, "service_type": service_type, "product_type": product_type, "city_geo_target": city_geo_target, "project_number": project_number, "additional_context": additional_context, "compression_quality": compression_quality, "sanitized_project_number": sanitized_project_number}

# --- Initialize Session State ---
if 'bulk_processed_data' not in st.session_state: st.session_state.bulk_processed_data = []
if 'bulk_compare_index' not in st.session_state: st.session_state.bulk_compare_index = None
if 'single_processed_item' not in st.session_state: st.session_state.single_processed_item = None
if 'single_compare_active' not in st.session_state: st.session_state.single_compare_active = False

# --- Tabbed Interface ---
tab1, tab2 = st.tabs(["Bulk Image Optimizer", "Singular Image Optimizer"])

# --- BULK OPTIMIZER TAB ---
with tab1:
    st.header("🖼️ Bulk Image Input (Max 20)")
    col1, col2 = st.columns(2)
    with col1: uploaded_files = st.file_uploader("Upload Images", type=['png','jpg','jpeg','webp'], accept_multiple_files=True, help="Max 20 images.")
    with col2: image_urls_input = st.text_area("Or Enter Image URLs (one per line)", height=150, help="Direct image URLs.")

    image_sources_input = []
    if uploaded_files: image_sources_input.extend(uploaded_files)
    if image_urls_input: image_sources_input.extend([url.strip() for url in image_urls_input.strip().split('\n') if url.strip()])
    total_images = len(image_sources_input)

    if st.button("✨ Optimize All Images", type="primary", key="bulk_optimize_button"):
        if not api_key: st.warning("🚨 Enter OpenAI API key.")
        elif not keyword: st.warning("🎯 Enter a primary Keyword.")
        elif not image_sources_input: st.info("➕ Upload images or provide URLs.")
        elif total_images > 20: st.error(f"❌ Max 20 images ({total_images} provided).")
        else:
            st.session_state.bulk_processed_data, st.session_state.bulk_compare_index = [], None
            client = OpenAI(api_key=api_key)
            progress_bar = st.progress(0, text="Initializing...")
            temp_data, temp_skipped, filenames = [], [], set()

            for idx, source in enumerate(image_sources_input):
                progress_bar.progress((idx + 1)/total_images, text=f"Processing image {idx + 1}/{total_images}...");
                result = process_image(source, client, config, filenames)
                if result['status'] == 'success':
                    filenames.add(result['optimized_filename']); temp_data.append(result)
                    if result.get('api_error'): st.warning(f"⚠️ OpenAI issue for {result['original_filename']}: {result['api_error']}. Defaults used.")
                else: temp_skipped.append(f"{result['original_filename']} ({result['message']})")
                time.sleep(0.5)
            
            progress_bar.empty(); st.session_state.bulk_processed_data = temp_data
            st.success(f"✅ Optimization complete: {len(temp_data)} items processed!")
            if temp_skipped: st.error("⚠️ Some files were skipped: " + ", ".join(temp_skipped))
            st.rerun()

    if st.session_state.bulk_processed_data:
        st.header("📊 Results & Downloads")
        df_data = [{"Original Filename": item["original_filename"], "Optimized Filename": item["optimized_filename"], "Alt Text": item["alt_text"], "Savings": f"{item['savings_percentage']:.1f}%"} for item in st.session_state.bulk_processed_data]
        st.dataframe(pd.DataFrame(df_data))

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = pd.DataFrame(df_data).to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV Summary", csv, 'image_seo_summary.csv', 'text/csv')
        with col_dl2:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for item in st.session_state.bulk_processed_data:
                    zf.writestr(sanitize_filename(item["optimized_filename"]), item["compressed_data"])
            st.download_button("📦 Download Optimized Images (.zip)", zip_buffer.getvalue(), 'optimized_images.zip', 'application/zip')

        st.header("✏️ Edit & Compare Images")
        for i, item in enumerate(st.session_state.bulk_processed_data):
            with st.expander(f"Image {i+1}: {item.get('optimized_filename', 'N/A')}", expanded=False):
                st.text_input(
                    "Optimized Filename", value=item['optimized_filename'], key=f"optimized_filename_{i}",
                    on_change=update_bulk_item, kwargs={'index': i, 'field_key_prefix': 'optimized_filename'},
                    help="Edit the filename here. Changes are saved automatically. Ensure it ends with .webp"
                )
                st.text_area(
                    "Generated Alt Text", value=item['alt_text'], height=75, key=f"alt_text_{i}",
                    on_change=update_bulk_item, kwargs={'index': i, 'field_key_prefix': 'alt_text'},
                    help="Edit the alt text here. Changes are saved automatically."
                )
                if image_comparison_available and item.get('original_data') and item.get('compressed_data'):
                    if st.button("Compare Original vs. Compressed", key=f"compare_btn_{i}"):
                        st.session_state.bulk_compare_index = i; st.rerun()
                elif item.get('compressed_data'):
                    st.image(BytesIO(item["compressed_data"]), caption="Compressed Preview", width=250)

    if st.session_state.get('bulk_compare_index') is not None:
        idx, item = st.session_state.bulk_compare_index, st.session_state.bulk_processed_data[st.session_state.bulk_compare_index]
        with st.container(border=True):
            st.subheader(f"Comparing: {item.get('original_filename', 'Original Image')}")
            image_comparison(img1=Image.open(BytesIO(item["original_data"])), img2=Image.open(BytesIO(item["compressed_data"])), label1="Original", label2=f"WebP Q{compression_quality}")
            if st.button("Close Comparison", key=f"close_compare_{idx}"):
                st.session_state.bulk_compare_index = None; st.rerun()

# --- SINGULAR OPTIMIZER TAB ---
with tab2:
    st.header("🖼️ Single Image Input")
    single_source = st.file_uploader("Upload an Image", type=['png','jpg','jpeg','webp'], key="single_uploader") or st.text_input("Or Enter an Image URL", key="single_url_input").strip()

    if st.button("✨ Optimize Single Image", type="primary", key="single_optimize_button"):
        if not api_key: st.warning("🚨 Enter OpenAI API key.")
        elif not keyword: st.warning("🎯 Enter a primary Keyword.")
        elif not single_source: st.info("➕ Upload an image or provide a URL.")
        else:
            st.session_state.single_processed_item, st.session_state.single_compare_active = None, False
            client = OpenAI(api_key=api_key)
            with st.spinner("Processing your image..."):
                result = process_image(single_source, client, config)
                if result['status'] == 'success':
                    st.session_state.single_processed_item = result; st.success("✅ Optimization Complete!")
                    if result.get('api_error'): st.warning(f"⚠️ OpenAI issue: {result['api_error']}. Defaults were used.")
                else: st.error(f"❌ Error: {result['message']}")
            st.rerun()

    if st.session_state.get('single_processed_item'):
        item = st.session_state.single_processed_item
        st.header("📊 Result & Download")
        col1_res, col2_res = st.columns(2)
        with col1_res:
            st.markdown(f"**Original Filename:** `{item['original_filename']}`")
            st.text_input(
                "Optimized Filename", value=item['optimized_filename'], key="single_filename_input",
                on_change=update_single_item, kwargs={'field_to_update': 'optimized_filename', 'widget_key': 'single_filename_input'},
                help="Edit the filename. Changes are saved automatically. Ensure it ends with .webp"
            )
            st.text_area(
                "Generated Alt Text", value=item['alt_text'], height=100, key="single_alt_text_input",
                on_change=update_single_item, kwargs={'field_to_update': 'alt_text', 'widget_key': 'single_alt_text_input'},
                help="Edit the alt text. Changes are saved automatically."
            )
            st.metric(label="Original Size", value=format_size(item['original_size_bytes']))
            st.metric(label="Compressed Size", value=format_size(item['compressed_size_bytes']), delta=f"-{item['savings_percentage']:.1f}% savings", delta_color="normal")
            st.download_button(
                label="📥 Download Optimized Image (.webp)", data=item['compressed_data'],
                file_name=sanitize_filename(item['optimized_filename']), mime='image/webp', key='single_download_button'
            )
        with col2_res:
            st.subheader("Image Preview")
            compare_active = st.session_state.get('single_compare_active', False)
            if image_comparison_available and not compare_active:
                st.image(BytesIO(item["compressed_data"]), caption="Compressed Preview")
                if st.button("Compare Original vs. Compressed", key="single_compare_btn"):
                    st.session_state.single_compare_active = True; st.rerun()
            else:
                image_comparison(img1=Image.open(BytesIO(item["original_data"])), img2=Image.open(BytesIO(item["compressed_data"])), label1="Original", label2=f"WebP Q{compression_quality}")
                if compare_active and st.button("Close Comparison", key="single_close_compare"):
                    st.session_state.single_compare_active = False; st.rerun()
